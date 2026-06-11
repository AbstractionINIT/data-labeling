"""
ML training dashboard for the from-scratch detector.

A self-contained, auto-refreshing web page (no external/CDN assets — all charts
are hand-rolled SVG so it works fully offline) that shows everything you care
about while annotating + training:

  * Service health (Label Studio + ML backend) and the live model version
  * Retrain progress: annotation events since last train -> next retrain
  * Annotated-image count, predictions stored in Label Studio
  * KPI strip: training runs, best val loss (with trend), data size, device
  * Pause/Resume generating: when paused, the backend stops producing
    predictions, so deleting them in Label Studio won't trigger a regen
  * Charts:
      - Loss curves (train vs val) for any recent run (run selector)
      - Loss components (obj / box / cls) for the selected run
      - Best val loss per run (bar)
      - Boxes per class (horizontal bar)
  * Full run-history table + checkpoints + live training-log tail

Training history is read from data/history.json (written by the trainer). Runs
that predate that file are reconstructed from data/logs/training.log.

Run:
    . .\\scripts\\env.ps1
    .\\scripts\\start_dashboard.ps1        # -> http://localhost:9091
"""
from __future__ import annotations

import json
import os
import re
import statistics
import threading
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, request

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
IMAGES_DIR = BACKEND_DIR / "images"
STATE_PATH = DATA_DIR / "state.json"
HISTORY_PATH = DATA_DIR / "history.json"
PROGRESS_PATH = DATA_DIR / "progress.json"
INFERENCE_PATH = DATA_DIR / "inference.json"
DELETED_PATH = DATA_DIR / "deleted_runs.json"   # run numbers hidden via the dashboard
LOG_PATH = DATA_DIR / "logs" / "training.log"
CKPT_DIR = DATA_DIR / "checkpoints"
PROGRESS_STALE = 25.0   # seconds before a progress section reads as inactive
IMG_INPUT = int(os.getenv("DET_IMG_SIZE", "512"))   # model input size (for recommend)
DEFAULT_INFERENCE = {"sliced": True, "slice": 1024, "overlap": 0.2,
                     "conf": 0.25, "class_conf": {}, "paused": False}

LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8090").rstrip("/")
LS_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
ML_URL = os.getenv("ML_BACKEND_URL", "http://localhost:9090").rstrip("/")
RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY", "25"))
PROJECT_TITLE = os.getenv("PROJECT_TITLE", "Construction Site Detection")
DASH_PORT = int(os.getenv("DASH_PORT", "9091"))
CURVE_RUNS = 12   # number of most-recent runs to ship full per-epoch curves for

app = Flask(__name__)


def _health(url: str) -> bool:
    try:
        return requests.get(url + "/health", timeout=2).status_code == 200
    except Exception:
        return False


def _ls_project():
    """Return a dict of live Label Studio project info (or empty on failure)."""
    if not LS_KEY:
        return {}
    try:
        r = requests.get(f"{LS_URL}/api/projects",
                         headers={"Authorization": f"Token {LS_KEY}"}, timeout=5)
        r.raise_for_status()
        body = r.json()
        items = body.get("results", body if isinstance(body, list) else [])
        proj = next((p for p in items if p.get("title") == PROJECT_TITLE),
                    items[0] if items else None)
        if not proj:
            return {}
        return {
            "id": proj["id"],
            "tasks": proj.get("task_number"),
            "annotated_images": proj.get("num_tasks_with_annotations"),
            "total_annotations": proj.get("total_annotations_number") or proj.get("annotation_count"),
            "total_predictions": proj.get("total_predictions_number"),
            "auto_predict": proj.get("evaluate_predictions_automatically"),
        }
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Training history: structured file first, log fallback for older runs
# --------------------------------------------------------------------------- #
def _load_history_file() -> dict:
    if HISTORY_PATH.exists():
        try:
            runs = json.loads(HISTORY_PATH.read_text(encoding="utf-8")).get("runs", [])
            return {int(r["run"]): r for r in runs}
        except Exception:
            return {}
    return {}


def _load_deleted() -> set:
    """Run numbers the user deleted from the dashboard (hidden everywhere)."""
    if DELETED_PATH.exists():
        try:
            return {int(n) for n in
                    json.loads(DELETED_PATH.read_text(encoding="utf-8")).get("runs", [])}
        except Exception:
            return set()
    return set()


def _save_deleted(runs: set) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DELETED_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"runs": sorted(runs)}, indent=2), encoding="utf-8")
    tmp.replace(DELETED_PATH)


_EP_RE = re.compile(
    r"run#(\d+) ep\s+(\d+)/(\d+) lr=([\deE.+-]+) "
    r"train_loss=([\d.]+) val_loss=([\d.]+) "
    r"obj=([\d.]+) box=([\d.]+) cls=([\d.]+)")


def _parse_log():
    """Reconstruct run records from the training log; also return the live tail."""
    by_run: dict[int, dict] = {}
    tail: list[str] = []
    if not LOG_PATH.exists():
        return by_run, tail
    lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[-80:]

    def rec(n: int) -> dict:
        return by_run.setdefault(n, {
            "run": n, "variant": "?", "device": None, "images": None,
            "train_images": None, "val_images": None, "total_boxes": None,
            "per_class": {}, "epochs": None, "best_val_loss": None,
            "duration_s": None, "checkpoint": None, "source": "log",
            "curve": {"epoch": [], "train_loss": [], "val_loss": [],
                      "obj": [], "box": [], "cls": []},
        })

    # Walk the log in order, attributing each line to the CURRENT run (set by the
    # most recent START). Lines that carry their own run number (epoch/DONE) use
    # that number directly. This avoids mis-attributing lines across runs.
    cur = None
    for ln in lines:
        m = re.search(r"TRAIN RUN #(\d+) START\s+variant=(\S+)\s+device=(.+)", ln)
        if m:
            cur = int(m.group(1)); r = rec(cur)
            r["variant"] = m.group(2); r["device"] = m.group(3).strip()
            # a fresh START (e.g. a re-run) resets this run's transient fields
            r["images"] = r["train_images"] = r["val_images"] = r["total_boxes"] = None
            r["curve"] = {"epoch": [], "train_loss": [], "val_loss": [],
                          "obj": [], "box": [], "cls": []}
            continue
        m = re.search(r"trained on (\d+) images \(train=(\d+), val=(\d+)\) \| (\d+) total boxes", ln)
        if m and cur is not None:
            rec(cur).update(images=int(m.group(1)), train_images=int(m.group(2)),
                            val_images=int(m.group(3)), total_boxes=int(m.group(4)))
            continue
        m = re.search(r"boxes per class:\s*(\{[^}]*\})", ln)
        if m and cur is not None:
            try:
                rec(cur)["per_class"] = json.loads(m.group(1).replace("'", '"'))
            except Exception:
                pass
            continue
        m = _EP_RE.search(ln)
        if m:
            c = rec(int(m.group(1)))["curve"]
            c["epoch"].append(int(m.group(2)))
            c["train_loss"].append(float(m.group(5)))
            c["val_loss"].append(float(m.group(6)))
            c["obj"].append(float(m.group(7)))
            c["box"].append(float(m.group(8)))
            c["cls"].append(float(m.group(9)))
            continue
        m = re.search(r"TRAIN RUN #(\d+) DONE\s+best_val_loss=([\d.]+) @ep(-?\d+) "
                      r"\| (\d+) images \| ([\d.]+)s \| ckpt=(\S+)", ln)
        if m:
            rec(int(m.group(1))).update(
                best_val_loss=float(m.group(2)), best_epoch=int(m.group(3)),
                images=int(m.group(4)), duration_s=float(m.group(5)), checkpoint=m.group(6))
            continue
    # epochs count fallback
    for r in by_run.values():
        if r["epochs"] is None and r["curve"]["epoch"]:
            r["epochs"] = r["curve"]["epoch"][-1] + 1
    return by_run, tail


def _runs():
    """Unified run list: structured history wins, log fills gaps. Returns
    (summaries[all], curves{recent}, per_class[latest], tail)."""
    hist = _load_history_file()
    log_runs, tail = _parse_log()
    merged = dict(log_runs)
    merged.update(hist)  # structured records override log-reconstructed ones
    # Drop runs the user deleted from the dashboard. The `or n in hist` clause is
    # a safety net: if a structured record for a "deleted" number still exists
    # (e.g. a history rewrite failed mid-delete), keep showing it rather than
    # hiding a run whose data is still on disk. Run numbers are monotonic, so a
    # deleted number is never reclaimed by a later retrain.
    deleted = _load_deleted()
    ids = [n for n in sorted(merged) if n not in deleted or n in hist]
    recent = set(ids[-CURVE_RUNS:])

    summary_keys = ("run", "best_val_loss", "images", "train_images", "val_images",
                    "total_boxes", "variant", "device", "epochs", "duration_s",
                    "checkpoint", "started_at", "finished_at", "per_class", "source")
    summaries, curves = [], {}
    for n in ids:
        r = merged[n]
        summaries.append({k: r.get(k) for k in summary_keys})
        if n in recent and r.get("curve") and r["curve"].get("epoch"):
            curves[str(n)] = r["curve"]
    per_class = (merged[ids[-1]].get("per_class") or {}) if ids else {}
    return summaries, curves, per_class, tail


def _inference():
    """Live SAHI config from data/inference.json (falls back to defaults)."""
    cfg = dict(DEFAULT_INFERENCE)
    if INFERENCE_PATH.exists():
        try:
            saved = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
            for k in cfg:
                if k in saved:
                    cfg[k] = saved[k]
        except Exception:
            pass
    return cfg


def _write_inference(updates: dict):
    """Validate + merge + persist inference config; returns the new config."""
    cfg = _inference()
    if "sliced" in updates:
        cfg["sliced"] = bool(updates["sliced"])
    if "slice" in updates:
        cfg["slice"] = max(256, min(8192, int(updates["slice"])))
    if "overlap" in updates:
        cfg["overlap"] = max(0.0, min(0.8, round(float(updates["overlap"]), 3)))
    if "conf" in updates:
        cfg["conf"] = max(0.0, min(1.0, round(float(updates["conf"]), 3)))
    if "class_conf" in updates and isinstance(updates["class_conf"], dict):
        cfg["class_conf"] = {str(k): max(0.0, min(1.0, round(float(v), 3)))
                             for k, v in updates["class_conf"].items()}
    if "paused" in updates:
        cfg["paused"] = bool(updates["paused"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INFERENCE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tmp.replace(INFERENCE_PATH)
    return cfg


def _median_image_size():
    sizes = []
    if IMAGES_DIR.exists():
        from PIL import Image
        for p in sorted(IMAGES_DIR.glob("*"))[:24]:
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
            except Exception:
                pass
    if not sizes:
        return None
    return int(statistics.median(s[0] for s in sizes)), int(statistics.median(s[1] for s in sizes))


def _recommend_slice():
    """Recommend SAHI config from the current images (adapts if they change)."""
    wh = _median_image_size()
    if not wh:
        return None, None
    W, H = wh
    short, longd = min(W, H), max(W, H)
    if longd <= IMG_INPUT * 1.5:           # already near model input -> no slicing
        return {"sliced": False, "slice": DEFAULT_INFERENCE["slice"], "overlap": 0.2}, wh
    # aim for ~2x downscale into the model input: tile ~= half the short side,
    # snapped to a multiple of 128 and clamped to a sane range.
    sl = int(round(short / 2 / 128) * 128)
    sl = max(IMG_INPUT, min(2048, sl))
    return {"sliced": True, "slice": sl, "overlap": 0.2}, wh


def _progress():
    """Live train/infer progress, filtered to recently-updated sections."""
    try:
        d = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    now = time.time()
    out = {}
    for k in ("train", "infer"):
        s = d.get(k)
        if s and (now - s.get("updated", 0) < PROGRESS_STALE):
            s = dict(s); s.pop("updated", None)
            out[k] = s
    return out


def _checkpoints():
    out = []
    if CKPT_DIR.exists():
        for p in sorted(CKPT_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            st = p.stat()
            out.append({"name": p.name, "mb": round(st.st_size / 1e6, 2), "mtime": int(st.st_mtime)})
    return out


@app.route("/api/status")
def status():
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    summaries, curves, per_class, tail = _runs()
    proj = _ls_project()
    events = state.get("events_since_train", 0)
    weights = state.get("active_weights")
    serving = bool(weights and Path(weights).exists())
    latest = summaries[-1] if summaries else {}
    return jsonify({
        "ls_health": _health(LS_URL),
        "ml_health": _health(ML_URL),
        "retrain_every": RETRAIN_EVERY,
        "events_since_train": events,
        "events_remaining": max(RETRAIN_EVERY - events, 0),
        "train_runs": state.get("train_runs", 0),
        "variant": state.get("variant", "?"),
        "device": latest.get("device"),
        "classes": state.get("classes", []),
        "active_weights": weights,
        "serving": serving,
        "model_version": f"scratchdet-r{state.get('train_runs', 0)}" if serving else None,
        "inference": _inference(),
        "last_metrics": state.get("last_metrics", {}),
        "annotated_images": proj.get("annotated_images"),
        "total_annotations": proj.get("total_annotations"),
        "total_predictions": proj.get("total_predictions"),
        "tasks": proj.get("tasks"),
        "auto_predict": proj.get("auto_predict"),
        "project_id": proj.get("id"),
        "progress": _progress(),
        "runs": summaries,
        "curves": curves,
        "per_class": per_class,
        "checkpoints": _checkpoints(),
        "log_tail": "\n".join(tail),
        "ls_url": LS_URL,
        "ml_url": ML_URL,
        "ls_port": LS_URL.rsplit(":", 1)[-1] if ":" in LS_URL else "8090",
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    """Force a retrain now (ignores the every-25 gate). When it finishes, the
    backend auto-pushes the new predictions into Label Studio."""
    proj = _ls_project()
    pid = proj.get("id")
    if not pid:
        return jsonify({"ok": False, "error": "project not found in Label Studio"}), 400

    def work():
        try:
            requests.post(f"{ML_URL}/webhook",
                          json={"action": "START_TRAINING", "project": {"id": pid}},
                          timeout=3600)
        except Exception:
            pass
    threading.Thread(target=work, daemon=True).start()
    return jsonify({"ok": True, "status": "training started"})


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Re-push the current model's predictions into Label Studio (delete stale +
    re-fetch) without retraining."""
    if not LS_KEY:
        return jsonify({"ok": False, "error": "LABEL_STUDIO_API_KEY not set"}), 400
    proj = _ls_project()
    pid = proj.get("id")
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    runs = state.get("train_runs", 0)
    if not pid or runs < 1:
        return jsonify({"ok": False, "error": "no project or no trained model yet"}), 400
    version = f"scratchdet-r{runs}"
    h = {"Authorization": f"Token {LS_KEY}", "Content-Type": "application/json"}
    sel = {"selectedItems": {"all": True, "excluded": []}}

    def work():
        try:
            requests.patch(f"{LS_URL}/api/projects/{pid}", headers=h,
                           json={"model_version": version}, timeout=30)
            requests.post(f"{LS_URL}/api/dm/actions",
                          params={"id": "delete_tasks_predictions", "project": pid},
                          headers=h, json=sel, timeout=300)
            requests.post(f"{LS_URL}/api/dm/actions",
                          params={"id": "retrieve_tasks_predictions", "project": pid},
                          headers=h, json=sel, timeout=900)
        except Exception:
            pass
    threading.Thread(target=work, daemon=True).start()
    return jsonify({"ok": True, "status": f"refreshing to {version}"})


@app.route("/api/run/<int:run_no>/delete", methods=["POST"])
def api_delete_run(run_no):
    """Delete a run's data from the dashboard: remove its history record, hide it
    from the (append-only) training log, and delete its checkpoint file. The
    currently-served model cannot be deleted. The raw training.log is left intact
    as an audit trail."""
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    # Normalise separators before taking the basename so a Windows-written path
    # (backslashes) still resolves correctly if the dashboard reads it on POSIX.
    active_name = os.path.basename((state.get("active_weights") or "").replace("\\", "/"))

    # Resolve which checkpoint(s) belong to this run (history record first, else
    # the conventional scratchdet_*_rNNN.pt naming).
    hist = _load_history_file()
    ckpts = []
    rec = hist.get(run_no)
    if rec and rec.get("checkpoint"):
        ckpts.append(CKPT_DIR / rec["checkpoint"])
    ckpts += [p for p in CKPT_DIR.glob(f"*_r{run_no:03d}.pt") if p not in ckpts]

    if active_name and any(p.name == active_name for p in ckpts):
        return jsonify({"ok": False, "error": f"run #{run_no} is the active served "
                        f"model — train a newer run before deleting it"}), 400

    # 1) drop the structured history record first, so a failed rewrite aborts
    #    cleanly without half-applying the delete.
    removed_hist = False
    if HISTORY_PATH.exists():
        try:
            doc = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            runs = [r for r in doc.get("runs", []) if int(r.get("run", -1)) != run_no]
            if len(runs) != len(doc.get("runs", [])):
                removed_hist = True
                tmp = HISTORY_PATH.with_suffix(".json.tmp")
                tmp.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
                tmp.replace(HISTORY_PATH)
        except Exception as e:
            return jsonify({"ok": False, "error": f"failed to rewrite history: {e}"}), 500

    # 2) hide from the log-reconstructed view (and future reads)
    deleted = _load_deleted()
    deleted.add(run_no)
    _save_deleted(deleted)

    # 3) delete the checkpoint file(s) on disk; report any that could not be
    #    removed (e.g. Windows file lock) rather than silently swallowing it.
    deleted_files, failed_files = [], []
    for p in ckpts:
        try:
            if p.exists():
                p.unlink()
                deleted_files.append(p.name)
        except Exception as e:
            failed_files.append(p.name)
            print(f"[dashboard] could not delete checkpoint {p.name}: {e}")

    return jsonify({"ok": True, "run": run_no, "removed_history": removed_hist,
                    "deleted_checkpoints": deleted_files,
                    "failed_checkpoints": failed_files})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    """Pause or resume prediction generation. When paused, the ML backend's
    predict() short-circuits: deleting predictions in Label Studio won't make it
    regenerate them, and any in-progress generation halts on the next task."""
    body = request.get_json(force=True, silent=True) or {}
    cfg = _write_inference({"paused": bool(body.get("paused"))})
    return jsonify({"ok": True, "paused": cfg.get("paused", False)})


@app.route("/api/sahi", methods=["POST"])
def api_sahi():
    """Update SAHI sliced-inference settings live (applies to next prediction)."""
    body = request.get_json(force=True, silent=True) or {}
    cfg = _write_inference(body)
    return jsonify({"ok": True, "inference": cfg})


@app.route("/api/thresholds", methods=["POST"])
def api_thresholds():
    """Set the default + per-class confidence thresholds (filters predictions)."""
    body = request.get_json(force=True, silent=True) or {}
    cfg = _write_inference({k: body[k] for k in ("conf", "class_conf") if k in body})
    return jsonify({"ok": True, "inference": cfg})


@app.route("/api/sahi/recommend", methods=["POST"])
def api_sahi_recommend():
    """Compute + apply a recommended slice size from the current images."""
    rec, wh = _recommend_slice()
    if not rec:
        return jsonify({"ok": False, "error": "no images found to measure"}), 400
    cfg = _write_inference(rec)
    return jsonify({"ok": True, "inference": cfg, "median_image": list(wh)})


@app.route("/")
def index():
    return PAGE


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchDet — Training Dashboard</title>
<style>
:root{
 --bg:#0d1117;--bg2:#010409;--card:#161b22;--card2:#1c2230;--bd:#30363d;
 --fg:#e6edf3;--mut:#8b949e;--dim:#6e7681;
 --ok:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#58a6ff;--accent2:#1f6feb;
 --orange:#f0883e;--purple:#bc8cff;--teal:#39c5cf;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#16203022,transparent),var(--bg);
 color:var(--fg);font:14px/1.55 system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:14px 24px;border-bottom:1px solid var(--bd);
 background:linear-gradient(180deg,#10151ccc,#0d1117cc);backdrop-filter:blur(8px);position:sticky;top:0;z-index:5}
header .logo{font-size:18px;font-weight:700;letter-spacing:.2px}
header .logo small{color:var(--mut);font-weight:500;font-size:12px;margin-left:8px}
.badge{padding:4px 11px;border-radius:20px;font-size:12px;font-weight:600;border:1px solid transparent;white-space:nowrap}
.up{background:rgba(63,185,80,.13);color:var(--ok);border-color:rgba(63,185,80,.3)}
.down{background:rgba(248,81,73,.13);color:var(--bad);border-color:rgba(248,81,73,.3)}
.ver{background:rgba(88,166,255,.12);color:var(--accent);border-color:rgba(88,166,255,.3)}
.spacer{margin-left:auto}
.upd{color:var(--dim);font-size:12px}
.wrap{padding:18px 24px;max-width:1500px;margin:0 auto}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px;margin-bottom:16px}
.kpi{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--bd);border-radius:12px;padding:14px 16px;position:relative;overflow:hidden}
.kpi h3{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);font-weight:600}
.kpi .v{font-size:30px;font-weight:750;line-height:1.05;letter-spacing:-.5px}
.kpi .v small{font-size:15px;color:var(--mut);font-weight:600}
.kpi .s{color:var(--mut);font-size:12px;margin-top:5px}
.trend{font-size:12px;font-weight:700;margin-left:8px}
.bar{height:8px;background:#0a0e14;border:1px solid #21262d;border-radius:6px;overflow:hidden;margin-top:10px}
.bar>div{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));transition:width .5s}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:linear-gradient(180deg,var(--card),#13161d);border:1px solid var(--bd);border-radius:12px;padding:14px 16px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 10px;font-weight:600;display:flex;align-items:center;gap:10px}
.card h2 .right{margin-left:auto;font-weight:500;text-transform:none;letter-spacing:0}
.col12{grid-column:span 12}.col8{grid-column:span 8}.col6{grid-column:span 6}.col4{grid-column:span 4}
@media(max-width:1100px){.col8,.col6,.col4{grid-column:span 12}}
.chart{width:100%;height:auto;display:block}
select{background:#0d1117;color:var(--fg);border:1px solid var(--bd);border-radius:7px;padding:4px 8px;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid #21262d;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em;position:sticky;top:0;background:#13161d}
tbody tr:hover{background:#1b2230}
.scroll{max-height:300px;overflow:auto}
.pill{display:inline-block;background:#0d1117;border:1px solid var(--bd);border-radius:20px;padding:3px 10px;margin:3px 3px 0 0;font-size:12px}
.row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d}.row:last-child{border:0}
.row span:last-child{font-weight:600}.row span:first-child{color:var(--mut)}
pre{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:12px;max-height:340px;overflow:auto;
 font:12px/1.55 ui-monospace,"Cascadia Code",Consolas,monospace;margin:0;white-space:pre-wrap;color:#c9d1d9}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.muted{color:var(--mut)}.dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:6px}
/* live progress banner */
.live{display:flex;flex-direction:column;gap:10px;margin-bottom:16px}
.pcard{background:linear-gradient(180deg,#16203a,#121826);border:1px solid #2b3a55;border-radius:12px;padding:13px 16px}
.pcard .top{display:flex;align-items:center;gap:10px;font-size:13px;margin-bottom:9px}
.pcard .top b{font-weight:700}.pcard .top .pct{margin-left:auto;font-weight:700;font-size:15px}
.pcard .top .live-dot{width:9px;height:9px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 0 rgba(63,185,80,.6);animation:pulse 1.4s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,185,80,.55)}70%{box-shadow:0 0 0 9px rgba(63,185,80,0)}100%{box-shadow:0 0 0 0 rgba(63,185,80,0)}}
.pbar{height:14px;background:#0a0e14;border:1px solid #21262d;border-radius:8px;overflow:hidden}
.pbar>div{height:100%;border-radius:8px 0 0 8px;transition:width .5s;
 background-image:linear-gradient(90deg,var(--accent2),var(--accent)),linear-gradient(45deg,rgba(255,255,255,.14) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.14) 50%,rgba(255,255,255,.14) 75%,transparent 75%);
 background-size:auto,22px 22px;background-blend-mode:overlay;animation:stripe 1s linear infinite}
.pbar.infer>div{background-image:linear-gradient(90deg,#1f7a4d,#39c5cf),linear-gradient(45deg,rgba(255,255,255,.14) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.14) 50%,rgba(255,255,255,.14) 75%,transparent 75%)}
@keyframes stripe{from{background-position:0 0,0 0}to{background-position:0 0,22px 0}}
.pcard .sub{color:var(--mut);font-size:12px;margin-top:7px;display:flex;gap:16px;flex-wrap:wrap}
.pcard .eta{margin-left:14px;font-size:13px;font-weight:700;color:var(--accent)}
/* action buttons */
.actions{display:flex;align-items:center;gap:10px;margin-bottom:16px;flex-wrap:wrap}
.btn{background:linear-gradient(180deg,#238636,#1f7a31);color:#fff;border:1px solid #2ea043;border-radius:9px;
 padding:9px 16px;font-size:13px;font-weight:600;cursor:pointer;transition:filter .15s}
.btn:hover{filter:brightness(1.12)}.btn:active{filter:brightness(.95)}
.btn:disabled{opacity:.5;cursor:not-allowed;filter:grayscale(.4)}
.btn.ghost{background:linear-gradient(180deg,#21262d,#181d24);border-color:var(--bd);color:var(--fg)}
.btn.sm{padding:6px 12px;font-size:12px}
#actmsg{font-size:12.5px}
/* SAHI control bar */
.sahibar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:var(--card);border:1px solid var(--bd);
 border-radius:10px;padding:10px 14px;margin-bottom:16px;font-size:13px}
.sahibar .lbl{font-weight:700;color:var(--accent)}
.sahibar .sw{display:flex;align-items:center;gap:6px;cursor:pointer}
.sahibar .num{width:78px;background:#0d1117;color:var(--fg);border:1px solid var(--bd);border-radius:6px;padding:5px 8px;font-size:13px}
#saMsg{font-size:12.5px}
/* thresholds */
.threshbar{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 14px;margin-bottom:16px;font-size:13px}
.threshbar .thead{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.threshbar .lbl{font-weight:700;color:var(--accent)}
.num{width:78px;background:#0d1117;color:var(--fg);border:1px solid var(--bd);border-radius:6px;padding:5px 8px;font-size:13px}
.thgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px;margin-top:12px;max-height:260px;overflow:auto;
 border-top:1px solid #21262d;padding-top:12px}
.throw{display:flex;align-items:center;justify-content:space-between;gap:8px;background:#0d1117;border:1px solid var(--bd);border-radius:8px;padding:6px 10px}
.throw span{color:var(--fg);font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.throw .num{width:66px}
#thMsg{font-size:12.5px}
/* custom tooltip (replaces native SVG <title>) */
#tip{position:fixed;z-index:50;pointer-events:none;display:none;max-width:280px;
 background:rgba(13,17,23,.97);border:1px solid var(--bd);border-radius:9px;padding:8px 11px;
 box-shadow:0 8px 28px rgba(0,0,0,.55);font-size:12.5px;line-height:1.5;color:var(--fg);
 backdrop-filter:blur(4px)}
#tip .th{font-weight:700;margin-bottom:4px;font-size:12px;color:var(--fg)}
#tip .tr{display:flex;align-items:center;gap:7px;justify-content:space-between;white-space:nowrap}
#tip .tr .nm{display:flex;align-items:center;gap:6px;color:var(--mut)}
#tip .tr b{font-weight:700;color:var(--fg)}
#tip .sw{width:9px;height:9px;border-radius:2px;display:inline-block;flex:0 0 auto}
svg .hit{cursor:crosshair}
svg .cx-line{stroke:var(--mut);stroke-width:1;stroke-dasharray:4 3;pointer-events:none}
svg .cx-dot{pointer-events:none}
/* per-run / per-class heatmap */
.hmwrap{overflow-x:auto}
.hm{border-collapse:separate;border-spacing:0;font-size:12px;min-width:100%}
.hm th,.hm td{border:1px solid #21262d;padding:0;text-align:center;white-space:nowrap}
.hm thead th{position:sticky;top:0;background:#13161d;color:var(--mut);font-weight:600;
 font-size:10.5px;text-transform:none;letter-spacing:0;height:96px;vertical-align:bottom;padding:4px}
.hm thead th.rh{left:0;z-index:3;text-align:left;min-width:64px}
.hm thead th .rot{display:inline-block;writing-mode:vertical-rl;transform:rotate(180deg);
 max-height:84px;overflow:hidden;text-overflow:ellipsis}
.hm tbody th{position:sticky;left:0;background:#13161d;color:var(--fg);font-weight:600;
 text-align:left;padding:5px 9px;z-index:1;border-right:1px solid var(--bd)}
.hm td{width:30px;height:26px;cursor:default;color:#e6edf3;font-variant-numeric:tabular-nums}
.hm td.z{color:var(--dim)}
/* run delete / re-run buttons */
.delrun{background:transparent;border:1px solid transparent;border-radius:6px;color:var(--mut);
 cursor:pointer;font-size:13px;padding:2px 7px;line-height:1;transition:all .12s}
.delrun:hover{color:var(--bad);border-color:rgba(248,81,73,.4);background:rgba(248,81,73,.1)}
.rerun{background:transparent;border:1px solid transparent;border-radius:6px;color:var(--accent);
 cursor:pointer;font-size:14px;padding:2px 7px;line-height:1;transition:all .12s}
.rerun:hover{color:var(--accent);border-color:rgba(88,166,255,.45);background:rgba(88,166,255,.12)}
.rerun:disabled{opacity:.5;cursor:not-allowed}
/* interrupted-run warning banner */
.pcard.warn{background:linear-gradient(180deg,#2a210f,#1c1810);border-color:#5c4612}
.pcard.warn .top b{color:var(--warn)}
</style></head><body>
<div id="tip"></div>
<header>
  <div class="logo">🏗️ ScratchDet<small>from-scratch detector · live training dashboard</small></div>
  <span id="ver" class="badge ver" style="display:none"></span>
  <span id="sahi" class="badge" style="display:none"></span>
  <div class="spacer"></div>
  <span id="ls" class="badge down">Label Studio …</span>
  <span id="ml" class="badge down">ML backend …</span>
  <span class="upd" id="upd"></span>
</header>
<div class="wrap">
  <div class="actions">
    <button id="btnTrain" class="btn">⚡ Force train now</button>
    <button id="btnRefresh" class="btn ghost">⟳ Refresh predictions in LS</button>
    <button id="btnPause" class="btn ghost">⏸ Pause generating</button>
    <span id="actmsg" class="muted"></span>
  </div>
  <div class="sahibar">
    <span class="lbl">SAHI inference</span>
    <label class="sw"><input type="checkbox" id="saOn"> sliced</label>
    <span>slice <input type="number" id="saSize" class="num" min="256" max="8192" step="64"> px</span>
    <span>overlap <input type="number" id="saOv" class="num" min="0" max="0.8" step="0.05"></span>
    <button id="saApply" class="btn ghost sm">Apply</button>
    <button id="saRec" class="btn sm" title="Pick a slice size from the current image dimensions">★ Recommended</button>
    <span id="saMsg" class="muted"></span>
  </div>
  <div class="threshbar">
    <div class="thead">
      <span class="lbl">Confidence thresholds</span>
      <span>default <input type="number" id="thDef" class="num" min="0" max="1" step="0.05"></span>
      <button id="thApply" class="btn ghost sm">Apply thresholds</button>
      <button id="thToggle" class="btn ghost sm">per-class ▾</button>
      <span class="muted" style="font-size:12px">higher = fewer boxes per class</span>
      <span id="thMsg" class="muted"></span>
    </div>
    <div id="thGrid" class="thgrid" style="display:none"></div>
  </div>
  <div id="prog"></div>
  <div class="kpis" id="kpis"></div>
  <div class="grid" id="grid"></div>
</div>
<script>
let LAST=null, LASTSIG=null, SEL=null, CMP_METRIC='val_loss';
const fmt=(n,d=0)=>n==null?'—':(typeof n==='number'?n.toLocaleString(undefined,{maximumFractionDigits:d}):n);
const f4=n=>n==null?'—':Number(n).toFixed(4);
function fmtDur(s){s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),x=s%60;return h?`${h}h ${m}m`:(m?`${m}m ${x}s`:`${x}s`);}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const RUNCOLORS=['#58a6ff','#3fb950','#f0883e','#bc8cff','#f85149','#39c5cf','#d29922','#ff7bbf','#7ee787','#a371f7','#ffa657','#79c0ff'];
const runColor=i=>RUNCOLORS[((i%RUNCOLORS.length)+RUNCOLORS.length)%RUNCOLORS.length];
// When `attr` is set the result is stashed in a data-tip="..." attribute and
// later re-injected via innerHTML; the browser HTML-decodes the attribute once,
// so dynamic (possibly user-named) values must be escaped TWICE to survive that
// round-trip — structural tags stay single so they still render. Direct callers
// (the crosshair) pass attr=false and escape once.
function tipHTML(title,rows,attr){const e=attr?(s=>esc(esc(s))):esc;
  return `<div class='th'>${e(title)}</div>`+rows.map(r=>
  `<div class='tr'><span class='nm'>${r.c?`<span class='sw' style='background:${r.c}'></span>`:''}${e(r.nm)}</span><b>${e(r.val)}</b></div>`).join('');}

/* ---------- custom tooltip + line-chart crosshair machinery ---------- */
let CHARTS={}, CID=0;
function showTip(html,x,y){const t=document.getElementById('tip'); if(!t)return; t.innerHTML=html; t.style.display='block';
  const pad=14,tw=t.offsetWidth,th=t.offsetHeight,vw=innerWidth,vh=innerHeight;
  let L=x+pad,T=y+pad; if(L+tw>vw-6)L=x-tw-pad; if(T+th>vh-6)T=y-th-pad;
  t.style.left=Math.max(6,L)+'px'; t.style.top=Math.max(6,T)+'px';}
function hideTip(){const t=document.getElementById('tip'); if(t)t.style.display='none';
  document.querySelectorAll('.cross').forEach(c=>c.style.display='none');}
function lineHover(hit,e){
  const id=hit.getAttribute('data-cid'),ch=CHARTS[id]; if(!ch){hideTip();return;}
  // moving straight from one chart onto another must not leave the old crosshair
  document.querySelectorAll('.cross').forEach(c=>{if(c.id!=='cross-'+id)c.style.display='none';});
  const svg=hit.ownerSVGElement,ctm=svg&&svg.getScreenCTM(); if(!ctm)return;
  const sp=svg.createSVGPoint(); sp.x=e.clientX; sp.y=e.clientY;
  const loc=sp.matrixTransform(ctm.inverse()), g=ch.geom;
  const vx=Math.max(g.pl,Math.min(g.pl+g.iw,loc.x));
  const xv=g.xmin+(vx-g.pl)/(g.iw||1)*(g.xmax-g.xmin);
  let snap=ch.xvals[0],bd=Infinity; for(const xx of ch.xvals){const dd=Math.abs(xx-xv); if(dd<bd){bd=dd;snap=xx;}}
  const X=x=>g.pl+(x-g.xmin)/((g.xmax-g.xmin)||1)*g.iw, Y=y=>g.pt+(1-(y-g.ymin)/((g.ymax-g.ymin)||1))*g.ih;
  // accept a series only at (essentially) the snapped x: half the grid step, so a
  // run that didn't reach this epoch isn't shown a neighbour's value mislabelled.
  let step=Infinity; for(let i=1;i<ch.xvals.length;i++){const dd=ch.xvals[i]-ch.xvals[i-1]; if(dd>0&&dd<step)step=dd;}
  const tol=isFinite(step)?step*0.5:0.5;
  const cross=svg.querySelector('#cross-'+id), dots=cross?cross.querySelectorAll('.cx-dot'):[];
  if(cross){cross.style.display=''; const ln=cross.querySelector('.cx-line'); if(ln){ln.setAttribute('x1',X(snap));ln.setAttribute('x2',X(snap));}}
  const rows=[];
  ch.series.forEach((s,si)=>{
    let bi=-1,bbd=Infinity; for(let i=0;i<s.pts.length;i++){const dd=Math.abs(s.pts[i][0]-snap); if(dd<bbd){bbd=dd;bi=i;}}
    const p=bi>=0?s.pts[bi]:null, ok=p&&p[1]!=null&&bbd<=tol, dot=dots[si];
    if(dot){ if(ok){dot.style.display='';dot.setAttribute('cx',X(p[0]));dot.setAttribute('cy',Y(p[1]));} else dot.style.display='none'; }
    if(ok) rows.push({nm:s.name,val:ch.yfmt(p[1]),c:s.c});
  });
  if(!rows.length){hideTip();return;}
  showTip(tipHTML(ch.xl+' '+snap,rows),e.clientX,e.clientY);
}

/* ---------- hand-rolled SVG charts (no external libs) ---------- */
function lineChart(series,opt){
 opt=opt||{}; const W=760,H=300,pl=54,pr=18,pt=22,pb=40,iw=W-pl-pr,ih=H-pt-pb;
 const yfmt=opt.yfmt||(v=>(+v).toFixed(v<1?4:3)), xl=opt.xl||'epoch';
 let xs=[],ys=[]; series.forEach(s=>s.pts.forEach(p=>{if(p[1]!=null){xs.push(p[0]);ys.push(p[1]);}}));
 if(!xs.length) return '<div class="muted" style="padding:40px 0;text-align:center">no training data yet</div>';
 let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const pad=(ymax-ymin)*0.08||0.5; ymax+=pad; ymin=Math.max(0,ymin-pad); if(xmax===xmin)xmax=xmin+1;
 const X=x=>pl+(x-xmin)/(xmax-xmin)*iw, Y=y=>pt+(1-(y-ymin)/((ymax-ymin)||1))*ih;
 let g=''; const T=5;
 for(let i=0;i<=T;i++){const v=ymin+(ymax-ymin)*i/T,yy=Y(v);
   g+=`<line x1="${pl}" y1="${yy.toFixed(1)}" x2="${pl+iw}" y2="${yy.toFixed(1)}" stroke="#21262d"/>`;
   g+=`<text x="${pl-9}" y="${(yy+3.5).toFixed(1)}" fill="#8b949e" font-size="11" text-anchor="end">${v.toFixed(v<1?2:1)}</text>`;}
 const XT=Math.min(6,Math.max(1,Math.round(xmax-xmin)));
 for(let i=0;i<=XT;i++){const v=xmin+(xmax-xmin)*i/XT,xx=X(v);
   g+=`<text x="${xx.toFixed(1)}" y="${H-pb+17}" fill="#8b949e" font-size="11" text-anchor="middle">${Math.round(v)}</text>`;}
 let paths='',defs='',leg='';
 series.forEach((s,si)=>{
   const dpts=s.pts.filter(p=>p[1]!=null);
   const pts=dpts.map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ');
   if(s.area&&dpts.length){const id='ar'+(CID+1)+'_'+si;
     defs+=`<linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${s.c}" stop-opacity=".28"/><stop offset="1" stop-color="${s.c}" stop-opacity="0"/></linearGradient>`;
     paths+=`<polygon points="${X(dpts[0][0]).toFixed(1)},${Y(ymin).toFixed(1)} ${pts} ${X(dpts[dpts.length-1][0]).toFixed(1)},${Y(ymin).toFixed(1)}" fill="url(#${id})"/>`;}
   paths+=`<polyline points="${pts}" fill="none" stroke="${s.c}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>`;
   if(dpts.length){const L=dpts[dpts.length-1]; paths+=`<circle cx="${X(L[0]).toFixed(1)}" cy="${Y(L[1]).toFixed(1)}" r="3.2" fill="${s.c}" stroke="#0d1117" stroke-width="1.5"/>`;}
   leg+=`<span class="pill" style="border-color:${s.c}66"><span class="dot" style="background:${s.c}"></span>${esc(s.name)}</span>`;
 });
 const ref=series.reduce((a,b)=>b.pts.length>a.pts.length?b:a,series[0]);
 const id=++CID;
 CHARTS[id]={geom:{W,H,pl,pr,pt,pb,iw,ih,xmin,xmax,ymin,ymax},
   xvals:ref.pts.filter(p=>p[1]!=null).map(p=>p[0]),
   series:series.map(s=>({name:s.name,c:s.c,pts:s.pts.filter(p=>p[1]!=null)})),yfmt,xl};
 const cross=`<g class="cross" id="cross-${id}" style="display:none"><line class="cx-line" x1="${pl}" x2="${pl}" y1="${pt}" y2="${pt+ih}"/>`+
   series.map(s=>`<circle class="cx-dot" r="4" fill="${s.c}" stroke="#0d1117" stroke-width="1.5" style="display:none"/>`).join('')+`</g>`;
 return `<div style="margin-bottom:6px">${leg}</div>`+
   `<svg viewBox="0 0 ${W} ${H}" class="chart"><defs>${defs}</defs>`+
   `<line x1="${pl}" y1="${pt}" x2="${pl}" y2="${pt+ih}" stroke="#30363d"/>`+
   `<line x1="${pl}" y1="${pt+ih}" x2="${pl+iw}" y2="${pt+ih}" stroke="#30363d"/>`+
   `${g}${paths}${cross}`+
   `<rect class="hit" data-cid="${id}" x="${pl}" y="${pt}" width="${iw}" height="${ih}" fill="transparent"/>`+
   `<text x="${pl+iw/2}" y="${H-2}" fill="#6e7681" font-size="11" text-anchor="middle">${esc(xl)}</text></svg>`;
}
function barChart(items,opt){
 opt=opt||{}; const W=760,H=280,pl=50,pr=14,pt=18,pb=46,iw=W-pl-pr,ih=H-pt-pb;
 if(!items.length) return '<div class="muted" style="padding:40px 0;text-align:center">no runs yet</div>';
 const vmax=Math.max(...items.map(i=>i.v))||1,gap=iw/items.length,bw=Math.min(46,gap*0.6);
 let g=''; const T=4;
 for(let i=0;i<=T;i++){const v=vmax*i/T,yy=pt+(1-i/T)*ih;
   g+=`<line x1="${pl}" y1="${yy.toFixed(1)}" x2="${pl+iw}" y2="${yy.toFixed(1)}" stroke="#21262d"/>`;
   g+=`<text x="${pl-9}" y="${(yy+3.5).toFixed(1)}" fill="#8b949e" font-size="11" text-anchor="end">${v.toFixed(vmax<10?2:0)}</text>`;}
 let bars='';
 items.forEach((it,i)=>{const x=pl+gap*i+(gap-bw)/2,bh=it.v/vmax*ih,y=pt+ih-bh;
   const tip=tipHTML(it.label,[{nm:opt.yl||'value',val:it.disp!=null?it.disp:it.v}].concat(it.extra||[]),true);
   bars+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(bh,1).toFixed(1)}" rx="4" fill="url(#bg1)" data-tip="${tip}"/>`;
   bars+=`<text x="${(x+bw/2).toFixed(1)}" y="${(y-6).toFixed(1)}" fill="#c9d1d9" font-size="11" text-anchor="middle">${it.disp!=null?it.disp:it.v}</text>`;
   bars+=`<text x="${(x+bw/2).toFixed(1)}" y="${H-pb+18}" fill="#8b949e" font-size="11" text-anchor="middle">${esc(it.label)}</text>`;});
 return `<svg viewBox="0 0 ${W} ${H}" class="chart"><defs><linearGradient id="bg1" x1="0" y1="0" x2="0" y2="1">`+
   `<stop offset="0" stop-color="#58a6ff"/><stop offset="1" stop-color="#1f6feb"/></linearGradient></defs>`+
   `<line x1="${pl}" y1="${pt+ih}" x2="${pl+iw}" y2="${pt+ih}" stroke="#30363d"/>${g}${bars}</svg>`;
}
function hbarChart(items){
 if(!items.length) return '<div class="muted" style="padding:24px 0;text-align:center">no labels yet</div>';
 const vmax=Math.max(...items.map(i=>i.v))||1,W=760,pl=128,pr=46,pt=6,rh=30,H=pt*2+items.length*rh,iw=W-pl-pr;
 let rows='';
 items.forEach((it,i)=>{const y=pt+i*rh+5,bw=it.v/vmax*iw;
   rows+=`<text x="${pl-10}" y="${(y+14).toFixed(1)}" fill="#c9d1d9" font-size="12.5" text-anchor="end">${esc(it.label)}</text>`;
   rows+=`<rect x="${pl}" y="${y}" width="${iw}" height="18" rx="5" fill="#0d1117" stroke="#21262d"/>`;
   rows+=`<rect x="${pl}" y="${y}" width="${Math.max(bw,3).toFixed(1)}" height="18" rx="5" fill="url(#bg2)"/>`;
   rows+=`<text x="${(pl+Math.max(bw,3)+7).toFixed(1)}" y="${(y+13.5).toFixed(1)}" fill="#8b949e" font-size="11.5">${it.v}</text>`;
   rows+=`<rect x="0" y="${(y-4).toFixed(1)}" width="${W}" height="26" fill="transparent" data-tip="${tipHTML(it.label,[{nm:'boxes',val:it.v}],true)}"/>`;});
 return `<svg viewBox="0 0 ${W} ${H}" class="chart"><defs><linearGradient id="bg2" x1="0" y1="0" x2="1" y2="0">`+
   `<stop offset="0" stop-color="#1f6feb"/><stop offset="1" stop-color="#39c5cf"/></linearGradient></defs>${rows}</svg>`;
}
function heatmapChart(rowsData,classes){
 if(!rowsData.length||!classes.length) return '<div class="muted" style="padding:24px 0;text-align:center">no per-class data yet</div>';
 let max=0; rowsData.forEach(r=>classes.forEach(c=>{const v=(r.per_class||{})[c]||0; if(v>max)max=v;}));
 const head=`<thead><tr><th class="rh">run ↓ · class →</th>`+
   classes.map(c=>`<th><span class="rot">${esc(c)}</span></th>`).join('')+`</tr></thead>`;
 const body='<tbody>'+rowsData.map(r=>{
   const tds=classes.map(c=>{const v=(r.per_class||{})[c]||0,t=max?Math.sqrt(v/max):0;
     const bg=v?`background:rgba(88,166,255,${(0.10+0.6*t).toFixed(3)})`:'';
     return `<td class="${v?'':'z'}" style="${bg}" data-tip="${tipHTML('run #'+r.run+' · '+c,[{nm:'boxes',val:v}],true)}">${esc(String(v||'·'))}</td>`;}).join('');
   return `<tr><th>#${r.run}</th>${tds}</tr>`;}).join('')+'</tbody>';
 return `<div class="hmwrap"><table class="hm">${head}${body}</table></div>`;
}

function curveFor(d,run){const c=(d.curves||{})[String(run)];return c&&c.epoch&&c.epoch.length?c:null;}

function render(d){
 hideTip(); CHARTS={};          // charts get rebuilt below; drop stale hover state
 // ---- header chips ----
 const setb=(id,ok,txt)=>{const e=document.getElementById(id);e.className='badge '+(ok?'up':'down');e.textContent=txt+(ok?' ● up':' ● down');};
 setb('ls',d.ls_health,'Label Studio'); setb('ml',d.ml_health,'ML backend');
 const v=document.getElementById('ver'); if(d.serving){v.style.display='';v.textContent='● serving '+d.model_version;}else{v.style.display='none';}
 // SAHI sliced-inference indicator
 const si=d.inference||{}, sb=document.getElementById('sahi');
 if(si.sliced===true){ sb.style.display=''; sb.className='badge up'; sb.textContent='SAHI ● sliced '+si.slice+'px / '+Math.round((si.overlap||0)*100)+'% overlap'; }
 else if(si.sliced===false){ sb.style.display=''; sb.className='badge down'; sb.textContent='SAHI ○ off (single pass)'; }
 else { sb.style.display='none'; }

 // ---- live progress banner (training / inference) ----
 const pg=d.progress||{}; let plive='';
 if(pg.train){const t=pg.train;
   const e=t.epoch + ((t.batch&&t.batches)? t.batch/t.batches : 0);
   const pc=t.epochs?Math.min(100,e/t.epochs*100):0;
   const sub=(t.batches&&t.batch&&t.epoch>=0)?` · batch ${t.batch}/${t.batches}`:(t.phase?` · ${t.phase}`:'');
   const eta=(t.eta_s!=null)?`<span class="eta">⏳ ~${fmtDur(t.eta_s)} left</span>`:`<span class="eta">⏳ estimating…</span>`;
   plive+=`<div class="pcard"><div class="top"><span class="live-dot"></span><b>Training</b> · run #${t.run} <span class="muted">· from scratch</span>${eta}<span class="pct">${pc.toFixed(1)}%</span></div>
     <div class="pbar"><div style="width:${pc}%"></div></div>
     <div class="sub"><span>epoch ${t.epoch} / ${t.epochs}${sub}</span>${t.elapsed_s!=null?`<span>elapsed ${fmtDur(t.elapsed_s)}</span>`:''}${t.images!=null?`<span>${t.images} images</span>`:''}${t.train_loss!=null?`<span>train loss ${t.train_loss}</span>`:''}${t.val_loss!=null?`<span>val loss ${t.val_loss}</span>`:''}${t.best_val!=null?`<span>best ${t.best_val}</span>`:''}</div></div>`;}
 if(pg.infer){const t=pg.infer,pc=t.total?Math.min(100,t.done/t.total*100):0;
   plive+=`<div class="pcard"><div class="top"><span class="live-dot"></span><b>Predicting</b> <span class="muted">· serving boxes to Label Studio</span><span class="pct">${pc.toFixed(0)}%</span></div>
     <div class="pbar infer"><div style="width:${pc}%"></div></div>
     <div class="sub"><span>${t.done} / ${t.total} tasks</span>${t.version?`<span>${t.version}</span>`:''}</div></div>`;}
 // ---- interrupted/killed runs: a run with no best_val (never wrote a DONE record)
 //      that isn't the one currently training was killed mid-run -> offer a re-run.
 const activeRun=pg.train?pg.train.run:null;
 const killed=(d.runs||[]).filter(r=>r.best_val_loss==null&&r.run!==activeRun);
 let kbanner='';
 if(killed.length&&!pg.train){
   const k=killed[killed.length-1];
   kbanner=`<div class="pcard warn"><div class="top"><span>⚠️</span><b>Run #${k.run} was interrupted</b>
       <span class="muted">· stopped after ${k.epochs!=null?k.epochs:'?'} epoch(s); never finished</span>
       <span style="margin-left:auto"><button class="btn sm rerun" data-rerun="${k.run}">↻ Re-run training</button></span></div>
     <div class="sub"><span>A killed run leaves an orphan checkpoint and no final record (the trainer has no early-stop — it ran to 100 epochs). Re-running starts a fresh from-scratch run on all current annotations.${killed.length>1?` · ${killed.length} interrupted runs total`:''}</span></div></div>`;
 }
 document.getElementById('prog').innerHTML=(plive||kbanner)?`<div class="live">${plive}${kbanner}</div>`:'';

 const ev=d.events_since_train,every=d.retrain_every,pct=Math.min(100,ev/every*100);
 const runs=d.runs||[],pc=d.per_class||{};
 const last=runs[runs.length-1]||{},prev=runs[runs.length-2]||{};
 // best-val trend
 let trend='';
 if(last.best_val_loss!=null&&prev.best_val_loss!=null){
   const dlt=last.best_val_loss-prev.best_val_loss, better=dlt<0;
   trend=`<span class="trend" style="color:${better?'var(--ok)':'var(--bad)'}">${better?'▼':'▲'} ${Math.abs(dlt).toFixed(3)}</span>`;
 }
 const bestEver=runs.filter(r=>r.best_val_loss!=null).reduce((m,r)=>Math.min(m,r.best_val_loss),Infinity);

 // ---- KPI strip ----
 document.getElementById('kpis').innerHTML=`
  <div class="kpi"><h3>Next retrain in</h3><div class="v">${d.events_remaining}<small> events</small></div>
    <div class="s">${ev} / ${every} since last train</div><div class="bar"><div style="width:${pct}%"></div></div></div>
  <div class="kpi"><h3>Annotated images</h3><div class="v">${fmt(d.annotated_images)}</div>
    <div class="s">${fmt(d.total_annotations)} annotations · <a href="http://${location.hostname}:${d.ls_port}" target="_blank">open LS →</a></div></div>
  <div class="kpi"><h3>Predictions in LS</h3><div class="v">${fmt(d.total_predictions)}<small> / ${fmt(d.tasks)}</small></div>
    <div class="s">auto-predict ${d.auto_predict==null?'—':(d.auto_predict?'<span style="color:var(--ok)">ON</span>':'<span style="color:var(--warn)">OFF</span>')} · inference: ${si.sliced===true?`<span style="color:var(--ok)">SAHI ${si.slice}px</span>`:(si.sliced===false?'single-pass':'—')}</div></div>
  <div class="kpi"><h3>Training runs</h3><div class="v">${d.train_runs}</div>
    <div class="s">variant <b>${d.variant}</b> · from scratch</div></div>
  <div class="kpi"><h3>Best val loss</h3><div class="v">${last.best_val_loss!=null?last.best_val_loss.toFixed(3):'—'}${trend}</div>
    <div class="s">best ever ${isFinite(bestEver)?bestEver.toFixed(3):'—'}</div></div>
  <div class="kpi"><h3>Compute</h3><div class="v" style="font-size:18px;padding-top:8px">${(d.device||'—').replace(/\(.*\)/,'').trim()||'—'}</div>
    <div class="s">${(d.device&&d.device.match(/\((.*)\)/)||['',''])[1]||(last.duration_s?last.duration_s+'s last run':'')}</div></div>`;

 // ---- run selector options ----
 const curveIds=Object.keys(d.curves||{}).map(Number).sort((a,b)=>a-b);
 if(SEL==null||!curveIds.includes(SEL)) SEL=curveIds[curveIds.length-1]??null;
 const opts=curveIds.map(r=>`<option value="${r}" ${r===SEL?'selected':''}>run #${r}${r===curveIds[curveIds.length-1]?' (latest)':''}</option>`).join('');

 // ---- main loss curve (selected run) ----
 const c=curveFor(d,SEL);
 let lossSvg='<div class="muted" style="padding:40px 0;text-align:center">no per-epoch data for this run</div>';
 let compSvg=lossSvg;
 if(c){
   const E=c.epoch;
   lossSvg=lineChart([
     {name:'train loss',c:'var(--orange)',area:true,pts:E.map((e,i)=>[e,c.train_loss[i]])},
     {name:'val loss',c:'var(--accent)',pts:E.map((e,i)=>[e,c.val_loss[i]])},
   ]);
   if(c.obj&&c.obj.length){
     compSvg=lineChart([
       {name:'objectness',c:'var(--teal)',pts:E.map((e,i)=>[e,c.obj[i]])},
       {name:'box',c:'var(--purple)',pts:E.map((e,i)=>[e,c.box[i]])},
       {name:'cls',c:'var(--warn)',pts:E.map((e,i)=>[e,c.cls[i]])},
     ]);
   }
 }

 // ---- all-runs performance: one line per run, metric-selectable ----
 const METRICS={val_loss:'val loss',train_loss:'train loss',obj:'objectness',box:'box',cls:'cls'};
 if(!METRICS[CMP_METRIC]) CMP_METRIC='val_loss';
 const cmpRuns=curveIds.map(rid=>{const cc=d.curves[String(rid)]||{},yv=cc[CMP_METRIC]||[];
   return {name:'run #'+rid,c:runColor(rid),pts:(cc.epoch||[]).map((e,j)=>[e,yv[j]]).filter(p=>p[1]!=null)};
 }).filter(s=>s.pts.length);
 const cmpSvg=cmpRuns.length?lineChart(cmpRuns,{xl:'epoch'})
   :'<div class="muted" style="padding:40px 0;text-align:center">no runs with per-epoch data yet</div>';
 const cmpOpts=Object.entries(METRICS).map(([k,l])=>`<option value="${k}" ${k===CMP_METRIC?'selected':''}>${l}</option>`).join('');

 // ---- per-run best-val bars ----
 const runBars=runs.filter(r=>r.best_val_loss!=null).slice(-20)
   .map(r=>({label:'#'+r.run,v:r.best_val_loss,disp:r.best_val_loss.toFixed(2),
     extra:[{nm:'images',val:fmt(r.images)},{nm:'epochs',val:fmt(r.epochs)},
            {nm:'time',val:r.duration_s!=null?r.duration_s+'s':'—'}]}));
 // ---- per-class hbars (latest run) ----
 const pcItems=Object.entries(pc).sort((a,b)=>b[1]-a[1]).map(([k,v])=>({label:k,v}));
 // ---- per-run per-class box-count heatmap ----
 const hmRuns=runs.filter(r=>r.per_class&&Object.keys(r.per_class).length)
   .map(r=>({run:r.run,per_class:r.per_class}));
 let hmClasses=(d.classes&&d.classes.length)?d.classes.slice():[];
 if(!hmClasses.length){const set=new Set(); hmRuns.forEach(r=>Object.keys(r.per_class).forEach(k=>set.add(k))); hmClasses=[...set];}

 // ---- run history table ----
 const rows=runs.slice().map(r=>`<tr>
   <td>#${r.run}</td><td>${r.variant||'—'}</td><td>${fmt(r.images)}</td><td>${fmt(r.total_boxes)}</td>
   <td>${r.best_val_loss!=null?r.best_val_loss.toFixed(4):'—'}</td><td>${fmt(r.epochs)}</td>
   <td>${r.duration_s!=null?r.duration_s+'s':'—'}</td>
   <td class="muted" style="text-align:left">${(r.finished_at||'').replace('T',' ').replace('+00:00','')||(r.source==='log'?'(from log)':'')}${(r.best_val_loss==null&&r.run!==activeRun)?' <span style="color:var(--warn)">⚠ interrupted</span>':''}</td>
   <td style="white-space:nowrap">${(r.best_val_loss==null&&r.run!==activeRun)?`<button class="rerun" data-rerun="${r.run}" ${pg.train?'disabled':''} title="Re-run training (run #${r.run} was interrupted)">↻</button> `:''}<button class="delrun" data-del="${r.run}" title="Delete run #${r.run} data">🗑</button></td></tr>`).join('')
   ||'<tr><td colspan="9" class="muted">no runs yet</td></tr>';

 const ck=(d.checkpoints||[]).map(x=>`<div class="row"><span>${x.name}</span><span>${x.mb} MB</span></div>`).join('')||'<span class="muted">none yet</span>';
 const cls=(d.classes||[]).map(x=>`<span class="pill">${x}</span>`).join('')||'<span class="muted">—</span>';

 document.getElementById('grid').innerHTML=`
  <div class="card col8"><h2>Loss curves
      <span class="right"><select id="runsel">${opts||'<option>—</option>'}</select></span></h2>${lossSvg}</div>
  <div class="card col4"><h2>Loss components <span class="right muted">run #${SEL??'—'}</span></h2>${compSvg}</div>
  <div class="card col12"><h2>All runs — performance
      <span class="right"><select id="cmpsel">${cmpOpts}</select> per epoch · ${cmpRuns.length} run${cmpRuns.length===1?'':'s'}</span></h2>${cmpSvg}</div>
  <div class="card col12"><h2>Best val loss per run</h2>${barChart(runBars,{yl:'best val loss'})}</div>
  <div class="card col12"><h2>Boxes per class — per run
      <span class="right muted">rows = runs · columns = classes · colour = count</span></h2>${heatmapChart(hmRuns,hmClasses)}</div>
  <div class="card col6"><h2>Boxes per class <span class="right muted">latest train</span></h2>${hbarChart(pcItems)}</div>
  <div class="card col6"><h2>Checkpoints</h2>${ck}
     <h2 style="margin-top:14px">Classes</h2><div>${cls}</div></div>
  <div class="card col12"><h2>Run history <span class="right muted">🗑 deletes a run's record + checkpoint</span></h2><div class="scroll"><table>
     <thead><tr><th>Run</th><th>Variant</th><th>Images</th><th>Boxes</th><th>Best val</th><th>Epochs</th><th>Time</th><th>Finished (UTC)</th><th></th></tr></thead>
     <tbody>${rows}</tbody></table></div></div>
  <div class="card col12"><h2>Training log <span class="right muted">live · data/logs/training.log</span></h2><pre id="log"></pre></div>`;

 const sel=document.getElementById('runsel');
 if(sel) sel.onchange=e=>{SEL=Number(e.target.value); if(LAST) render(LAST);};
 const cmp=document.getElementById('cmpsel');
 if(cmp) cmp.onchange=e=>{CMP_METRIC=e.target.value; if(LAST) render(LAST);};
 const lg=document.getElementById('log'); if(lg){lg.textContent=d.log_tail||'(no log yet)';lg.scrollTop=lg.scrollHeight;}
}

let ACTMSG_T=null;
function actmsg(t){ const e=document.getElementById('actmsg'); e.textContent=t;
  if(ACTMSG_T) clearTimeout(ACTMSG_T); ACTMSG_T=setTimeout(()=>{e.textContent='';},6000); }
async function post(url){ try{ const r=await fetch(url,{method:'POST'}); return await r.json(); }catch(e){ return {ok:false,error:String(e)}; } }
async function startTrain(confirmMsg){
  if(LAST&&LAST.progress&&LAST.progress.train){ actmsg('a training run is already in progress'); return; }
  if(confirmMsg&&!confirm(confirmMsg)) return;
  actmsg('starting training…'); const r=await post('/api/train');
  actmsg(r.ok?'✓ training started — watch the progress bar above':'✗ '+(r.error||'failed'));
}
document.getElementById('btnTrain').onclick=()=>startTrain(
  'Start a full from-scratch training run now? This uses the GPU for a few minutes. Predictions auto-update in Label Studio when it finishes.');
// Re-run an interrupted run (banner + per-row ↻). Training is always from scratch
// on all current data, so this is a fresh run with the next number; the killed run
// is left in place (delete it with 🗑 if you want it gone).
document.addEventListener('click',e=>{
  const b=e.target.closest?e.target.closest('.rerun'):null; if(!b) return;
  const n=b.getAttribute('data-rerun');
  startTrain('Re-run training'+(n?` — run #${n} was interrupted`:'')+'?\n\n'+
    'This starts a fresh from-scratch run on ALL current annotations (it gets the next run number). '+
    'The interrupted run is left as-is; delete it with 🗑 if you want it removed.');
});
document.getElementById('btnRefresh').onclick=async()=>{
  actmsg('refreshing predictions…'); const r=await post('/api/refresh');
  actmsg(r.ok?'✓ '+(r.status||'refreshing predictions in Label Studio'):'✗ '+(r.error||'failed'));
};
document.getElementById('btnPause').onclick=async()=>{
  const paused=!(LAST&&LAST.inference&&LAST.inference.paused);   // toggle
  actmsg(paused?'pausing generation…':'resuming generation…');
  const r=await postJSON('/api/pause',{paused});
  if(LAST&&LAST.inference) LAST.inference.paused=!!(r.ok?r.paused:!paused);  // reflect immediately
  actmsg(r.ok?(r.paused?'⏸ generation paused — deleting predictions in LS won’t regenerate them'
                       :'▶ generation resumed'):'✗ '+(r.error||'failed'));
  updateButtons(LAST);
};
function updateButtons(d){
  const training=!!(d.progress&&d.progress.train);
  const bt=document.getElementById('btnTrain'), br=document.getElementById('btnRefresh');
  bt.disabled=training; bt.textContent=training?'⏳ training…':'⚡ Force train now';
  br.disabled=!d.serving;
  const paused=!!(d.inference&&d.inference.paused);
  const bp=document.getElementById('btnPause');
  bp.textContent=paused?'▶ Resume generating':'⏸ Pause generating';
  bp.style.borderColor=paused?'#d29922':''; bp.style.color=paused?'#d29922':'';
}
let SA_T=null;
function saMsg(t){ const e=document.getElementById('saMsg'); e.textContent=t;
  if(SA_T) clearTimeout(SA_T); SA_T=setTimeout(()=>{e.textContent='';},8000); }
async function postJSON(url,obj){ try{
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj||{})});
  return await r.json(); }catch(e){ return {ok:false,error:String(e)}; } }
function syncSahi(d){               // reflect server config, but never clobber a field being edited
  const inf=d.inference||{}, on=document.getElementById('saOn'), sz=document.getElementById('saSize'), ov=document.getElementById('saOv');
  if(document.activeElement!==on) on.checked=inf.sliced!==false;
  if(document.activeElement!==sz) sz.value=inf.slice??1024;
  if(document.activeElement!==ov) ov.value=inf.overlap??0.2;
}
document.getElementById('saApply').onclick=async()=>{
  const body={sliced:document.getElementById('saOn').checked,
              slice:+document.getElementById('saSize').value,
              overlap:+document.getElementById('saOv').value};
  saMsg('saving…'); const r=await postJSON('/api/sahi',body);
  saMsg(r.ok?('✓ saved: '+(r.inference.sliced?('sliced '+r.inference.slice+'px / '+Math.round(r.inference.overlap*100)+'%'):'single-pass')+' — applies to next prediction; click "Refresh predictions" to re-push now')
            :'✗ '+(r.error||'failed'));
};
document.getElementById('saRec').onclick=async()=>{
  saMsg('measuring images…'); const r=await postJSON('/api/sahi/recommend',{});
  if(r.ok){ const i=r.inference;
    document.getElementById('saOn').checked=i.sliced!==false;
    document.getElementById('saSize').value=i.slice;
    document.getElementById('saOv').value=i.overlap;
    const m=r.median_image?(' from '+r.median_image[0]+'×'+r.median_image[1]+' images'):'';
    saMsg('★ recommended + applied: '+(i.sliced?('sliced '+i.slice+'px'+m):'single-pass (small images)')+' — click "Refresh predictions" to re-push');
  } else saMsg('✗ '+(r.error||'failed')); };

// ---- per-class confidence thresholds ----
let TH_T=null, THRESH_SIG=null;
function thMsg(t){ const e=document.getElementById('thMsg'); e.textContent=t;
  if(TH_T) clearTimeout(TH_T); TH_T=setTimeout(()=>{e.textContent='';},8000); }
function buildThresholds(d){
  const cls=d.classes||[], inf=d.inference||{}, cc=inf.class_conf||{}, def=inf.conf??0.25;
  const thDef=document.getElementById('thDef'), grid=document.getElementById('thGrid');
  if(document.activeElement!==thDef) thDef.value=def;
  const sig=cls.join('|');
  if(sig!==THRESH_SIG){     // rebuild only when the class set changes
    THRESH_SIG=sig;
    grid.innerHTML=cls.length? cls.map(c=>`<label class="throw"><span title="${esc(c)}">${esc(c)}</span>`+
      `<input type="number" class="num thcls" data-cls="${esc(c)}" min="0" max="1" step="0.05" value="${cc[c]??''}" placeholder="${def}"></label>`).join('')
      :'<span class="muted">no classes yet — add labels in Label Studio</span>';
  } else {                  // sync values for inputs not being edited
    grid.querySelectorAll('input.thcls').forEach(inp=>{ if(document.activeElement!==inp){
      const v=cc[inp.dataset.cls]; inp.value=(v==null?'':v); inp.placeholder=def; }});
  }
}
document.getElementById('thToggle').onclick=()=>{
  const g=document.getElementById('thGrid'), sh=g.style.display==='none';
  g.style.display=sh?'grid':'none';
  document.getElementById('thToggle').textContent=sh?'per-class ▴':'per-class ▾';
};
document.getElementById('thApply').onclick=async()=>{
  const def=parseFloat(document.getElementById('thDef').value);
  const conf=isNaN(def)?0.25:def, cc={};
  document.querySelectorAll('input.thcls').forEach(inp=>{ if(inp.value!=='') cc[inp.dataset.cls]=parseFloat(inp.value); });
  thMsg('saving…'); const r=await postJSON('/api/thresholds',{conf:conf,class_conf:cc});
  thMsg(r.ok?('✓ saved (default '+conf+', '+Object.keys(cc).length+' per-class) — applies to next prediction; click "Refresh predictions" to re-push')
            :'✗ '+(r.error||'failed'));
};
// ---- custom tooltips: line-chart crosshair + simple data-tip (delegated, so
//      it survives the grid being re-rendered every tick) ----
document.addEventListener('pointermove',e=>{
  const t=e.target, hit=t.closest?t.closest('.hit[data-cid]'):null;
  if(hit){ lineHover(hit,e); return; }
  const el=t.closest?t.closest('[data-tip]'):null;
  if(el){ document.querySelectorAll('.cross').forEach(c=>c.style.display='none');
          showTip(el.getAttribute('data-tip'),e.clientX,e.clientY); return; }
  hideTip();
});
document.addEventListener('pointerleave',hideTip);
window.addEventListener('scroll',hideTip,true);

// ---- delete a run's data (delegated; rows are re-rendered each tick) ----
document.addEventListener('click',async e=>{
  const b=e.target.closest?e.target.closest('.delrun'):null; if(!b) return;
  const n=b.getAttribute('data-del');
  if(!confirm('Delete run #'+n+'?\n\nThis removes its history record and checkpoint file and hides it from the dashboard. The raw training.log is kept as an audit trail. This cannot be undone.')) return;
  b.disabled=true; actmsg('deleting run #'+n+'…');
  const r=await post('/api/run/'+n+'/delete');
  if(r.ok){ const ck=r.deleted_checkpoints||[];
    actmsg('✓ deleted run #'+n+(ck.length?' (removed '+ck.join(', ')+')':''));
    LASTSIG=null; tick(); }
  else { actmsg('✗ '+(r.error||'failed')); b.disabled=false; }
});

async function tick(){
 let d; try{ d=await (await fetch('/api/status')).json() }catch(e){ return }
 LAST=d; document.getElementById('upd').textContent='updated '+new Date().toLocaleTimeString();
 updateButtons(d); syncSahi(d); buildThresholds(d);
 const sig=JSON.stringify(d);            // skip re-render when nothing changed (keeps dropdown stable while idle)
 if(sig!==LASTSIG){ LASTSIG=sig; render(d); }
}
tick(); setInterval(tick,3000);
</script></body></html>
"""

def _lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


if __name__ == "__main__":
    ip = _lan_ip()
    print(f"Dashboard — served on the local network (LS={LS_URL}, ML={ML_URL})")
    print(f"  this machine : http://localhost:{DASH_PORT}")
    print(f"  on your LAN  : http://{ip}:{DASH_PORT}")
    app.run(host="0.0.0.0", port=DASH_PORT, debug=False, use_reloader=False)
