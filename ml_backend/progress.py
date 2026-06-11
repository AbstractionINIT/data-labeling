"""
Tiny shared progress reporter for the dashboard.

The trainer (detector.train) and the predictor (model.predict) write their live
progress here; the dashboard reads data/progress.json and renders progress bars.
A section is considered "active" only if its `updated` timestamp is recent — so a
crashed/finished run stops showing a bar without needing an explicit clear.

No locking is needed: a single ML-backend process writes, the dashboard only
reads, and writes are atomic (temp file + replace).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PROGRESS_PATH = Path(__file__).resolve().parent.parent / "data" / "progress.json"
STALE_AFTER = 25.0   # seconds; a section older than this reads as inactive


def _read() -> dict:
    try:
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(d: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d), encoding="utf-8")
    tmp.replace(PROGRESS_PATH)


def set_train(run: int, epoch: int, epochs: int, **extra) -> None:
    d = _read()
    d["train"] = {"run": run, "epoch": epoch, "epochs": epochs,
                  "updated": time.time(), **extra}
    _write(d)


def clear_train() -> None:
    d = _read(); d.pop("train", None); _write(d)


def set_infer(done: int, total: int, **extra) -> None:
    d = _read()
    d["infer"] = {"done": done, "total": total, "updated": time.time(), **extra}
    _write(d)


def clear_infer() -> None:
    d = _read(); d.pop("infer", None); _write(d)


def train_is_active(stale: float = 60.0) -> bool:
    """True if a training heartbeat was written within `stale` seconds. Works
    across processes/subprocesses (the job manager runs training in a subprocess,
    so an in-process lock can't guard it). Self-heals if a run dies (heartbeat
    goes stale)."""
    t = _read().get("train")
    return bool(t and (time.time() - t.get("updated", 0) < stale))
