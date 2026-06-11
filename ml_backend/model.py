"""
Label Studio ML backend (label-studio-ml 1.0.9 API) for construction-site
object detection using a CUSTOM detector trained FROM SCRATCH (no pretrained
weights).

  predict(tasks)            -> list of pre-annotations (one per task)
  fit((), event=, data=)    -> called via /webhook on each annotation event;
                               retrains from scratch every RETRAIN_EVERY images.

Image resolution: we run on the same machine as Label Studio with Local Storage,
so a task URL like  /data/local-files/?d=images/foo.jpeg  is mapped directly to
  $LOCAL_FILES_DOCUMENT_ROOT / images/foo.jpeg
(fast, no HTTP download even when training on the whole set).
"""
from __future__ import annotations

import os
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import requests
from label_studio_ml.model import LabelStudioMLBase
from PIL import Image

import detector as D
import extra_data
import progress
from trainlog import get_logger

log = get_logger()

RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY", "25"))
LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")
LS_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
DOC_ROOT = Path(os.getenv("LOCAL_FILES_DOCUMENT_ROOT",
                          str(Path(__file__).resolve().parent.parent)))


class ScratchDetBackend(LabelStudioMLBase):
    # ------------------------------------------------------------------ #
    # Config / parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_control(label_config: str):
        root = ET.fromstring(label_config)
        rect = root.find(".//RectangleLabels")
        if rect is None:
            return "label", "image", []
        from_name = rect.get("name", "label")
        to_name = rect.get("toName", "image")
        classes = [el.get("value") for el in rect.findall("Label") if el.get("value")]
        return from_name, to_name, classes

    def _control(self, project_id=None):
        """Prefer the label_config handed to us; else fetch it from the API."""
        if self.label_config:
            return self._parse_control(self.label_config)
        if project_id is not None and LS_API_KEY:
            r = requests.get(f"{LS_URL}/api/projects/{project_id}",
                             headers={"Authorization": f"Token {LS_API_KEY}"}, timeout=30)
            if r.ok and r.json().get("label_config"):
                return self._parse_control(r.json()["label_config"])
        return "label", "image", []

    @staticmethod
    def _project_id(data):
        if isinstance(data, dict):
            proj = data.get("project")
            if isinstance(proj, dict) and proj.get("id"):
                return proj["id"]
            if data.get("project_id"):
                return data["project_id"]
        return None

    def _resolve_image(self, img_url: str):
        """Map a task image URL to a local file path."""
        if not img_url:
            return None
        if "local-files" in img_url and "d=" in img_url:
            rel = unquote(img_url.split("d=", 1)[1].split("&")[0])
            return str(DOC_ROOT / rel)
        if img_url.startswith("/data/upload") or img_url.startswith("/data/"):
            try:
                return self.get_local_path(img_url)
            except Exception:
                return None
        if img_url.startswith("http"):
            try:
                return self.get_local_path(img_url)
            except Exception:
                return None
        return img_url  # already a local path

    def _export_annotated_tasks(self, project_id: int):
        r = requests.get(
            f"{LS_URL}/api/projects/{project_id}/export",
            params={"exportType": "JSON", "download_all_tasks": "false"},
            headers={"Authorization": f"Token {LS_API_KEY}"},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def _tasks_to_samples(self, tasks):
        samples = []
        for t in tasks:
            results = []
            for a in t.get("annotations") or []:
                if a.get("was_cancelled") or a.get("skipped"):
                    continue
                results.extend(a.get("result") or [])
            if not results:
                continue
            local = self._resolve_image((t.get("data") or {}).get("image"))
            if not local or not Path(local).exists():
                continue
            samples.append({"image_path": local, "results": results})
        return samples

    # ------------------------------------------------------------------ #
    # Prediction (pre-annotations)
    # ------------------------------------------------------------------ #
    def predict(self, tasks, **kwargs):
        state = D.load_state()
        weights = state.get("active_weights")
        classes = state.get("classes") or []
        if not weights or not Path(weights).exists():
            return []   # no model yet -> annotate from blank

        # Generation can be paused from the dashboard. When paused, generate
        # nothing — so deleting predictions in Label Studio (which makes LS call
        # back here to re-fetch) won't silently regenerate them.
        if D.load_inference().get("paused"):
            log.info("  prediction generation is PAUSED (dashboard); skipping")
            return []

        from_name, to_name, _ = self._control()
        model_version = f"scratchdet-r{state.get('train_runs', 0)}"
        predictions = []
        total = len(tasks)
        try:
            for i, task in enumerate(tasks):
                inf = D.load_inference()       # live config (dashboard-adjustable)
                # Re-check each task so a pause clicked mid-run halts generation.
                if inf.get("paused"):
                    log.info(f"  generation PAUSED from dashboard; stopping at "
                             f"{i}/{total} tasks")
                    break
                local = self._resolve_image((task.get("data") or {}).get("image"))
                if not local or not Path(local).exists():
                    predictions.append({"result": [], "score": 0.0})
                    progress.set_infer(i + 1, total, version=model_version)
                    continue
                img = Image.open(local).convert("RGB")
                w, h = img.size
                conf0 = float(inf.get("conf", 0.25))
                cc = inf.get("class_conf") or {}
                # detect down to the lowest active threshold, then filter per class
                base = max(0.01, min([conf0] + [float(v) for v in cc.values()]))
                dets = (D.predict_boxes_sliced(weights, img, conf=base, iou=0.45,
                                               slice_size=int(inf["slice"]),
                                               overlap=float(inf["overlap"]))
                        if inf.get("sliced") else
                        D.predict_boxes(weights, img, conf=base, iou=0.45))
                items, scores = [], []
                for cls_id, score, x1, y1, x2, y2 in dets:
                    name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
                    if score < float(cc.get(name, conf0)):   # per-class threshold
                        continue
                    items.append({
                        "from_name": from_name, "to_name": to_name,
                        "type": "rectanglelabels",
                        "original_width": w, "original_height": h, "image_rotation": 0,
                        "value": {
                            "x": x1 / w * 100, "y": y1 / h * 100,
                            "width": (x2 - x1) / w * 100, "height": (y2 - y1) / h * 100,
                            "rotation": 0, "rectanglelabels": [name],
                        },
                        "score": score,
                    })
                    scores.append(score)
                predictions.append({
                    "result": items,
                    "score": sum(scores) / len(scores) if scores else 0.0,
                    "model_version": model_version,
                })
                progress.set_infer(i + 1, total, version=model_version)
        finally:
            progress.clear_infer()
        return predictions

    # ------------------------------------------------------------------ #
    # Event dispatch
    # ------------------------------------------------------------------ #
    # The base class only routes its built-in TRAIN_EVENTS (ANNOTATION_*) to
    # fit(); a forced "START_TRAINING" (from the dashboard button / force_train.py)
    # is otherwise dropped. Handle it explicitly so manual retrains actually run.
    def process_event(self, event, data, job_id, additional_params):
        if event == "START_TRAINING":
            return self.fit((), event=event, data=data, job_id=job_id, **additional_params)
        return super().process_event(event, data, job_id, additional_params)

    # ------------------------------------------------------------------ #
    # Training (triggered by /webhook on every annotation event)
    # ------------------------------------------------------------------ #
    def fit(self, tasks, event=None, data=None, **kwargs):
        if event not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED", "START_TRAINING"):
            return {"status": "ignored", "event": event}
        if not LS_API_KEY:
            log.warning("LABEL_STUDIO_API_KEY not set; cannot export annotations.")
            return {"status": "error", "detail": "LABEL_STUDIO_API_KEY not set"}

        data = data or {}
        project_id = self._project_id(data)
        if project_id is None:
            return {"status": "no_project_id"}

        state = D.load_state()
        forced = event == "START_TRAINING"

        # Count this as one annotation EVENT (a submit or an edit). Retraining
        # fires every RETRAIN_EVERY events since the last training, so re-labeling
        # images you've already annotated (e.g. adding new classes) counts too.
        if event in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
            ann = data.get("annotation") if isinstance(data, dict) else None
            n_boxes = len([r for r in ((ann or {}).get("result") or [])
                           if r.get("type") == "rectanglelabels"])
            verb = "created" if event == "ANNOTATION_CREATED" else "updated"
            state["events_since_train"] = state.get("events_since_train", 0) + 1
            D.save_state(state)
            log.info(f"EVENT {event}: task {(ann or {}).get('task')} {verb} "
                     f"with {n_boxes} boxes "
                     f"({state['events_since_train']}/{RETRAIN_EVERY} since last train)")

        events = state.get("events_since_train", 0)
        if not forced and events < RETRAIN_EVERY:
            log.info(f"  waiting: {events}/{RETRAIN_EVERY} annotation events since "
                     f"last train ({RETRAIN_EVERY - events} more to go)")
            return {"status": "waiting", "events_since_train": events,
                    "retrain_every": RETRAIN_EVERY}

        # Only one training at a time. Training runs in the job manager's
        # subprocess, so an in-process lock can't guard it — use the shared
        # training heartbeat instead (fresh heartbeat = a run is active).
        if progress.train_is_active(60):
            log.info("  a training run is already in progress; ignoring this trigger")
            return {"status": "already_training"}
        progress.set_train(0, 0, 1, phase="preparing")   # claim the slot immediately
        try:
            # Pull the full labeled set as the source of truth.
            all_tasks = self._export_annotated_tasks(project_id)
            samples = self._tasks_to_samples(all_tasks)

            # Merge any pre-existing annotations dropped into extra_data/ (deduped
            # by filename; Label Studio annotations win over the dropped-in copy).
            try:
                seen = {Path(s["image_path"]).name.lower() for s in samples}
                added = 0
                for s in extra_data.load_extra_samples(log=log):
                    key = Path(s["image_path"]).name.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    samples.append(s)
                    added += 1
                if added:
                    log.info(f"  merged {added} pre-annotated image(s) from extra_data/")
            except Exception as e:
                log.warning(f"  extra_data merge failed: {e}")

            completed = len(samples)
            state["annotations_seen"] = completed

            _, _, classes = self._control(project_id)
            if not classes or completed == 0:
                log.info("  skipped: no classes or no labeled data yet")
                D.save_state(state)
                progress.clear_train()
                return {"status": "skipped", "reason": "no classes or no labeled data"}

            log.info(f"TRIGGER: {events} annotation events reached {RETRAIN_EVERY} "
                     f"(forced={forced}) -> training from scratch on {completed} images")
            state = D.train(samples, classes, state)
            state["events_since_train"] = 0          # reset the counter after training
            D.save_state(state)

            # Push the fresh model's predictions into Label Studio automatically:
            # point the project at the new version and re-fetch every task. Done in
            # a background thread so this webhook returns first (the re-fetch calls
            # back into /predict on this same backend).
            version = f"scratchdet-r{state.get('train_runs', 0)}"
            self._spawn_refresh(project_id, version)

            return {
                "status": "trained",
                "images": completed,
                "events": events,
                "classes": classes,
                "metrics": state["last_metrics"],
                "weights": state["active_weights"],
                "model_version": version,
            }
        except Exception:
            progress.clear_train()   # release the guard if training errors out
            raise

    # ------------------------------------------------------------------ #
    # Auto-refresh predictions in Label Studio after a retrain
    # ------------------------------------------------------------------ #
    def _spawn_refresh(self, project_id: int, version: str):
        def work():
            try:
                time.sleep(1.0)   # let fit() return so the worker is free
                h = {"Authorization": f"Token {LS_API_KEY}", "Content-Type": "application/json"}
                sel = {"selectedItems": {"all": True, "excluded": []}}
                requests.patch(f"{LS_URL}/api/projects/{project_id}", headers=h,
                               json={"model_version": version}, timeout=30)
                # Clear old predictions first so the fresh model's boxes actually
                # replace them (LS dedupes by model_version) and stale versions
                # don't accumulate.
                requests.post(f"{LS_URL}/api/dm/actions",
                              params={"id": "delete_tasks_predictions", "project": project_id},
                              headers=h, json=sel, timeout=300)
                requests.post(f"{LS_URL}/api/dm/actions",
                              params={"id": "retrieve_tasks_predictions", "project": project_id},
                              headers=h, json=sel, timeout=900)
                log.info(f"  auto-refreshed Label Studio predictions -> {version}")
            except Exception as e:
                log.warning(f"  auto-refresh of predictions failed ({e}); "
                            f"run scripts/refresh_predictions.* to do it manually")
        threading.Thread(target=work, daemon=True).start()


# Seed data/inference.json from env defaults on first run (preserves any values
# already set from the dashboard). The dashboard reads + writes this file to show
# and adjust SAHI sliced inference live.
try:
    D.save_inference(D.load_inference())
except Exception:
    pass
