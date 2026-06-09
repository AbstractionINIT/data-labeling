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
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import requests
from label_studio_ml.model import LabelStudioMLBase
from PIL import Image

import detector as D
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

        from_name, to_name, _ = self._control()
        model_version = f"scratchdet-r{state.get('train_runs', 0)}"
        predictions = []
        for task in tasks:
            local = self._resolve_image((task.get("data") or {}).get("image"))
            if not local or not Path(local).exists():
                predictions.append({"result": [], "score": 0.0})
                continue
            img = Image.open(local).convert("RGB")
            w, h = img.size
            dets = D.predict_boxes(weights, img, conf=0.25, iou=0.45)
            items, scores = [], []
            for cls_id, score, x1, y1, x2, y2 in dets:
                name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
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
        return predictions

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

        # Log the CURRENT image's annotation that triggered this event.
        ann = data.get("annotation") if isinstance(data, dict) else None
        if ann:
            n_boxes = len([r for r in (ann.get("result") or [])
                           if r.get("type") == "rectanglelabels"])
            log.info(f"EVENT {event}: task {ann.get('task')} submitted with {n_boxes} boxes")

        # Count completed annotations from the authoritative export.
        all_tasks = self._export_annotated_tasks(project_id)
        samples = self._tasks_to_samples(all_tasks)
        completed = len(samples)
        state["annotations_seen"] = completed
        D.save_state(state)

        forced = event == "START_TRAINING"
        crossed = completed > 0 and (completed // RETRAIN_EVERY) > (
            state["last_trained_at"] // RETRAIN_EVERY)
        if not forced and not crossed:
            nxt = (completed // RETRAIN_EVERY + 1) * RETRAIN_EVERY
            log.info(f"  progress: {completed} annotated images; next retrain at {nxt}")
            return {"status": "waiting", "completed": completed, "next_train_at": nxt}

        _, _, classes = self._control(project_id)
        if not classes or completed == 0:
            return {"status": "skipped", "reason": "no classes or no labeled data"}

        log.info(f"TRIGGER: {completed} annotated images is a multiple of "
                 f"{RETRAIN_EVERY} (forced={forced}) -> training from scratch")
        state = D.train(samples, classes, state)
        return {
            "status": "trained",
            "images": completed,
            "classes": classes,
            "metrics": state["last_metrics"],
            "weights": state["active_weights"],
        }
