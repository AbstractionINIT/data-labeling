"""
One-shot setup of the Label Studio project, fully via the REST API.

It will:
  1. Create the object-detection project with the labeling config.
  2. Register the images/ folder as Local Storage and sync it (imports 158 tasks).
  3. Connect the YOLO ML backend and enable "train on annotation submit".

Prerequisites (see SETUP.md):
  * Label Studio is running on $LABEL_STUDIO_URL with local file serving enabled:
        LOCAL_FILES_SERVING_ENABLED=true
        LOCAL_FILES_DOCUMENT_ROOT=d:\annotation
  * The ML backend is running on $ML_BACKEND_URL (default http://localhost:9090).
  * Env vars set:  LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY

Run:  python scripts/bootstrap_project.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

LS_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "")
ML_URL = os.environ.get("ML_BACKEND_URL", "http://localhost:9090").rstrip("/")
PROJECT_TITLE = os.environ.get("PROJECT_TITLE", "Construction Site Detection")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
CONFIG_PATH = Path(__file__).resolve().parent / "labeling_config.xml"

H = {"Authorization": f"Token {API_KEY}", "Content-Type": "application/json"}


def die(msg: str):
    print(f"ERROR: {msg}")
    sys.exit(1)


def get_or_create_project() -> int:
    if not API_KEY:
        die("LABEL_STUDIO_API_KEY is not set (LS > Account & Settings > Access Token).")
    # Reuse an existing project with the same title if present.
    r = requests.get(f"{LS_URL}/api/projects", headers=H, timeout=30)
    r.raise_for_status()
    for p in r.json().get("results", []):
        if p["title"] == PROJECT_TITLE:
            print(f"Reusing existing project id={p['id']}")
            return p["id"]
    label_config = CONFIG_PATH.read_text(encoding="utf-8")
    r = requests.post(
        f"{LS_URL}/api/projects",
        headers=H,
        json={
            "title": PROJECT_TITLE,
            "label_config": label_config,
            "description": "Construction-site object detection with auto-retraining YOLO.",
        },
        timeout=30,
    )
    r.raise_for_status()
    pid = r.json()["id"]
    print(f"Created project id={pid}")
    return pid


def add_local_storage(project_id: int):
    # Path is relative to LOCAL_FILES_DOCUMENT_ROOT on the LS server.
    r = requests.get(
        f"{LS_URL}/api/storages/localfiles",
        headers=H,
        params={"project": project_id},
        timeout=30,
    )
    if r.ok and r.json():
        sid = r.json()[0]["id"]
        print(f"Local storage already exists id={sid}; re-syncing.")
    else:
        r = requests.post(
            f"{LS_URL}/api/storages/localfiles",
            headers=H,
            json={
                "project": project_id,
                "title": "images-folder",
                "path": str(IMAGES_DIR),
                "regex_filter": r".*\.(jpe?g|png|bmp|tiff?)$",
                "use_blob_urls": True,  # serve as images, not as text tasks
            },
            timeout=30,
        )
        if not r.ok:
            die(
                "Failed to create local storage. Make sure Label Studio was started "
                "with LOCAL_FILES_SERVING_ENABLED=true and "
                f"LOCAL_FILES_DOCUMENT_ROOT set to a parent of {IMAGES_DIR}.\n{r.text}"
            )
        sid = r.json()["id"]
        print(f"Created local storage id={sid}")
    r = requests.post(
        f"{LS_URL}/api/storages/localfiles/{sid}/sync", headers=H, timeout=120
    )
    r.raise_for_status()
    print("Synced local storage (images imported as tasks).")


def connect_ml_backend(project_id: int):
    r = requests.get(
        f"{LS_URL}/api/ml", headers=H, params={"project": project_id}, timeout=30
    )
    if r.ok and any(b.get("url") == ML_URL for b in r.json()):
        print("ML backend already connected.")
        ml_id = next(b["id"] for b in r.json() if b.get("url") == ML_URL)
    else:
        r = requests.post(
            f"{LS_URL}/api/ml",
            headers=H,
            json={
                "project": project_id,
                "title": "ScratchDet (from-scratch detector)",
                "url": ML_URL,
                "is_interactive": False,
            },
            timeout=60,
        )
        if not r.ok:
            die(f"Failed to connect ML backend. Is it running at {ML_URL}?\n{r.text}")
        ml_id = r.json()["id"]
        print(f"Connected ML backend id={ml_id}")

    # Training-on-submit: connecting an ML backend AUTO-CREATES a webhook
    # (send_for_all_actions=true) that POSTs annotation events to the backend's
    # /webhook -> fit(). So training already fires; this PATCH is a best-effort
    # extra for older LS versions and is harmless if the field doesn't exist.
    requests.patch(
        f"{LS_URL}/api/ml/{ml_id}",
        headers=H,
        json={"start_training_on_annotation_update": True},
        timeout=30,
    )
    # Verify the training webhook exists (this is what actually drives retraining).
    wh = requests.get(f"{LS_URL}/api/webhooks/", headers=H,
                      params={"project": project_id}, timeout=30)
    if wh.ok and any(ML_URL in (w.get("url") or "") for w in wh.json()):
        print("Training webhook is active -> model retrains on annotation submit.")
    else:
        print("WARNING: no webhook to the ML backend was found; retraining may "
              "not fire automatically. Check Project > Settings > Webhooks.")
    # Show model pre-annotations AND auto-fetch them from the backend when a task
    # loads, so predictions appear without a manual "Retrieve Predictions" step.
    requests.patch(
        f"{LS_URL}/api/projects/{project_id}",
        headers=H,
        json={"show_collab_predictions": True,
              "evaluate_predictions_automatically": True},
        timeout=30,
    )


def main():
    pid = get_or_create_project()
    add_local_storage(pid)
    connect_ml_backend(pid)
    print("\nDone. Open Label Studio, start annotating, and the model will")
    print(f"retrain automatically every {os.getenv('RETRAIN_EVERY', '25')} annotations.")


if __name__ == "__main__":
    main()
