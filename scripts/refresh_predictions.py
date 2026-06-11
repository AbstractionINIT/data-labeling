"""
Push the latest trained model's predictions into Label Studio.

The backend now does this automatically after each retrain, but run this any
time you want to force a refresh (e.g. after restoring a checkpoint). It
AUTO-DETECTS the newest model version from data/state.json — no arguments.

    python scripts/refresh_predictions.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "state.json"
LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8090").rstrip("/")
API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
PROJECT_TITLE = os.getenv("PROJECT_TITLE", "Construction Site Detection")
H = {"Authorization": f"Token {API_KEY}", "Content-Type": "application/json"}


def main():
    if not API_KEY:
        sys.exit("LABEL_STUDIO_API_KEY not set (source scripts/env.ps1 or env.sh first).")
    if not STATE.exists():
        sys.exit("No data/state.json yet — train at least once first.")
    runs = json.loads(STATE.read_text()).get("train_runs", 0)
    if runs < 1:
        sys.exit("No trained model yet (train_runs=0).")
    version = f"scratchdet-r{runs}"

    r = requests.get(f"{LS_URL}/api/projects", headers=H, timeout=30)
    r.raise_for_status()
    items = r.json().get("results", r.json() if isinstance(r.json(), list) else [])
    proj = next((p for p in items if p.get("title") == PROJECT_TITLE), items[0] if items else None)
    if not proj:
        sys.exit(f"Project '{PROJECT_TITLE}' not found in Label Studio.")
    pid = proj["id"]

    print(f"Refreshing project '{proj['title']}' (id={pid}) -> {version} ...")
    requests.patch(f"{LS_URL}/api/projects/{pid}", headers=H,
                   json={"model_version": version}, timeout=30).raise_for_status()
    sel = {"selectedItems": {"all": True, "excluded": []}}
    # Delete existing predictions first: Label Studio dedupes by model_version, so
    # without this a re-fetch of the SAME version keeps the old boxes (and stale
    # older versions pile up). Predictions are not annotations — safe to clear.
    dele = requests.post(f"{LS_URL}/api/dm/actions",
                         params={"id": "delete_tasks_predictions", "project": pid},
                         headers=H, json=sel, timeout=300)
    if dele.ok:
        print(f"  cleared old predictions: {dele.json().get('detail', '')}")
    resp = requests.post(f"{LS_URL}/api/dm/actions",
                         params={"id": "retrieve_tasks_predictions", "project": pid},
                         headers=H, json=sel, timeout=900)
    resp.raise_for_status()
    print(f"  {resp.json().get('detail', resp.text)}")
    print(f"Done. Label Studio now shows {version}. Reopen a task to see the new boxes.")


if __name__ == "__main__":
    main()
