"""
Manually trigger a retrain right now (ignores the every-25 gate) by POSTing a
START_TRAINING webhook to the running ML backend.

Usage:  python scripts/force_train.py <project_id>
Find the project id in the Label Studio URL: .../projects/<id>/data
"""
import os
import sys

import requests

ML_URL = os.environ.get("ML_BACKEND_URL", "http://localhost:9090").rstrip("/")

if len(sys.argv) < 2:
    print("Usage: python scripts/force_train.py <project_id>")
    sys.exit(1)

project_id = int(sys.argv[1])
r = requests.post(
    f"{ML_URL}/webhook",
    json={"action": "START_TRAINING", "project": {"id": project_id}},
    timeout=600,
)
print(r.status_code, r.text)
