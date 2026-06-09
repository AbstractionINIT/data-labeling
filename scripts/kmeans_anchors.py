"""
(Optional) Refine the anchor priors to match YOUR boxes.

The defaults in model_arch.DEFAULT_ANCHORS are generic. Once you have a batch of
annotations, run this to k-means the box width/heights and print better anchors,
then paste them into model_arch.DEFAULT_ANCHORS.

    python scripts/kmeans_anchors.py <project_id> [k]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_backend"))

LS_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/kmeans_anchors.py <project_id> [k]")
        sys.exit(1)
    project_id = int(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    r = requests.get(
        f"{LS_URL}/api/projects/{project_id}/export",
        params={"exportType": "JSON"},
        headers={"Authorization": f"Token {API_KEY}"},
        timeout=120,
    )
    r.raise_for_status()
    wh = []
    for t in r.json():
        for a in t.get("annotations") or []:
            for res in a.get("result") or []:
                v = res.get("value", {})
                if "width" in v and "height" in v:
                    wh.append([v["width"] / 100.0, v["height"] / 100.0])
    wh = np.array(wh)
    if len(wh) < k:
        print(f"Need at least {k} boxes, have {len(wh)}.")
        sys.exit(1)

    # simple k-means (Lloyd) on normalized w,h
    rng = np.random.default_rng(0)
    centers = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(100):
        d = ((wh[:, None, :] - centers[None]) ** 2).sum(-1)
        assign = d.argmin(1)
        new = np.array([wh[assign == i].mean(0) if (assign == i).any() else centers[i]
                        for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new

    centers = centers[centers[:, 0].argsort()]
    print(f"Refined anchors from {len(wh)} boxes:")
    print("DEFAULT_ANCHORS = [", ", ".join(f"({w:.3f}, {h:.3f})" for w, h in centers), "]")


if __name__ == "__main__":
    main()
