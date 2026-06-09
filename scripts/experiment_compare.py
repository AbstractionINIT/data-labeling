"""
Architecture-selection experiment (the "try multiple, keep the best" step).

Within the from-scratch constraint, the variable we sweep is the ARCHITECTURE
SIZE of our custom detector: 'tiny' vs 'small' vs 'medium'. Each is trained from
random init on the SAME current annotations and ranked by validation loss.

Prereqs: Label Studio + ML backend running, env loaded, and at least one batch
annotated. Run:

    python scripts/experiment_compare.py <project_id>
    python scripts/experiment_compare.py <project_id> tiny small   # custom subset

Find the project id in the LS URL: .../projects/<id>/data
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

# import the backend package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_backend"))
import detector as D  # noqa: E402

LS_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "")


def export_samples(project_id: int):
    r = requests.get(
        f"{LS_URL}/api/projects/{project_id}/export",
        params={"exportType": "JSON", "download_all_tasks": "false"},
        headers={"Authorization": f"Token {API_KEY}"},
        timeout=120,
    )
    r.raise_for_status()
    samples, classes = [], set()
    for t in r.json():
        results = []
        for a in t.get("annotations") or []:
            if a.get("was_cancelled"):
                continue
            results.extend(a.get("result") or [])
        if not results:
            continue
        # resolve local path the same way LS local storage serves it
        img = (t.get("data") or {}).get("image", "")
        # local-files URLs look like /data/local-files/?d=images/foo.jpg
        if "d=" in img:
            rel = img.split("d=", 1)[1]
            from urllib.parse import unquote
            local = Path(os.environ.get("LOCAL_FILES_DOCUMENT_ROOT", "")) / unquote(rel)
        else:
            local = Path(img)
        for r_ in results:
            for lb in (r_.get("value", {}).get("rectanglelabels") or []):
                classes.add(lb)
        samples.append({"image_path": str(local), "results": results})
    return samples, sorted(classes)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/experiment_compare.py <project_id> [variants...]")
        sys.exit(1)
    project_id = int(sys.argv[1])
    variants = sys.argv[2:] or ["tiny", "small", "medium"]

    samples, classes = export_samples(project_id)
    if not samples:
        print("No annotated data found yet.")
        sys.exit(1)
    print(f"Comparing {variants} on {len(samples)} images, classes={classes}\n")

    results = []
    for v in variants:
        print(f"=== Training variant '{v}' from scratch ===")
        state = {"train_runs": 0, "annotations_seen": len(samples),
                 "last_trained_at": 0, "variant": v}
        state = D.train(samples, classes, state, variant=v)
        results.append((v, state["last_metrics"]["best_val_loss"],
                        state["last_metrics"]["duration_s"]))

    results.sort(key=lambda x: x[1])  # lower val loss = better
    print("\n============== RANKING (by best val loss) ==============")
    print(f"{'variant':<10}{'best_val_loss':>16}{'seconds':>10}")
    for v, loss, dur in results:
        print(f"{v:<10}{loss:>16.4f}{dur:>10.1f}")
    print(f"\nBest: {results[0][0]}  ->  set DET_VARIANT={results[0][0]} in scripts/env.ps1")


if __name__ == "__main__":
    main()
