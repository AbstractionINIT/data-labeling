# Handover — Annotation + Incremental Training Pipeline

Label Studio + a from-scratch PyTorch detector (**ScratchDet**) with sliced (SAHI-style)
inference and a Flask dashboard. Webhook-driven retraining on an RTX 2070 Super (CUDA).

## Current state (2026-06-11)
- **Run #3 complete.** 25 classes, 1368 images, trained on **all** data (no held-out split).
  best_val_loss 2.55, 100 epochs, ~47 min. Active weights: `data/checkpoints/scratchdet_small_r003.pt`.
- Predictions in Label Studio are tagged **scratchdet-r3** (~79.94% on the panorama).
- Inference: **sliced ON**, slice 1024, overlap 0.2, conf 0.5 (`data/inference.json`).

## Servers (3 terminals, or `scripts/start_all.ps1`)
| Service | Port | Start |
|---|---|---|
| Label Studio | 8090 | `scripts/start_label_studio.ps1` |
| ML backend | 9090 | `scripts/start_ml_backend.ps1` |
| Dashboard | 9091 | `scripts/start_dashboard.ps1` |

All bind `0.0.0.0` for LAN access. LS host auto-detects the LAN IP.

## Key files
- `ml_backend/model.py` — LS ML wrapper. `predict()` (per-class thresholds, sliced),
  `fit()` (cross-process train guard via heartbeat), `process_event()` (handles
  **START_TRAINING** so the dashboard's Force-Train works), `_spawn_refresh()`
  (PATCH model_version + delete/retrieve predictions after a run).
- `ml_backend/detector.py` — the detector. Training, `predict_boxes_sliced()`,
  letterbox disk cache (`data/cache/letterbox`), per-batch progress heartbeat + ETA,
  run-history (`data/history.json`).
- `ml_backend/progress.py` — shared live progress (`data/progress.json`).
  `train_is_active(stale=60)` is the **cross-process** concurrency guard.
- `ml_backend/extra_data.py` — drop-in importer. Put images+labels in
  `extra_data/` (YOLO/COCO/VOC/LS-JSON auto-detected) → merged into the next run.
- `scripts/dashboard.py` — Flask dashboard: SVG charts with **custom hover tooltips**
  (line-chart crosshair + data-tip), live train/infer progress + ETA,
  Force-Train / Refresh buttons, **Pause/Resume generating** (`paused` flag in
  `inference.json`; `predict()` short-circuits so LS deletes don't regenerate and an
  in-progress run halts), SAHI controls (slice/overlap/recommend),
  per-class confidence thresholds. Comparison views: **all-runs performance**
  (metric-selectable val/train/obj/box/cls per epoch), **best-val-per-run** (full-width
  bar), **boxes-per-class-per-run heatmap**. **Delete a run** from the run-history table
  (`🗑`): drops its history record, deletes its checkpoint, and hides it via
  `data/deleted_runs.json` (the active served model is protected; the raw
  `training.log` is kept). A future retrain reusing the number reappears.
  **Interrupted-run detection**: a run with no `best_val_loss` that isn't the one
  currently training was killed mid-run → a warning banner + a per-row `↻ Re-run`
  button (both just trigger a fresh from-scratch run via `/api/train`, since training
  has no resume — it always restarts on all current data with the next run number).

## Runtime data (gitignored)
`data/state.json` (run count, active weights, classes) · `data/history.json` (run records) ·
`data/progress.json` (live) · `data/inference.json` (SAHI + thresholds) ·
`data/checkpoints/*.pt` · `data/cache/letterbox/*.npz`.

## Hard-won gotchas
- **Training runs in a multiprocessing subprocess.** Killing the backend *orphans* the
  training worker — it keeps running. An in-process lock can't guard it; that's why the
  guard is the `progress.json` heartbeat. Cleanup must also kill the fork workers.
- **Predictions are read-only in LS.** To edit, copy a prediction → annotation, or enable
  **Settings → Annotation → "Use predictions to prelabel tasks"** pinned to **scratchdet-r3**.
  Only annotations created *after* that setting is on get pre-filled.
- **Image URLs must be relative** (`/data/local-files/?d=images/<file>`, forward slashes) or
  LAN access 401s on the session cookie. `scripts/fix_image_urls.py` rewrites them.
- **PowerShell 5.1:** `$pid` is read-only (use `$procId`); em-dash (—) in `.ps1` files
  corrupts parsing under cp1252 — keep scripts ASCII.

## Class mapping note
data.yaml's 21 classes were mapped to LS: 8 exact, Bobcat→`bob_cat`, 13 new added.
User chose to keep `moxy`/`haul_truck`/`dump_truck` and `fork_lift`/`telehandler` **separate**
(not merged). Project now has 25 classes.

## Open / next
- **Not committed yet** — the whole session's work is unstaged. Commit when ready.
- **Optional improvement:** tiled *training* (model currently sees whole 7571px panoramas
  downsampled to 512, so small objects train weakly; SAHI only fixes *inference*).
