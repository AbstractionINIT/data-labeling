# 🏗️ Construction-Site Detection — annotate & train from scratch

An end-to-end pipeline to **annotate** construction-site images in
[Label Studio](https://labelstud.io/) while a **custom object detector — built
from raw PyTorch layers and trained from scratch (no pretrained weights)** —
**retrains automatically every 25 completed annotations** and feeds its
predictions back as pre-annotations.

> Annotate → the model retrains itself every 25 images → its boxes pre-fill the
> next images so you correct instead of draw. The loop accelerates labeling as it
> learns.

---

## ✨ Highlights

- **Truly "empty" model** — a custom `ScratchDet` network (Darknet-style CNN +
  anchor grid head) written from `Conv2d`/`BatchNorm`/`SiLU`, randomly
  initialized. Nothing borrowed from COCO or any pretrained checkpoint.
- **Auto-retrain every 25 annotations** — a Label Studio webhook triggers a full
  from-scratch retrain on all labels collected so far.
- **Add classes mid-annotation** — the class list is read live from the project on
  every retrain; the detection head is rebuilt to match.
- **Pre-annotations** — once trained, the model serves boxes back into Label
  Studio so you only correct them.
- **GPU-aware setup** — `setup.sh` auto-detects **NVIDIA (CUDA)**,
  **AMD/Linux (ROCm)**, **AMD/Intel-Windows (DirectML)**, or falls back to CPU.
- **Full training log** — every run records start time, data size, per-image
  annotation counts, per-epoch losses, and a summary; capped at 2 GB with
  automatic rotation.
- **Live dashboard** (http://localhost:9091) — service health, retrain progress,
  annotated-image + per-class counts, a val-loss curve, checkpoints, and the live
  training log, all auto-refreshing.

---

## 🚀 Quick start

```bash
# 1. Install everything into a venv (auto-detects your GPU)
bash setup.sh

# 2. Configure (paste your Label Studio token)
#    Windows: edit scripts\env.ps1      Linux: edit scripts/env.sh

# 3. Three terminals (PowerShell shown; .sh equivalents exist for Linux)
#    Terminal 1 — Label Studio
. .\scripts\env.ps1 ; .\scripts\start_label_studio.ps1
#    Terminal 2 — the detector ML backend
. .\scripts\env.ps1 ; .\scripts\start_ml_backend.ps1
#    Terminal 3 — create project, import images, connect the model
. .\scripts\env.ps1 ; .\.venv\Scripts\python.exe scripts\bootstrap_project.py
```

Then open <http://localhost:8080>, **Label All Tasks**, and start drawing boxes.
The model retrains itself every 25 submissions.

📖 Full step-by-step (incl. GPU notes & troubleshooting): **[SETUP.md](SETUP.md)**

---

## 🧠 The model (in one breath)

```
image 512×512
  └─ Darknet-style CNN backbone (stride 16)
       └─ neck (feature mixing)
            └─ 1×1 head → 32×32 grid × 3 anchors × (box + objectness + classes)
                 └─ decode + per-class NMS → boxes
```

A single-stage, anchor-based detector (YOLO-v2/v3 *family* in spirit, but our own
code), trained with a multipart loss (box + objectness + classification),
AdamW + cosine LR, on letterboxed inputs. Three size variants
(`tiny` / `small` / `medium`) so you can run the "train several, keep the best"
experiment.

📖 Architecture, training approach, and design trade-offs: **[TRAINING.md](TRAINING.md)**

> ⚠️ **Expectation setting:** a from-scratch detector has *no* prior knowledge, so
> it needs **many** labels (realistically hundreds, not 25) before predictions are
> reliably useful. The per-25 loop is what climbs that curve. This is the inherent
> cost of an "empty" model — if you'd rather have usable boxes after ~25 images, a
> pretrained backbone is the alternative.

---

## 📁 Project structure

```
annotation/
├─ setup.sh                 # one-shot installer (GPU auto-detect)
├─ README.md / SETUP.md / TRAINING.md
├─ images/                  # your source images (git-ignored)
├─ ml_backend/
│  ├─ model_arch.py         # the custom network (raw PyTorch)
│  ├─ detector.py           # dataset, target encoding, loss, train loop, NMS
│  ├─ device_util.py        # CUDA / ROCm / DirectML / CPU selection
│  ├─ model.py              # Label Studio ML backend (predict + fit)
│  ├─ trainlog.py           # 2 GB-rotating training logger
│  └─ _wsgi.py              # backend server (port 9090)
├─ scripts/
│  ├─ env.example.{ps1,sh}  # copy → env.{ps1,sh}, paste your LS token
│  ├─ start_label_studio.* / start_ml_backend.*
│  ├─ bootstrap_project.py  # create project + import + connect model
│  ├─ labeling_config.xml   # label classes (edit to add classes)
│  ├─ experiment_compare.py # train tiny/small/medium, keep the best
│  ├─ kmeans_anchors.py     # refine anchor priors to your boxes
│  ├─ status.py / force_train.py
└─ data/                    # runtime: checkpoints, logs, state.json (git-ignored)
```

---

## 🔧 Common tasks

```bash
# watch progress / metrics
python scripts/status.py
# live training log
Get-Content data\logs\training.log -Tail 40 -Wait   # PowerShell
tail -f data/logs/training.log                        # bash

# force a retrain now (id from the Label Studio URL)
python scripts/force_train.py <project_id>

# pick the best architecture once you have ~50+ labels
python scripts/experiment_compare.py <project_id>
```

Key knobs live in `scripts/env.{ps1,sh}`: `RETRAIN_EVERY`, `DET_VARIANT`,
`DET_IMG_SIZE`, `DET_BATCH`, `DET_EPOCHS`, `FORCE_DEVICE`.

---

## 🧰 Tech stack

Python 3.12 · PyTorch (CUDA/ROCm/DirectML) · Label Studio + `label-studio-ml` ·
Pillow · NumPy · a hand-written YOLO-style detector.

## 📝 Notes

- The default classes (`excavator`, `dump_truck`, `crane`, `worker`,
  `concrete_pipe`, `building`, `vehicle`) are placeholders — edit
  `scripts/labeling_config.xml` or the Label Studio UI to match your project.
- `.venv/`, `data/`, `images/`, and the token-bearing `env.{ps1,sh}` are
  git-ignored by design.
