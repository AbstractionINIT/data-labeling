# Setup — Label Studio + from-scratch auto-training detector

End-to-end setup for annotating the construction-site images in `images/` and
training a **custom object detector, from scratch (random weights, no pretrained
model)**, that **retrains automatically every 25 completed annotations**.
Everything lives in a local Python virtual environment.

- **OS used:** Windows 11, PowerShell
- **Python:** 3.12.10
- **GPU:** NVIDIA RTX 2070 Super (8 GB), driver 591.86 — CUDA verified working
- **Annotation type:** object detection (bounding boxes)
- **Model:** custom `ScratchDet` (raw PyTorch, random init) — see [TRAINING.md](TRAINING.md)

---

## Quick start (automated)

One script does steps 1–4 below (venv + the right PyTorch for your GPU + all deps).
It auto-detects **NVIDIA (CUDA)**, **AMD on Linux (ROCm)**, **AMD/Intel on Windows
(DirectML)**, or falls back to **CPU**:

```bash
bash setup.sh
# force a path if needed:   GPU_VENDOR=amd bash setup.sh
```

Then run the **zero-browser one-command setup** (creates the account + API token,
starts Label Studio on **http://localhost:8090** and the ML backend on :9090, and
bootstraps the project — no clicking required):

```powershell
.\scripts\auto_setup.ps1
```

When it finishes, open <http://localhost:8090> (login `admin@local.dev` /
`Annotate123!`, override via `$env:LS_EMAIL`/`$env:LS_PASSWORD`), open
**Construction Site Detection**, and start labeling. The ML backend is already
connected and retrains every 25 submissions.

> This project runs on **port 8090** with its **own database** (`data\.ls-data`)
> so it never collides with another Label Studio you run on 8080.
>
> Heads-up if you've used Label Studio before: a leftover user/registry variable
> `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` can override the images folder. The
> start script sets the correct value for its own process, so you don't need to
> touch the registry.

If you'd rather do it manually instead of `auto_setup.ps1`, follow **steps 5–8**
below (paste your Label Studio token, start the two servers, bootstrap).

> AMD note: the fast AMD path (ROCm) is **Linux-only** — PyTorch has no ROCm build
> for Windows. On Windows, AMD GPUs use DirectML (best-effort); for serious AMD
> training, run on Linux. The code itself is device-agnostic (`ml_backend/device_util.py`):
> ROCm shows up as `cuda` automatically, and you can override with `FORCE_DEVICE=cuda|dml|cpu`.

Everything below is the manual / per-OS detail if you'd rather not use the script.

---

## 0. Project layout

```
d:\annotation\
├─ .venv\                      # the virtual environment (all deps here)
├─ images\                     # 158 source .jpeg images (already present)
├─ setup.sh                    # one-shot installer (GPU auto-detect: NVIDIA/AMD/CPU)
├─ ml_backend\
│  ├─ model_arch.py            # the custom network (from raw PyTorch layers)
│  ├─ detector.py              # dataset, target encoding, loss, train loop, NMS
│  ├─ device_util.py           # CUDA / ROCm / DirectML / CPU device selection
│  ├─ model.py                 # Label Studio ML backend: predict() + fit()
│  ├─ trainlog.py              # 2 GB-rotating training logger
│  ├─ _wsgi.py                 # backend server entrypoint (port 9090)
│  └─ requirements.txt
├─ scripts\
│  ├─ env.ps1 / env.sh         # env vars (EDIT: paste your LS token)
│  ├─ start_label_studio.ps1 / .sh   # start LS on :8080 with local-file serving
│  ├─ start_ml_backend.ps1 / .sh     # start the detector backend on :9090
│  ├─ bootstrap_project.py     # create project + import images + connect model
│  ├─ labeling_config.xml      # the label classes (edit to add classes)
│  ├─ status.py                # show counter / active model / metrics
│  ├─ force_train.py           # trigger a retrain on demand
│  ├─ experiment_compare.py    # train tiny/small/medium, keep the best
│  └─ kmeans_anchors.py        # (optional) refine anchor priors to your boxes
├─ data\                       # created at runtime:
│  ├─ checkpoints\             #   trained model weights (.pt)
│  ├─ logs\training.log        #   rotating training log (<= 2 GB each)
│  └─ state.json              #   counter, active model, classes, metrics
├─ SETUP.md                    # this file
└─ TRAINING.md                 # model architecture & training approach
```

---

## 1. Create the virtual environment

```powershell
cd d:\annotation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

## 2. Install PyTorch with CUDA (GPU build) — do this FIRST

The default `pip install torch` on Windows is **CPU-only**. Install the CUDA
build so the RTX 2070 is used:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> cu124 wheels work with driver 591.86 (the driver only needs to be newer than
> the bundled CUDA runtime). No system CUDA toolkit / `nvcc` required.

## 3. Install the rest (Label Studio + ML backend deps)

There is **no Ultralytics and no pretrained model** here — the detector is our
own code. The backend deps:

```powershell
.\.venv\Scripts\python.exe -m pip install -r ml_backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install label-studio
```

> Note: `label-studio-ml` resolves to **1.0.9** (its latest on PyPI); the backend
> code targets that API.

## 4. Verify the GPU is visible to PyTorch

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# verified output:  2.6.0+cu124  True  NVIDIA GeForce RTX 2070 SUPER
```

---

## 5. Configure environment variables

Edit `scripts\env.ps1` and paste your Label Studio access token into
`LABEL_STUDIO_API_KEY` (get it once LS is running: **Account & Settings →
Access Token**; use a *Legacy Token* if offered). This file also holds the model
knobs (`DET_VARIANT`, `DET_IMG_SIZE`, `DET_BATCH`, `DET_EPOCHS`, `RETRAIN_EVERY`).

## 6. Start Label Studio (terminal #1)

```powershell
cd d:\annotation
. .\scripts\env.ps1
.\scripts\start_label_studio.ps1
```

Runs `label-studio start --no-browser` with `LOCAL_FILES_SERVING_ENABLED=true`
and `LOCAL_FILES_DOCUMENT_ROOT=d:\annotation`, so `images\` is served without
re-upload. First run: create your account at <http://localhost:8080>, then paste
your token into `scripts\env.ps1`.

## 7. Start the detector ML backend (terminal #2)

```powershell
cd d:\annotation
. .\scripts\env.ps1
.\scripts\start_ml_backend.ps1
```

Serves on <http://localhost:9090> (`/health`, `/predict`, `/webhook`). Verified:
`/health` returns `{"status":"UP"}`.

## 8. Bootstrap the project (terminal #3, one time)

```powershell
cd d:\annotation
. .\scripts\env.ps1
.\.venv\Scripts\python.exe scripts\bootstrap_project.py
```

Creates the project, registers `images\` as Local Storage and syncs it (imports
the 158 images), connects the ML backend, and enables **“start training on
annotation submit.”**

---

## 9. Annotate → auto-train loop

1. Open <http://localhost:8080> → **Construction Site Detection** → **Label All
   Tasks**.
2. Draw boxes; **Submit** each image.
3. On each submit, Label Studio POSTs to the backend `/webhook`. The backend
   counts completed images and **retrains the model from scratch when the count
   crosses a multiple of 25** (25, 50, 75, …). Training runs on the GPU.
4. After the first training, the backend serves **pre-annotations** (click
   **Retrieve Predictions** if they don't auto-load). Expect these to be rough
   at first — a from-scratch model needs many labels before it's useful
   (see TRAINING.md).

### Adding a new class mid-annotation
Edit classes in **Project → Settings → Labeling Interface** (or
`scripts\labeling_config.xml`) and keep going. The backend reads the current
class list on every retrain and rebuilds the model head, so the new class is
included on the next 25-image cycle. No restart needed.

### Watching progress / logs
```powershell
.\.venv\Scripts\python.exe scripts\status.py            # counter + last metrics
Get-Content data\logs\training.log -Tail 40 -Wait       # live training log
.\.venv\Scripts\python.exe scripts\force_train.py <id>  # retrain now (id from URL)
```
The log records each annotation event (which image, how many boxes), when each
run starts, how much data it trains on, per-epoch losses, and a summary. Each log
file is capped at 2 GB and rolls over to `training.log.1`, `.2`, … automatically.

### Pick the best architecture (optional)
Once you have ~50+ labels:
```powershell
.\.venv\Scripts\python.exe scripts\experiment_compare.py <project_id>
```
Trains `tiny`/`small`/`medium` from scratch on your data and ranks them; set the
winner as `DET_VARIANT` in `env.ps1`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | You installed the CPU wheel. Re-run step 2 with the `--index-url` flag inside the venv. |
| Bootstrap: "Failed to create local storage" | LS wasn't started with `LOCAL_FILES_SERVING_ENABLED=true` and a `LOCAL_FILES_DOCUMENT_ROOT` that is a parent of `images\`. Use `start_label_studio.ps1`. |
| No pre-annotations appear | Train at least once (annotate 25), then **Retrieve Predictions**; check the backend terminal/log for errors. Early from-scratch models may legitimately predict nothing yet. |
| Training never fires | **Project → Settings → Model** must have the backend connected with training-on-submit enabled; confirm the log shows `EVENT ANNOTATION_CREATED ...`. Or run `force_train.py`. |
| Out of VRAM | Lower `DET_IMG_SIZE` to 416 or `DET_BATCH` to 4/2, or set `DET_VARIANT=tiny` in `env.ps1`. |
| Can't delete files in `data\` | A backend process still holds them — stop the backend (Ctrl-C in terminal #2) first. |
