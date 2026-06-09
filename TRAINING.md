# Training approach & model architecture (from scratch)

This explains the **custom detector built from raw PyTorch layers**, why it's
shaped the way it is, how it trains incrementally as you annotate, and how to run
the architecture-selection experiment to pick the best variant on your own data.

> Key constraint you set: an **empty / raw model** — random initialization, **no
> pretrained weights**, nothing borrowed. Everything below honors that.

---

## 1. The problem

- **Task:** object detection on wide, fish-eye **construction-site panoramas**
  (excavators, dump trucks, cranes, workers, concrete pipes, buildings, …).
- **Data regime:** small and **growing** — labels arrive in batches of 25, and
  **new classes may appear partway through**.
- **Hardware:** single 8 GB RTX 2070; retrain must finish fast enough to not
  block annotation.

---

## 2. The architecture — `ScratchDet`

A compact **single-stage, anchor-based grid detector** (YOLO-v2/v3 *family* by
design, but written from primitive layers in
[ml_backend/model_arch.py](ml_backend/model_arch.py)). Random init, trained from
scratch.

```
image [B,3,512,512]
  │  Backbone — Darknet-style CNN, 4× stride-2 downsample → stride 16
  ├─ stem      ConvBNAct(3→c0, k3, s2)                      /2
  ├─ stage1    ConvBNAct(c0→c1, s2) + N×ResidualBlock       /4
  ├─ stage2    ConvBNAct(c1→c2, s2) + N×ResidualBlock       /8
  ├─ stage3    ConvBNAct(c2→c3, s2) + N×ResidualBlock       /16
  │  Neck      2× ConvBNAct(c3→c3, k3)        (feature mixing)
  │  Head      Conv2d(c3 → A·(5+nc), k1)      (raw logits)
  ▼
prediction grid  [B, A=3, S=32, S=32, 5+nc]
```

**Atomic blocks (all from `nn.Conv2d`/`nn.BatchNorm2d`/`nn.SiLU`):**
- `ConvBNAct` = Conv(bias=False) → BatchNorm → SiLU
- `ResidualBlock` = 1×1 squeeze → 3×3 expand + skip (Darknet bottleneck)

**Output encoding.** The image is divided into a 32×32 cell grid (stride 16).
Each cell owns **3 anchor boxes**; each anchor predicts
`[tx, ty, tw, th, objectness, class₁…class_nc]`. Boxes decode as
`bx=(σ(tx)+cx)/S`, `by=(σ(ty)+cy)/S`, `bw=anchor_w·exp(tw)`, `bh=anchor_h·exp(th)`.

**Anchors.** 3 priors `(w,h)` normalized to the image, covering small/medium/large
objects (`DEFAULT_ANCHORS`). Refine them to your data with
`scripts/kmeans_anchors.py` once you have boxes.

**From-scratch init details that matter:**
- Kaiming-normal conv init; BN weight=1, bias=0.
- **Objectness bias initialized to −4.0** so early epochs aren't swamped by
  false positives (standard detector trick).

### Size variants (the "try multiple, keep the best" knob)
| variant | params | backbone channels |
|---|---|---|
| `tiny` | 0.51 M | 16 / 32 / 64 / 128 |
| `small` (default) | 1.36 M | 24 / 48 / 96 / 192 |
| `medium` | 2.77 M | 32 / 64 / 128 / 256 |

Set via `DET_VARIANT` in `env.ps1`.

### Why this design (and what was rejected)
- **Single-stage anchor grid** → simplest detector that still handles multiple
  objects per image and varied shapes, and is cheap to train repeatedly on 8 GB.
- **Two-stage (Faster R-CNN)** rejected: far more code/compute for marginal gain
  at this scale, and awkward to rebuild for changing classes.
- **Transformer detectors (DETR/RT-DETR)** rejected: extremely data-hungry —
  hopeless from scratch on tens of images.
- **Pretrained YOLO** rejected by your requirement (not an empty model).

---

## 3. The from-scratch training loss

Multipart loss in `detector.DetLoss`, computed on the prediction grid:

- **Objectness** — `BCEWithLogits` over *all* cells; positive cells (those that
  own a ground-truth box) weighted `λ_obj=1`, negatives `λ_noobj=0.5`.
- **Box** — only on positive cells: `xy` via BCE on the in-cell offset, `wh` via
  MSE on `log(gt/anchor)`. Weighted `λ_box=5`.
- **Classification** — `BCEWithLogits` per class on positive cells, `λ_cls=1`.

**Target assignment:** each GT box goes to the cell containing its center, and to
the anchor whose *shape* (w,h) has the highest IoU with the box.

Optimizer **AdamW** (lr `2e-3`, wd `5e-4`), **cosine** LR decay, grad-clip 10,
light augmentation (horizontal flip + brightness jitter). Wide panoramas are
**letterboxed** (aspect kept, gray padding) to the square input so distant
objects aren't squished.

---

## 4. How incremental "retrain every 25" works

On each annotation submit, Label Studio POSTs a webhook → backend `fit()`
([ml_backend/model.py](ml_backend/model.py)):

1. **Count** completed annotations via the LS export API (authoritative).
2. **Gate:** train only when the count *crosses a new multiple of 25*.
3. **Read classes live** from the labeling config (new classes auto-included).
4. **Convert** LS rectangles (percent, top-left) → normalized boxes, letterbox,
   build train/val split (`detector.DetDataset`).
5. **Train from random init** for an auto-scaled number of epochs (300 if <50
   images, 200 if <150, else 150) on the GPU.
6. **Promote** the new checkpoint to active and record metrics in
   `data/state.json`; subsequent `predict()` calls serve its boxes.

### Why re-init from scratch every cycle (not resume)
You asked for a pure-scratch model, and it's also the *correct* choice here:
re-initializing each cycle (a) cleanly supports a **changing class set**, (b)
trains on **all** labels collected so far, and (c) avoids accumulating overfit on
the tiny early data. It's cheap at this scale. (If the dataset grows large and
classes freeze, warm-starting to save time becomes a reasonable later change.)

### Honest expectation for a from-scratch detector
With no borrowed knowledge, the model needs **a lot** of your labels before
predictions are reliably useful — realistically **hundreds**, not 25. Early
cycles may predict little or noisy boxes. This is the inherent cost of "empty
model," and the per-25 loop is what climbs that curve. Track `best_val_loss` in
`status.py` / the log to watch it improve as you label more.

### Verified working
A controlled overfit test (6 images, 2 classes, 150 epochs) drove val loss
**45 → 2.8** and produced correct boxes at 0.99 confidence — confirming the
architecture, target encoding, loss, decode, and NMS are all correct.

---

## 5. Architecture-selection experiment

Within the from-scratch constraint, the thing to sweep is the **architecture
size**. `scripts/experiment_compare.py` trains `tiny`/`small`/`medium` from
random init on your current annotations and ranks them by validation loss:

```powershell
. .\scripts\env.ps1
.\.venv\Scripts\python.exe scripts\experiment_compare.py <project_id>
```

Pick the winner → set `DET_VARIANT` in `env.ps1`; future auto-retrains use it.
Re-run as the dataset grows (bigger data usually favors `medium`). Metric to
trust: validation loss (and, qualitatively, whether pre-annotations look right).

---

## 6. Tuning knobs (env.ps1)

| Var | Default | Effect |
|---|---|---|
| `DET_VARIANT` | `small` | architecture size: tiny/small/medium |
| `DET_IMG_SIZE` | `512` | network input; raise for tiny distant objects (more VRAM) |
| `DET_BATCH` | `8` | lower if you hit OOM |
| `DET_EPOCHS` | `0` (auto) | fix the epoch count if you want |
| `DET_LR` | `2e-3` | AdamW learning rate |
| `RETRAIN_EVERY` | `25` | retrain cadence |

Also tunable in code: `DEFAULT_ANCHORS`, `STRIDE`, loss weights (`DetLoss`).
