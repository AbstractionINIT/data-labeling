"""
Training + inference for the from-scratch custom detector (ScratchDet).

Contains everything that is NOT the network definition:
  * Label Studio result  ->  normalized boxes
  * letterbox resize (keeps aspect of the wide panoramas) + box remap
  * Dataset / dataloader with light augmentation
  * YOLO-style target encoding onto the anchor grid
  * multipart loss (box + objectness + classification)
  * training loop (random init every cycle -> pure from-scratch)
  * decode + per-class NMS for predictions
  * persistent state (counter, active checkpoint, classes, metrics)

No pretrained weights are ever loaded. Each retrain starts from random init and
trains on ALL annotations collected so far.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms

import progress
from device_util import device_label, get_device
from model_arch import DEFAULT_ANCHORS, STRIDE, build_model
from trainlog import get_logger

log = get_logger()

# ---- paths ----------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CKPT_DIR = DATA_DIR / "checkpoints"
CACHE_DIR = DATA_DIR / "cache" / "letterbox"   # cached letterboxed images
STATE_PATH = DATA_DIR / "state.json"
HISTORY_PATH = DATA_DIR / "history.json"   # structured per-run training history
HISTORY_MAX_RUNS = 200                     # keep the most recent N runs

# ---- knobs (env-overridable) ---------------------------------------------- #
IMG_SIZE = int(os.getenv("DET_IMG_SIZE", "512"))
VARIANT = os.getenv("DET_VARIANT", "small")          # tiny | small | medium
BATCH = int(os.getenv("DET_BATCH", "8"))
BASE_EPOCHS = int(os.getenv("DET_EPOCHS", "0"))      # 0 = auto-scale by data size
LR = float(os.getenv("DET_LR", "2e-3"))

# ---- SAHI-style sliced inference (for very large / panoramic images) ------- #
# These construction panoramas are ~7571x2619; a single 512px pass downsamples
# ~15x and loses small objects. Slicing detects on overlapping tiles instead.
SLICED = os.getenv("DET_SLICED", "1").lower() not in ("0", "", "false", "no", "off")
SLICE_SIZE = int(os.getenv("DET_SLICE", "1024"))     # tile size in original px
SLICE_OVERLAP = float(os.getenv("DET_SLICE_OVERLAP", "0.2"))   # 0..1 tile overlap
CONF = float(os.getenv("DET_CONF", "0.25"))          # default confidence threshold
WORKERS = int(os.getenv("DET_WORKERS", "0"))         # DataLoader workers (0 = main thread)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "annotations_seen": 0,        # distinct images with annotations (info only)
        "events_since_train": 0,      # annotation submits/edits since last training
        "last_trained_at": 0,
        "train_runs": 0,
        "active_weights": None,
        "classes": [],
        "variant": VARIANT,
        "last_metrics": {},
    }


def save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------- #
# Runtime inference config (SAHI) — adjustable live from the dashboard.
# Kept in its OWN file so dashboard writes never race the trainer's state.json.
# --------------------------------------------------------------------------- #
INFERENCE_PATH = DATA_DIR / "inference.json"


def load_inference() -> dict:
    """Effective inference config: data/inference.json over env defaults.

    Keys: sliced/slice/overlap (SAHI), conf (default confidence threshold),
    class_conf ({class_name: threshold} overrides for individual classes),
    paused (when True, predict() generates nothing — dashboard-controlled).
    """
    cfg = {"sliced": SLICED, "slice": SLICE_SIZE, "overlap": SLICE_OVERLAP,
           "conf": CONF, "class_conf": {}, "paused": False}
    if INFERENCE_PATH.exists():
        try:
            saved = json.loads(INFERENCE_PATH.read_text(encoding="utf-8"))
            for k in cfg:
                if k in saved:
                    cfg[k] = saved[k]
        except Exception:
            pass
    return cfg


def save_inference(cfg: dict) -> dict:
    """Merge + persist inference config; returns the new effective config."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cur = load_inference()
    if "sliced" in cfg:
        cur["sliced"] = bool(cfg["sliced"])
    if "slice" in cfg:
        cur["slice"] = max(256, min(8192, int(cfg["slice"])))
    if "overlap" in cfg:
        cur["overlap"] = max(0.0, min(0.8, float(cfg["overlap"])))
    if "conf" in cfg:
        cur["conf"] = max(0.0, min(1.0, float(cfg["conf"])))
    if "class_conf" in cfg and isinstance(cfg["class_conf"], dict):
        cur["class_conf"] = {str(k): max(0.0, min(1.0, float(v)))
                             for k, v in cfg["class_conf"].items()}
    if "paused" in cfg:
        cur["paused"] = bool(cfg["paused"])
    tmp = INFERENCE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    tmp.replace(INFERENCE_PATH)
    return cur


# --------------------------------------------------------------------------- #
# Structured training history (one record per run, with full per-epoch curves)
# --------------------------------------------------------------------------- #
def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8")).get("runs", [])
        except Exception:
            return []
    return []


def append_history(record: dict) -> None:
    """Append one run record and atomically rewrite history.json (capped)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_history()
    runs.append(record)
    runs = runs[-HISTORY_MAX_RUNS:]
    tmp = HISTORY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    tmp.replace(HISTORY_PATH)


# --------------------------------------------------------------------------- #
# LS results -> normalized boxes
# --------------------------------------------------------------------------- #
def ls_results_to_boxes(results: list[dict], class_to_id: dict) -> list[tuple]:
    """Return list of (cls_id, xc, yc, w, h) normalized to [0,1]."""
    boxes = []
    for r in results:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        labels = v.get("rectanglelabels") or []
        if not labels or labels[0] not in class_to_id:
            continue
        xc = (v["x"] + v["width"] / 2) / 100.0
        yc = (v["y"] + v["height"] / 2) / 100.0
        w = v["width"] / 100.0
        h = v["height"] / 100.0
        boxes.append((class_to_id[labels[0]], xc, yc, w, h))
    return boxes


# --------------------------------------------------------------------------- #
# Letterbox (keep aspect) + box remap
# --------------------------------------------------------------------------- #
def letterbox(img: Image.Image, size: int):
    """Resize keeping aspect, pad to size x size. Returns (np_img, r, pad_x, pad_y)."""
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    return np.asarray(canvas), r, pad_x, pad_y, w, h


def remap_boxes(boxes, r, pad_x, pad_y, ow, oh, size):
    """Map normalized-to-original boxes into normalized-to-letterboxed boxes."""
    out = []
    for cls, xc, yc, bw, bh in boxes:
        nxc = (xc * ow * r + pad_x) / size
        nyc = (yc * oh * r + pad_y) / size
        nbw = bw * ow * r / size
        nbh = bh * oh * r / size
        out.append((cls, nxc, nyc, nbw, nbh))
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DetDataset(Dataset):
    def __init__(self, samples, class_to_id, size, train=True, cache=True):
        self.samples = samples
        self.class_to_id = class_to_id
        self.size = size
        self.train = train
        self.cache = cache
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def _letterboxed(self, path):
        """Letterbox once, then cache the 512x512 array on disk so later epochs
        skip the expensive open+resize of the full-res image."""
        if not self.cache:
            return letterbox(Image.open(path).convert("RGB"), self.size)
        try:
            st = os.stat(path)
            key = hashlib.md5(
                f"{path}|{int(st.st_mtime)}|{st.st_size}|{self.size}".encode()).hexdigest()
            f = CACHE_DIR / f"{key}.npz"
            if f.exists():
                d = np.load(f)
                return (d["img"], float(d["r"]), int(d["px"]),
                        int(d["py"]), int(d["ow"]), int(d["oh"]))
        except Exception:
            f = None
        np_img, r, px, py, ow, oh = letterbox(Image.open(path).convert("RGB"), self.size)
        if f is not None:
            try:
                np.savez(f, img=np_img, r=r, px=px, py=py, ow=ow, oh=oh)
            except Exception:
                pass
        return np_img, r, px, py, ow, oh

    def __getitem__(self, idx):
        s = self.samples[idx]
        boxes = ls_results_to_boxes(s["results"], self.class_to_id)

        np_img, r, px, py, ow, oh = self._letterboxed(s["image_path"])
        boxes = remap_boxes(boxes, r, px, py, ow, oh, self.size)

        # Light augmentation: horizontal flip + brightness jitter (train only).
        if self.train:
            if torch.rand(1).item() < 0.5:
                np_img = np_img[:, ::-1, :].copy()
                boxes = [(c, 1.0 - xc, yc, w, h) for (c, xc, yc, w, h) in boxes]
            if torch.rand(1).item() < 0.5:
                factor = 0.7 + 0.6 * torch.rand(1).item()
                np_img = np.clip(np_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        if boxes:
            tgt = torch.tensor(boxes, dtype=torch.float32)  # [N,5] cls,xc,yc,w,h
        else:
            tgt = torch.zeros((0, 5), dtype=torch.float32)
        return img_t, tgt


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


# --------------------------------------------------------------------------- #
# Target encoding + loss
# --------------------------------------------------------------------------- #
def _wh_iou(wh1, wh2):
    """IoU between box shapes (ignoring position). wh1:[N,2] wh2:[A,2] -> [N,A]."""
    w1, h1 = wh1[:, None, 0], wh1[:, None, 1]
    w2, h2 = wh2[None, :, 0], wh2[None, :, 1]
    inter = torch.min(w1, w2) * torch.min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-9)


class DetLoss(nn.Module):
    def __init__(self, anchors, num_classes, stride, img_size,
                 l_box=5.0, l_obj=1.0, l_noobj=0.5, l_cls=1.0):
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)  # normalized w,h
        self.nc = num_classes
        self.S = img_size // stride
        self.l_box, self.l_obj, self.l_noobj, self.l_cls = l_box, l_obj, l_noobj, l_cls
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_none = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, preds, targets):
        # preds: [B, A, S, S, 5+nc]
        device = preds.device
        B, A, S, _, no = preds.shape
        anchors = self.anchors.to(device)

        tobj = torch.zeros((B, A, S, S), device=device)
        obj_mask = torch.zeros((B, A, S, S), dtype=torch.bool, device=device)
        txy = torch.zeros((B, A, S, S, 2), device=device)
        twh = torch.zeros((B, A, S, S, 2), device=device)
        tcls = torch.zeros((B, A, S, S, self.nc), device=device)

        for b in range(B):
            t = targets[b].to(device)
            if t.numel() == 0:
                continue
            cls = t[:, 0].long()
            xc, yc, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
            gi = (xc * S).long().clamp(0, S - 1)
            gj = (yc * S).long().clamp(0, S - 1)
            ious = _wh_iou(t[:, 3:5], anchors)        # [N,A]
            best_a = ious.argmax(dim=1)               # [N]
            for n in range(t.shape[0]):
                a, i, j = best_a[n], gi[n], gj[n]
                obj_mask[b, a, j, i] = True
                tobj[b, a, j, i] = 1.0
                txy[b, a, j, i, 0] = xc[n] * S - i
                txy[b, a, j, i, 1] = yc[n] * S - j
                twh[b, a, j, i, 0] = torch.log(w[n] / anchors[a, 0] + 1e-9)
                twh[b, a, j, i, 1] = torch.log(h[n] / anchors[a, 1] + 1e-9)
                if 0 <= cls[n] < self.nc:
                    tcls[b, a, j, i, cls[n]] = 1.0

        # Objectness over all cells (positives weighted vs negatives)
        obj_loss_all = self.bce_none(preds[..., 4], tobj)
        pos = obj_mask
        neg = ~obj_mask
        obj_loss = (self.l_obj * obj_loss_all[pos].sum() +
                    self.l_noobj * obj_loss_all[neg].sum()) / max(B, 1)

        if pos.any():
            p = preds[pos]                            # [P, no]
            # xy via BCE (targets in [0,1)), wh via MSE on raw, cls via BCE
            xy_loss = self.bce_none(p[:, 0:2], txy[pos]).mean()
            wh_loss = self.mse(p[:, 2:4], twh[pos])
            cls_loss = self.bce(p[:, 5:], tcls[pos])
            box_loss = self.l_box * (xy_loss + wh_loss)
            cls_loss = self.l_cls * cls_loss
        else:
            box_loss = preds.sum() * 0.0
            cls_loss = preds.sum() * 0.0

        total = obj_loss + box_loss + cls_loss
        return total, {"obj": float(obj_loss), "box": float(box_loss), "cls": float(cls_loss)}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _split(samples):
    """Train on ALL samples (maximize data). The returned 'val' set is only a
    small subset used to monitor val_loss and pick the best checkpoint — those
    samples are still part of training, i.e. nothing is held out."""
    train = samples
    if len(samples) <= 8:
        return train, samples
    step = max(1, len(samples) // 60)        # ~60 images just for monitoring
    val = samples[::step][:60]
    return train, val


def train(samples, classes, state, variant=None):
    variant = variant or state.get("variant") or VARIANT
    device = get_device()
    class_to_id = {c: i for i, c in enumerate(classes)}
    nc = len(classes)
    run_no = state["train_runs"] + 1
    t_start = time.time()
    started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    train_s, val_s = _split(samples)

    # ---- log run header + per-image annotation counts ---------------------- #
    total_boxes = 0
    per_class = {c: 0 for c in classes}
    for s in samples:
        bxs = ls_results_to_boxes(s["results"], class_to_id)
        total_boxes += len(bxs)
        for cid, *_ in bxs:
            per_class[classes[cid]] += 1
    log.info("=" * 70)
    log.info(f"TRAIN RUN #{run_no} START  variant={variant}  device={device_label(device)}")
    log.info(f"  trained on {len(samples)} images "
             f"(train={len(train_s)}, val={len(val_s)}) | {total_boxes} total boxes")
    log.info(f"  classes ({nc}): {classes}")
    log.info(f"  boxes per class: {per_class}")
    for s in samples:
        name = Path(s['image_path']).name
        k = len(ls_results_to_boxes(s['results'], class_to_id))
        log.info(f"    image {name}: {k} annotations")
    _dl = {"num_workers": WORKERS}
    if WORKERS > 0:
        _dl.update(persistent_workers=True, prefetch_factor=2)
    tr = DataLoader(DetDataset(train_s, class_to_id, IMG_SIZE, train=True),
                    batch_size=BATCH, shuffle=True, collate_fn=collate, **_dl)
    va = DataLoader(DetDataset(val_s, class_to_id, IMG_SIZE, train=False),
                    batch_size=BATCH, shuffle=False, collate_fn=collate, **_dl)

    model = build_model(nc, variant=variant, anchors=DEFAULT_ANCHORS).to(device)
    crit = DetLoss(DEFAULT_ANCHORS, nc, STRIDE, IMG_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)

    n = len(samples)
    # Fewer epochs as the dataset grows (more data converges sooner).
    epochs = BASE_EPOCHS or (300 if n < 50 else 200 if n < 150 else
                             150 if n < 600 else 100)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log.info(f"  settings: img_size={IMG_SIZE} batch={BATCH} epochs={epochs} "
             f"lr={LR} anchors={len(DEFAULT_ANCHORS)}")

    best_val = math.inf
    best_ep = -1
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / f"scratchdet_{variant}_r{run_no:03d}.pt"

    # per-epoch curves for the structured history
    hist = {"epoch": [], "train_loss": [], "val_loss": [],
            "obj": [], "box": [], "cls": [], "lr": []}

    progress.set_train(run_no, 0, epochs, images=n, phase="starting")
    nbatches = len(tr)
    for ep in range(epochs):
        model.train()
        tloss, nb = 0.0, 0
        for bi, (imgs, targets) in enumerate(tr):
            imgs = imgs.to(device)
            preds = model(imgs)
            loss, parts = crit(preds, targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tloss += float(loss); nb += 1
            # heartbeat so the dashboard shows live progress even within a long
            # epoch (otherwise the progress bar goes "stale" between epochs).
            if bi % 5 == 0:
                frac = (ep + (bi + 1) / max(nbatches, 1)) / max(epochs, 1)
                el = time.time() - t_start
                eta = round(el / frac * (1 - frac)) if frac > 0.005 else None
                progress.set_train(run_no, ep, epochs, images=n, batch=bi + 1,
                                   batches=nbatches, elapsed_s=round(el), eta_s=eta)
        sched.step()
        tloss /= max(nb, 1)

        # validation loss
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for imgs, targets in va:
                imgs = imgs.to(device)
                l, _ = crit(model(imgs), targets)
                vloss += float(l)
        vloss /= max(len(va), 1)
        if vloss < best_val:
            best_val = vloss
            best_ep = ep
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "variant": variant,
                    "anchors": DEFAULT_ANCHORS,
                    "img_size": IMG_SIZE,
                },
                best_path,
            )
        log.info(f"  run#{run_no} ep {ep:3d}/{epochs} lr={sched.get_last_lr()[0]:.2e} "
                 f"train_loss={tloss:.4f} val_loss={vloss:.4f} "
                 f"obj={parts['obj']:.3f} box={parts['box']:.3f} cls={parts['cls']:.3f}")
        hist["epoch"].append(ep)
        hist["train_loss"].append(round(tloss, 5))
        hist["val_loss"].append(round(vloss, 5))
        hist["obj"].append(round(float(parts["obj"]), 5))
        hist["box"].append(round(float(parts["box"]), 5))
        hist["cls"].append(round(float(parts["cls"]), 5))
        hist["lr"].append(float(sched.get_last_lr()[0]))
        _frac = (ep + 1) / max(epochs, 1)
        _el = time.time() - t_start
        progress.set_train(run_no, ep + 1, epochs, images=n,
                           train_loss=round(tloss, 4), val_loss=round(vloss, 4),
                           best_val=round(best_val, 4), elapsed_s=round(_el),
                           eta_s=round(_el / _frac * (1 - _frac)) if _frac > 0.005 else None)

    progress.clear_train()
    dur = time.time() - t_start
    log.info(f"TRAIN RUN #{run_no} DONE  best_val_loss={best_val:.4f} @ep{best_ep} "
             f"| {len(samples)} images | {dur:.1f}s | ckpt={best_path.name}")
    log.info("=" * 70)

    state["train_runs"] += 1
    state["last_trained_at"] = state["annotations_seen"]
    state["active_weights"] = str(best_path)
    state["classes"] = list(classes)
    state["variant"] = variant
    state["last_metrics"] = {"best_val_loss": best_val, "images": n,
                             "epochs": epochs, "duration_s": round(dur, 1)}
    save_state(state)

    # ---- persist the full structured run record ---------------------------- #
    append_history({
        "run": run_no,
        "started_at": started_iso,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variant": variant,
        "device": device_label(device),
        "images": n,
        "train_images": len(train_s),
        "val_images": len(val_s),
        "total_boxes": total_boxes,
        "per_class": per_class,
        "classes": list(classes),
        "img_size": IMG_SIZE,
        "batch": BATCH,
        "epochs": epochs,
        "lr": LR,
        "anchors": len(DEFAULT_ANCHORS),
        "best_val_loss": round(best_val, 5),
        "best_epoch": best_ep,
        "duration_s": round(dur, 1),
        "checkpoint": best_path.name,
        "curve": hist,
    })
    return state


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
_LOADED = {"path": None, "model": None, "meta": None}


def load_detector(weights_path: str):
    if _LOADED["path"] == weights_path and _LOADED["model"] is not None:
        return _LOADED["model"], _LOADED["meta"]
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    device = get_device()
    model = build_model(len(ckpt["classes"]), variant=ckpt["variant"], anchors=ckpt["anchors"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    _LOADED.update(path=weights_path, model=model, meta=ckpt)
    return model, ckpt


@torch.no_grad()
def predict_boxes(weights_path: str, img: Image.Image, conf=0.25, iou=0.45):
    """Return list of (cls_id, score, x1,y1,x2,y2) in ORIGINAL image pixels."""
    model, meta = load_detector(weights_path)
    device = next(model.parameters()).device
    size = meta["img_size"]
    anchors = torch.tensor(meta["anchors"], device=device)

    np_img, r, px, py, ow, oh = letterbox(img, size)
    x = torch.from_numpy(np_img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    preds = model(x)[0]                                # [A,S,S,no]
    A, S, _, no = preds.shape

    # build grid
    gy, gx = torch.meshgrid(torch.arange(S, device=device),
                            torch.arange(S, device=device), indexing="ij")
    boxes, scores, classes = [], [], []
    for a in range(A):
        p = preds[a]                                   # [S,S,no]
        bx = (torch.sigmoid(p[..., 0]) + gx) / S
        by = (torch.sigmoid(p[..., 1]) + gy) / S
        bw = torch.exp(p[..., 2]).clamp(max=10) * anchors[a, 0]
        bh = torch.exp(p[..., 3]).clamp(max=10) * anchors[a, 1]
        obj = torch.sigmoid(p[..., 4])
        cls_prob = torch.sigmoid(p[..., 5:])
        cls_score, cls_id = cls_prob.max(dim=-1)
        score = obj * cls_score
        m = score > conf
        if m.sum() == 0:
            continue
        # to letterboxed pixel xyxy
        cx, cy = bx[m] * size, by[m] * size
        w_, h_ = bw[m] * size, bh[m] * size
        x1 = cx - w_ / 2; y1 = cy - h_ / 2; x2 = cx + w_ / 2; y2 = cy + h_ / 2
        boxes.append(torch.stack([x1, y1, x2, y2], dim=1))
        scores.append(score[m])
        classes.append(cls_id[m])

    if not boxes:
        return []
    # Move to CPU for NMS + extraction so this works on any backend
    # (torchvision.ops.nms isn't implemented for the DirectML device).
    boxes = torch.cat(boxes).cpu()
    scores = torch.cat(scores).cpu()
    classes = torch.cat(classes).cpu()

    # per-class NMS, then map letterboxed pixels back to original image pixels
    out = []
    for c in classes.unique():
        sel = classes == c
        keep = nms(boxes[sel], scores[sel], iou)
        for k in keep:
            x1, y1, x2, y2 = boxes[sel][k].tolist()
            ox1 = (x1 - px) / r; oy1 = (y1 - py) / r
            ox2 = (x2 - px) / r; oy2 = (y2 - py) / r
            out.append((int(c), float(scores[sel][k]),
                        max(0, ox1), max(0, oy1), min(ow, ox2), min(oh, oy2)))
    return out


def _merge_dets(dets, iou):
    """Global per-class NMS over detections already in original-image pixels."""
    if not dets:
        return []
    boxes = torch.tensor([[d[2], d[3], d[4], d[5]] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d[1] for d in dets], dtype=torch.float32)
    classes = torch.tensor([d[0] for d in dets])
    out = []
    for c in classes.unique():
        sel = classes == c
        bsel, ssel = boxes[sel], scores[sel]
        for k in nms(bsel, ssel, iou):
            x1, y1, x2, y2 = bsel[k].tolist()
            out.append((int(c), float(ssel[k]), x1, y1, x2, y2))
    return out


@torch.no_grad()
def predict_boxes_sliced(weights_path: str, img: Image.Image, conf=0.25, iou=0.45,
                         slice_size=None, overlap=None, include_full=True):
    """SAHI-style sliced inference for large images.

    Tile the image into overlapping windows, run the detector on each (where a
    small object spans enough pixels to survive the 512px letterbox), then merge
    all tiles' detections with one global per-class NMS. A final full-image pass
    catches objects bigger than a single tile.
    """
    W, H = img.size
    sz = int(slice_size or SLICE_SIZE)
    ov = SLICE_OVERLAP if overlap is None else overlap
    if W <= sz and H <= sz:                      # small image -> no point slicing
        return predict_boxes(weights_path, img, conf=conf, iou=iou)

    step = max(1, int(sz * (1 - ov)))
    xs = list(range(0, max(1, W - sz + 1), step)) or [0]
    ys = list(range(0, max(1, H - sz + 1), step)) or [0]
    if xs[-1] + sz < W:
        xs.append(max(0, W - sz))                # make the last tile hit the edge
    if ys[-1] + sz < H:
        ys.append(max(0, H - sz))

    dets = []
    for y0 in ys:
        for x0 in xs:
            crop = img.crop((x0, y0, min(x0 + sz, W), min(y0 + sz, H)))
            for cl, sc, a, b, c2, d in predict_boxes(weights_path, crop, conf=conf, iou=iou):
                dets.append((cl, sc, a + x0, b + y0, c2 + x0, d + y0))
    if include_full:
        dets.extend(predict_boxes(weights_path, img, conf=conf, iou=iou))
    return _merge_dets(dets, iou)
