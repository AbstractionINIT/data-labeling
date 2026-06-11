"""
Drop-in importer for *pre-existing* annotations.

Paste images + their labels into the `extra_data/` folder (override with
$EXTRA_DATA_DIR) and they are merged into every training run automatically —
no need to re-draw them in Label Studio.

Auto-detected formats (mix freely; images can sit anywhere under the folder):

  1. Label Studio JSON export   — any *.json that is a list of task dicts with
     `annotations[].result` rectanglelabels. Images matched by filename.
  2. COCO                       — a *.json with {images, annotations, categories}.
  3. YOLO  (<stem>.txt)         — lines: `<class> <cx> <cy> <w> <h>` (normalized).
                                  `<class>` may be a NAME, or an index resolved via
                                  a names file (classes.txt / *.names / data.yaml).
  4. Pascal VOC (<stem>.xml)    — <object><name> + <bndbox> pixel coords.

Every box is converted to the same shape the trainer consumes (Label-Studio
rectanglelabels, percent coords). Boxes whose class isn't in the project's label
config are kept here but silently dropped by the trainer until you add that class
in Label Studio — so adding a class later retroactively picks them up.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
EXTRA_DIR = Path(os.getenv("EXTRA_DATA_DIR", str(PROJECT_DIR / "extra_data")))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _log(log, msg):
    (log.info if log else print)(msg)


def _box(name, x_pct, y_pct, w_pct, h_pct):
    """One LS-format rectanglelabels result (percent, top-left origin)."""
    clamp = lambda v: max(0.0, min(100.0, v))
    return {"type": "rectanglelabels",
            "value": {"x": clamp(x_pct), "y": clamp(y_pct),
                      "width": clamp(w_pct), "height": clamp(h_pct),
                      "rectanglelabels": [name]}}


def _index_images(root: Path):
    by_name, by_stem = {}, {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            by_name[p.name] = p
            by_stem[p.stem] = p
    return by_name, by_stem


def _load_names(root: Path):
    """Find a YOLO names file -> list of class names (index order)."""
    for cand in ["classes.txt", "obj.names", "names.txt"]:
        f = root / cand
        if f.exists():
            return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for f in list(root.rglob("*.names")) + list(root.rglob("data.yaml")) + list(root.rglob("*.yaml")):
        try:
            if f.suffix == ".names":
                return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            txt = f.read_text(encoding="utf-8")
            # crude data.yaml `names:` parse (list or {idx: name})
            if "names:" in txt:
                seg = txt.split("names:", 1)[1]
                names = []
                for ln in seg.splitlines():
                    ln = ln.strip()
                    if ln.startswith("- "):
                        names.append(ln[2:].strip().strip("'\""))
                    elif ln and ln[0].isdigit() and ":" in ln:
                        names.append(ln.split(":", 1)[1].strip().strip("'\""))
                    elif ln.startswith("[") and ln.endswith("]"):
                        names = [s.strip().strip("'\"") for s in ln[1:-1].split(",") if s.strip()]
                    elif names and (not ln or ln.endswith(":")):
                        break
                if names:
                    return names
        except Exception:
            pass
    return None


def _resolve(name_or_path: str, by_name, by_stem, root: Path):
    """Map an image reference (path / url / filename) to an actual file."""
    if not name_or_path:
        return None
    base = os.path.basename(name_or_path.replace("\\", "/").split("?")[0])
    base = base.split("d=")[-1] if "d=" in name_or_path else base
    from urllib.parse import unquote
    base = os.path.basename(unquote(base))
    if base in by_name:
        return by_name[base]
    if Path(base).stem in by_stem:
        return by_stem[Path(base).stem]
    p = (root / name_or_path)
    return p if p.exists() else None


# --------------------------------------------------------------------------- #
# Per-format parsers -> {image_path: [results]}
# --------------------------------------------------------------------------- #
def _from_ls_export(tasks, by_name, by_stem, root, out, log):
    n = 0
    for t in tasks:
        if not isinstance(t, dict):
            return False
        img = (t.get("data") or {}).get("image") or t.get("image")
        results = []
        for a in t.get("annotations") or t.get("predictions") or []:
            for r in a.get("result") or []:
                if r.get("type") == "rectanglelabels":
                    results.append(r)
        if not results:
            continue
        p = _resolve(img, by_name, by_stem, root)
        if p:
            out.setdefault(str(p), []).extend(results); n += 1
    if n:
        _log(log, f"  [extra] Label Studio export: {n} images")
    return n > 0


def _from_coco(doc, by_name, by_stem, root, out, log):
    cats = {c["id"]: c["name"] for c in doc.get("categories", [])}
    imgs = {im["id"]: im for im in doc.get("images", [])}
    n = 0
    for a in doc.get("annotations", []):
        im = imgs.get(a.get("image_id"))
        if not im or "bbox" not in a:
            continue
        p = _resolve(im.get("file_name", ""), by_name, by_stem, root)
        if not p:
            continue
        W, H = im.get("width"), im.get("height")
        if not W or not H:
            W, H = Image.open(p).size
        x, y, w, h = a["bbox"]  # pixel x,y,w,h (top-left)
        out.setdefault(str(p), []).append(
            _box(cats.get(a["category_id"], str(a["category_id"])),
                 x / W * 100, y / H * 100, w / W * 100, h / H * 100))
        n += 1
    if n:
        _log(log, f"  [extra] COCO: {n} boxes")
    return n > 0


def _from_yolo_txt(txt_path: Path, img_path: Path, names, out):
    boxes = []
    for ln in txt_path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = parts[0], *map(float, parts[1:5])
        if c.replace(".", "", 1).isdigit() and names:
            idx = int(float(c))
            name = names[idx] if 0 <= idx < len(names) else str(idx)
        elif c.replace(".", "", 1).isdigit():
            name = str(int(float(c)))     # numeric id, no names file
        else:
            name = c                       # already a class name
        boxes.append(_box(name, (cx - w / 2) * 100, (cy - h / 2) * 100, w * 100, h * 100))
    if boxes:
        out.setdefault(str(img_path), []).extend(boxes)
    return len(boxes)


def _from_voc_xml(xml_path: Path, img_path: Path, out):
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return 0
    size = root.find("size")
    W = float(size.findtext("width")) if size is not None and size.findtext("width") else None
    H = float(size.findtext("height")) if size is not None and size.findtext("height") else None
    if not W or not H:
        W, H = Image.open(img_path).size
    n = 0
    for obj in root.findall("object"):
        name = obj.findtext("name")
        b = obj.find("bndbox")
        if not name or b is None:
            continue
        x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
        x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
        out.setdefault(str(img_path), []).append(
            _box(name, x1 / W * 100, y1 / H * 100, (x2 - x1) / W * 100, (y2 - y1) / H * 100))
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_extra_samples(log=None) -> list[dict]:
    """Return [{image_path, results}] for everything found under EXTRA_DIR."""
    if not EXTRA_DIR.exists():
        return []
    by_name, by_stem = _index_images(EXTRA_DIR)
    if not by_name:
        return []
    names = _load_names(EXTRA_DIR)
    out: dict[str, list] = {}

    # JSON files first (LS export or COCO)
    for jf in EXTRA_DIR.rglob("*.json"):
        try:
            doc = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(doc, list):
            _from_ls_export(doc, by_name, by_stem, EXTRA_DIR, out, log)
        elif isinstance(doc, dict) and "annotations" in doc and "images" in doc:
            _from_coco(doc, by_name, by_stem, EXTRA_DIR, out, log)

    # Per-image sidecars (YOLO .txt / VOC .xml) matched by stem
    yolo_n = voc_n = 0
    for stem, img_path in by_stem.items():
        txt = img_path.with_suffix(".txt")
        if not txt.exists():
            txt = next((p for p in EXTRA_DIR.rglob(stem + ".txt")), None)
        if txt and txt.exists() and txt.name not in {"classes.txt", "names.txt"}:
            yolo_n += _from_yolo_txt(txt, img_path, names, out)
        xml = img_path.with_suffix(".xml")
        if not xml.exists():
            xml = next((p for p in EXTRA_DIR.rglob(stem + ".xml")), None)
        if xml and xml.exists():
            voc_n += _from_voc_xml(xml, img_path, out)
    if yolo_n:
        _log(log, f"  [extra] YOLO txt: {yolo_n} boxes")
    if voc_n:
        _log(log, f"  [extra] Pascal VOC: {voc_n} boxes")

    samples = [{"image_path": k, "results": v} for k, v in out.items() if v]
    if samples:
        found = sorted({r["value"]["rectanglelabels"][0]
                        for s in samples for r in s["results"]})
        total = sum(len(s["results"]) for s in samples)
        _log(log, f"  [extra] loaded {len(samples)} images / {total} boxes "
                  f"from {EXTRA_DIR} | classes: {found}")
    return samples
