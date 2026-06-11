# `extra_data/` — drop your pre-existing annotations here

Anything you put in this folder is **merged into every training run automatically**
(starting with the next retrain). Use it to fold in images you already labelled
elsewhere, without re-drawing them in Label Studio.

Just paste your images and their label files in heredid you (subfolders are fine — the
loader scans recursively). The format is **auto-detected**, so use whichever you
already have:

### 1. Label Studio JSON export  *(easiest if you used LS before)*
Drop the exported `*.json` (a list of tasks) **plus the images** anywhere in here.
Images are matched to tasks by filename.

### 2. YOLO  (one `.txt` next to each image, same name)
```
images/site01.jpg
images/site01.txt        # lines:  <class> <cx> <cy> <w> <h>   (all normalized 0–1)
```
`<class>` can be the **class name** directly, or a numeric index if you also drop a
names file (`classes.txt`, `*.names`, or `data.yaml` with a `names:` list).

### 3. COCO
A single `*.json` with `images` / `annotations` / `categories`, plus the images.

### 4. Pascal VOC  (one `.xml` next to each image)
Standard `<object><name>…</name><bndbox>…</bndbox></object>` with pixel coords.

---

### Notes
- **Class names must match your Label Studio labels.** A box whose class isn't in
  the current label config is kept but skipped by the trainer until you add that
  class in Label Studio — then it's picked up retroactively on the next retrain.
- If the same image filename exists both here and in Label Studio, the **Label
  Studio** annotation wins (the dropped-in copy is skipped).
- Check `data/logs/training.log` after a retrain — it logs how many extra images
  and boxes were merged and which classes were found.
- To use a different folder, set `EXTRA_DATA_DIR` before starting the ML backend.
- This folder's contents are git-ignored (only this README is tracked).
