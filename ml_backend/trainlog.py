"""
Shared training logger.

Writes a single human-readable log of everything that matters:
  * each annotation event (which task/image, how many boxes on it, running total)
  * when a training run starts, on how much data, with what settings
  * per-epoch losses and the final summary (best val loss, checkpoint, duration)

Rotation: the active file is capped at 2 GB. When it fills, it is rolled over to
training.log.1 (then .2, ...) and a fresh training.log is started automatically,
so logging never stops and no single file exceeds the cap.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
LOG_PATH = LOG_DIR / "training.log"
MAX_BYTES = 2 * 1024 ** 3   # 2 GB per file
BACKUP_COUNT = 100          # keep up to 100 rolled files before the oldest is dropped


def get_logger() -> logging.Logger:
    logger = logging.getLogger("annotation_trainer")
    if logger.handlers:           # already configured
        return logger
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(
        LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()   # also echo to the backend console
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger
