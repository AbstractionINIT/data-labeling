"""
Backend-agnostic device selection so the SAME code runs on:

  * NVIDIA CUDA            -> torch.cuda (torch.cuda.is_available() == True)
  * AMD ROCm (Linux)       -> torch's ROCm build ALSO reports as 'cuda', so it is
                              picked up by the same check, transparently.
  * AMD / Intel on Windows -> DirectML (torch-directml), a separate device API.
  * CPU                    -> fallback.

Override with env FORCE_DEVICE = cuda | dml | cpu.
"""
from __future__ import annotations

import os
import torch


def get_device():
    forced = os.getenv("FORCE_DEVICE", "").lower()
    if forced == "cpu":
        return torch.device("cpu")
    if forced in ("cuda", "rocm"):
        return torch.device("cuda")
    if forced == "dml":
        import torch_directml
        return torch_directml.device()

    # Auto-detect. ROCm builds report True here too.
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass
    return torch.device("cpu")


def device_label(dev) -> str:
    dev = torch.device(dev) if not isinstance(dev, torch.device) else dev
    if dev.type == "cuda":
        try:
            # works for both NVIDIA CUDA and AMD ROCm builds
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    if dev.type == "privateuseone":   # DirectML tensors report this type
        return "directml (AMD/Intel GPU via DirectML)"
    return str(dev)
