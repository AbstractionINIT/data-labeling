#!/usr/bin/env bash
# =============================================================================
# One-shot setup for the annotation + from-scratch detector project.
#
# Creates the .venv, installs the correct PyTorch build for your GPU, then the
# rest of the dependencies. Works on Linux and on Windows (Git Bash / MSYS).
#
# GPU auto-detection:
#   * NVIDIA            -> CUDA wheels        (download.pytorch.org/whl/cu124)
#   * AMD on Linux      -> ROCm wheels        (download.pytorch.org/whl/rocm6.2)
#   * AMD/Intel Windows -> DirectML           (torch-directml)   [best-effort]
#   * none              -> CPU wheels
#
# Override detection:   GPU_VENDOR=nvidia|amd|cpu ./setup.sh
# Override versions:    CUDA_VERSION=cu124 ROCM_VERSION=6.2 ./setup.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

c_green='\033[0;32m'; c_yellow='\033[1;33m'; c_red='\033[0;31m'; c_blue='\033[0;34m'; c_off='\033[0m'
info()  { echo -e "${c_blue}==>${c_off} $*"; }
ok()    { echo -e "${c_green}OK ${c_off} $*"; }
warn()  { echo -e "${c_yellow}!! ${c_off} $*"; }
err()   { echo -e "${c_red}ERR${c_off} $*" >&2; }

CUDA_VERSION="${CUDA_VERSION:-cu124}"
ROCM_VERSION="${ROCM_VERSION:-6.2}"

# ---------------------------------------------------------------------------
# 1. Detect OS
# ---------------------------------------------------------------------------
case "$(uname -s)" in
  Linux*)                 OS=linux ;;
  Darwin*)                OS=mac ;;
  MINGW*|MSYS*|CYGWIN*)   OS=windows ;;
  *)                      OS=unknown ;;
esac
info "Operating system: $OS"

# ---------------------------------------------------------------------------
# 2. Locate a Python interpreter
# ---------------------------------------------------------------------------
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -n "$PY" ] || { err "No python found on PATH. Install Python 3.10+."; exit 1; }
info "Using Python launcher: $PY ($($PY --version 2>&1))"

# ---------------------------------------------------------------------------
# 3. Create the virtual environment
# ---------------------------------------------------------------------------
if [ ! -d "$ROOT/.venv" ]; then
  info "Creating virtual environment at .venv"
  "$PY" -m venv "$ROOT/.venv"
else
  ok ".venv already exists; reusing it"
fi

# venv python path differs by OS
if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  VPY="$ROOT/.venv/Scripts/python.exe"      # Windows
else
  VPY="$ROOT/.venv/bin/python"              # Linux/mac
fi
ok "venv python: $VPY"

info "Upgrading pip"
"$VPY" -m pip install --upgrade pip >/dev/null
ok "pip $("$VPY" -m pip --version | awk '{print $2}')"

# ---------------------------------------------------------------------------
# 4. Detect GPU vendor (unless overridden)
# ---------------------------------------------------------------------------
detect_vendor() {
  # NVIDIA
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo nvidia; return
  fi
  # AMD on Linux
  if command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
    echo amd; return
  fi
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -iE 'vga|3d|display' | grep -iqE 'amd|radeon|advanced micro'; then
    echo amd; return
  fi
  # AMD/Intel on Windows: query the video controllers via PowerShell or wmic
  if [ "$OS" = "windows" ]; then
    local gpus=""
    if command -v powershell >/dev/null 2>&1; then
      gpus="$(powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name" 2>/dev/null || true)"
    fi
    [ -z "$gpus" ] && command -v wmic >/dev/null 2>&1 && gpus="$(wmic path win32_VideoController get name 2>/dev/null || true)"
    echo "$gpus" | grep -iqE 'nvidia' && { echo nvidia; return; }
    echo "$gpus" | grep -iqE 'amd|radeon' && { echo amd; return; }
  fi
  echo cpu
}

VENDOR="${GPU_VENDOR:-$(detect_vendor)}"
info "GPU vendor: $VENDOR  (override with GPU_VENDOR=nvidia|amd|cpu)"

# ---------------------------------------------------------------------------
# 5. Install the matching PyTorch build
# ---------------------------------------------------------------------------
install_torch() {
  case "$VENDOR" in
    nvidia)
      info "Installing CUDA PyTorch ($CUDA_VERSION)"
      "$VPY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
      ;;
    amd)
      if [ "$OS" = "linux" ]; then
        info "Installing ROCm PyTorch (rocm$ROCM_VERSION) for AMD on Linux"
        "$VPY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
      elif [ "$OS" = "windows" ]; then
        warn "AMD on Windows: PyTorch has no ROCm build here; using DirectML (best-effort)."
        warn "ROCm (the fast path) needs Linux. For best AMD performance, run this on Linux."
        "$VPY" -m pip install torch-directml
        # torch-directml pins a compatible torch; add a matching torchvision.
        "$VPY" -m pip install torchvision || warn "torchvision/torch version mismatch possible under DirectML; FORCE_DEVICE=cpu still works."
      else
        warn "AMD detected on $OS without a supported GPU build; installing CPU PyTorch."
        "$VPY" -m pip install torch torchvision
      fi
      ;;
    *)
      info "Installing CPU-only PyTorch"
      "$VPY" -m pip install torch torchvision
      ;;
  esac
}
install_torch

# ---------------------------------------------------------------------------
# 6. Install the rest of the dependencies
# ---------------------------------------------------------------------------
info "Installing backend requirements"
"$VPY" -m pip install -r "$ROOT/ml_backend/requirements.txt"
info "Installing Label Studio"
"$VPY" -m pip install label-studio

# ---------------------------------------------------------------------------
# 7. Verify the device the code will actually use
# ---------------------------------------------------------------------------
info "Verifying PyTorch device"
"$VPY" - <<'PYEOF'
import sys
sys.path.insert(0, "ml_backend")
import torch
from device_util import get_device, device_label
print(f"  torch {torch.__version__}")
print(f"  selected device: {device_label(get_device())}")
PYEOF

# ---------------------------------------------------------------------------
# 8. Next steps
# ---------------------------------------------------------------------------
echo
ok "Setup complete."
echo
# On a fresh clone the env file is gitignored; seed it from the template.
[ -f "$ROOT/scripts/env.ps1" ] || cp "$ROOT/scripts/env.example.ps1" "$ROOT/scripts/env.ps1" 2>/dev/null || true
[ -f "$ROOT/scripts/env.sh" ]  || cp "$ROOT/scripts/env.example.sh"  "$ROOT/scripts/env.sh"  2>/dev/null || true

echo "Next steps:"
if [ "$OS" = "windows" ]; then
  echo "  PowerShell (recommended on Windows):"
  echo "    1) edit scripts\\env.ps1  -> paste your Label Studio token"
  echo "    2) terminal #1:  . .\\scripts\\env.ps1 ;  .\\scripts\\start_label_studio.ps1"
  echo "    3) terminal #2:  . .\\scripts\\env.ps1 ;  .\\scripts\\start_ml_backend.ps1"
  echo "    4) terminal #3:  . .\\scripts\\env.ps1 ;  .\\.venv\\Scripts\\python.exe scripts\\bootstrap_project.py"
else
  echo "  1) edit scripts/env.sh   -> paste your Label Studio token"
  echo "  2) terminal #1:  source scripts/env.sh && bash scripts/start_label_studio.sh"
  echo "  3) terminal #2:  source scripts/env.sh && bash scripts/start_ml_backend.sh"
  echo "  4) terminal #3:  source scripts/env.sh && \"$VPY\" scripts/bootstrap_project.py"
fi
echo
echo "See SETUP.md (full guide) and TRAINING.md (model architecture)."
