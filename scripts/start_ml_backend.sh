#!/usr/bin/env bash
# Start the from-scratch detector ML backend. Run after sourcing env.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -x "$ROOT/.venv/bin/python" ]; then
  VPY="$ROOT/.venv/bin/python"
else
  VPY="$ROOT/.venv/Scripts/python.exe"
fi

cd "$ROOT/ml_backend"
echo "Starting detector ML backend on ${ML_BACKEND_URL:-http://localhost:9090}"
"$VPY" _wsgi.py
