#!/usr/bin/env bash
# Start the from-scratch detector ML backend. Run after sourcing env.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -x "$ROOT/.venv/bin/python" ]; then
  VPY="$ROOT/.venv/bin/python"
else
  VPY="$ROOT/.venv/Scripts/python.exe"
fi

PORT="${ML_PORT:-9090}"
LANIP="${LAN_IP:-}"
if [ -z "$LANIP" ]; then LANIP="$(hostname -I 2>/dev/null | awk '{print $1}')"; fi
if [ -z "$LANIP" ]; then LANIP="localhost"; fi

cd "$ROOT/ml_backend"
echo "Starting from-scratch detector ML backend — served on the local network"
echo "  this machine : http://localhost:$PORT"
echo "  on your LAN  : http://$LANIP:$PORT"
"$VPY" _wsgi.py
