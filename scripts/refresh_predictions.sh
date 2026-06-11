#!/usr/bin/env bash
# Push the latest trained model's predictions into Label Studio (auto-detects
# the newest version). Source env.sh first so the token/URL are set.
#
#   . ./scripts/env.sh
#   ./scripts/refresh_predictions.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VPY="$ROOT/.venv/bin/python"; [ -x "$VPY" ] || VPY="$ROOT/.venv/Scripts/python.exe"
exec "$VPY" "$ROOT/scripts/refresh_predictions.py"
