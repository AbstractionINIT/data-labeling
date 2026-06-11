#!/usr/bin/env bash
# Launch ALL THREE servers (Label Studio 8090 + ML backend 9090 + dashboard 9091)
# in the background. Logs go to data/logs/<service>.out.
#
#   ./scripts/start_all.sh
#
# For a brand-new setup (account/token + image import + model wiring) you still
# do the one-time bootstrap; this script is for day-to-day startup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/logs"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

nohup bash "$ROOT/scripts/start_label_studio.sh" >"$ROOT/data/logs/ls.out" 2>&1 &
nohup bash "$ROOT/scripts/start_ml_backend.sh"   >"$ROOT/data/logs/ml.out" 2>&1 &
nohup "$ROOT/.venv/bin/python" "$ROOT/scripts/dashboard.py" >"$ROOT/data/logs/dash.out" 2>&1 &

echo "Launched in background:"
echo "  Label Studio -> http://localhost:8090   (log: data/logs/ls.out)"
echo "  ML backend   -> http://localhost:9090   (log: data/logs/ml.out)"
echo "  Dashboard    -> http://localhost:9091   (log: data/logs/dash.out)"
echo "Stop them with:  pkill -f 'serve_ls.py|_wsgi.py|dashboard.py'"
