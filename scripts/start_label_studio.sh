#!/usr/bin/env bash
# Start THIS project's Label Studio instance (port 8090, separate database).
# Run after sourcing env.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -x "$ROOT/.venv/bin/label-studio" ]; then
  LS="$ROOT/.venv/bin/label-studio"
else
  LS="$ROOT/.venv/Scripts/label-studio.exe"
fi

PORT=8090
# Detect the primary LAN IPv4 so LS is reachable from other devices (waitress
# binds 0.0.0.0; HOST must be the LAN IP or links/CSS/CSRF break off-machine).
LANIP="${LAN_IP:-}"
if [ -z "$LANIP" ]; then LANIP="$(hostname -I 2>/dev/null | awk '{print $1}')"; fi
if [ -z "$LANIP" ]; then LANIP="$(ipconfig getifaddr en0 2>/dev/null || true)"; fi
if [ -z "$LANIP" ]; then LANIP="localhost"; fi

# LABEL_STUDIO_-prefixed vars override the unprefixed ones and any leftover
# registry/user var (e.g. LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT) — set both.
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$ROOT"
export LOCAL_FILES_SERVING_ENABLED="true"
export LOCAL_FILES_DOCUMENT_ROOT="$ROOT"
export LABEL_STUDIO_BASE_DATA_DIR="$ROOT/data/.ls-data"   # separate DB
export LABEL_STUDIO_HOST="http://$LANIP:$PORT"            # LAN IP so off-machine access works
export LS_PORT="$PORT"
export LS_THREADS="8"

# Served by WAITRESS (multi-threaded) instead of Django's dev server, so the
# concurrent UI requests during annotation don't crash SQLite with
# "Cannot operate on a closed database". Static files via WhiteNoise.
VPY="$ROOT/.venv/bin/python"; [ -x "$VPY" ] || VPY="$ROOT/.venv/Scripts/python.exe"
echo "Starting Label Studio (waitress) — served on the local network"
echo "  this machine : http://localhost:$PORT"
echo "  on your LAN  : http://$LANIP:$PORT   (open this from other devices)"
echo "  doc root     : $ROOT"
echo "  database     : $LABEL_STUDIO_BASE_DATA_DIR"
"$VPY" "$ROOT/scripts/serve_ls.py"
