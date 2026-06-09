#!/usr/bin/env bash
# Start Label Studio with local-file serving. Run after sourcing env.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -x "$ROOT/.venv/bin/label-studio" ]; then
  LS="$ROOT/.venv/bin/label-studio"
else
  LS="$ROOT/.venv/Scripts/label-studio.exe"
fi

export LOCAL_FILES_SERVING_ENABLED="${LOCAL_FILES_SERVING_ENABLED:-true}"
export LOCAL_FILES_DOCUMENT_ROOT="${LOCAL_FILES_DOCUMENT_ROOT:-$ROOT}"

echo "Starting Label Studio on http://localhost:8080 (doc root: $LOCAL_FILES_DOCUMENT_ROOT)"
"$LS" start --no-browser
