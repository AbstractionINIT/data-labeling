# Start THIS project's Label Studio instance. Run in its OWN terminal.
#
#   . .\scripts\env.ps1
#   .\scripts\start_label_studio.ps1
#
# It runs on port 8090 with:
#   * a SEPARATE database (data\.ls-data) so it never clashes with another LS on 8080
#   * LOCAL_FILES_DOCUMENT_ROOT = the project root (so images\ can be served)
#   * LABEL_STUDIO_HOST = http://localhost:8090  (must match the port, or CSS breaks)
#   * served by WAITRESS (multi-threaded), NOT Django's dev server, so concurrent
#     UI requests while annotating don't crash SQLite with
#     "Cannot operate on a closed database". Static files served via WhiteNoise.

$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\Activate.ps1"

$port = "8090"
# NOTE: Label Studio prefers the LABEL_STUDIO_-prefixed vars and they OVERRIDE
# the unprefixed ones. A leftover user/registry var
# (LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT) can otherwise force the wrong folder,
# so we set the prefixed forms explicitly here.
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = $root
$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LOCAL_FILES_DOCUMENT_ROOT   = $root
$env:LABEL_STUDIO_BASE_DATA_DIR  = "$root\data\.ls-data"
$env:LABEL_STUDIO_HOST           = "http://localhost:$port"
$env:LS_PORT    = $port
$env:LS_THREADS = "8"

Write-Host "Starting Label Studio (waitress) on http://localhost:$port"
Write-Host "  doc root : $root"
Write-Host "  database : $env:LABEL_STUDIO_BASE_DATA_DIR"
& "$root\.venv\Scripts\python.exe" "$root\scripts\serve_ls.py"
