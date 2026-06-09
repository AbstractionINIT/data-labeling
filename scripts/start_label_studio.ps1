# Start Label Studio with local-file serving enabled so the images/ folder
# can be served directly (no re-upload). Run in its OWN terminal.
#
#   . .\scripts\env.ps1
#   .\scripts\start_label_studio.ps1

$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\Activate.ps1"

# Allow LS to serve files from the project folder (parent of images\).
$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LOCAL_FILES_DOCUMENT_ROOT   = $root
# Skip the browser auto-open noise; we open it ourselves.
$env:LABEL_STUDIO_PORT = "8080"

Write-Host "Starting Label Studio on http://localhost:8080 (doc root: $root)"
label-studio start --no-browser
