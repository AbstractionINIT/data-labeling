# Start the YOLO ML backend. Run in its OWN terminal AFTER env.ps1.
#
#   . .\scripts\env.ps1
#   .\scripts\start_ml_backend.ps1

$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\Activate.ps1"

Set-Location "$root\ml_backend"
Write-Host "Starting YOLO ML backend on $($env:ML_BACKEND_URL)"
python _wsgi.py
