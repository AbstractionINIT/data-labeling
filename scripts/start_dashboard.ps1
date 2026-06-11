# Start the training dashboard. Run after sourcing env.ps1.
#
#   . .\scripts\env.ps1
#   .\scripts\start_dashboard.ps1   ->  http://localhost:9091

$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\Activate.ps1"
if (-not $env:DASH_PORT) { $env:DASH_PORT = "9091" }
Write-Host "Dashboard -> http://localhost:$($env:DASH_PORT)"
& "$root\.venv\Scripts\python.exe" "$root\scripts\dashboard.py"
