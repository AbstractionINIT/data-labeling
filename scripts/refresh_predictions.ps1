# Push the latest trained model's predictions into Label Studio (auto-detects
# the newest version). Source env.ps1 first so the token/URL are set.
#
#   . .\scripts\env.ps1
#   .\scripts\refresh_predictions.ps1
$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\python.exe" "$root\scripts\refresh_predictions.py"
