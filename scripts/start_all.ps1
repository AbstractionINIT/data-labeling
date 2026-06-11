# Launch ALL THREE servers, each in its own window:
#   Label Studio (8090) + ML backend (9090) + dashboard (9091)
#
#   .\scripts\start_all.ps1
#
# This is for day-to-day startup once you're already set up. For a brand-new
# setup (creates the account/token, imports images, connects the model), run
# .\scripts\auto_setup.ps1 instead.

$root = Split-Path -Parent $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit","-Command",". '$root\scripts\env.ps1'; & '$root\scripts\start_label_studio.ps1'"
Start-Process powershell -ArgumentList "-NoExit","-Command",". '$root\scripts\env.ps1'; & '$root\scripts\start_ml_backend.ps1'"
Start-Process powershell -ArgumentList "-NoExit","-Command",". '$root\scripts\env.ps1'; & '$root\scripts\start_dashboard.ps1'"

Write-Host "Launched in separate windows:" -ForegroundColor Green
Write-Host "  Label Studio -> http://localhost:8090"
Write-Host "  ML backend   -> http://localhost:9090"
Write-Host "  Dashboard    -> http://localhost:9091"
Write-Host "Close those windows to stop the servers."
