# =============================================================================
# ZERO-BROWSER one-command setup.
#
#   . .\scripts\env.ps1   # (optional) to override creds/knobs first
#   .\scripts\auto_setup.ps1
#
# It will:
#   1. Create a Label Studio account + API token (no browser) and write the
#      token into scripts\env.ps1 automatically.
#   2. Launch Label Studio (port 8090, separate DB) and the ML backend (9090)
#      each in their own window.
#   3. Wait for both to be healthy.
#   4. Bootstrap the project: create it, import images, connect the ML backend.
#
# Credentials (override by setting these before running):
#   $env:LS_EMAIL     (default admin@local.dev)
#   $env:LS_PASSWORD  (default Annotate123!)
# =============================================================================
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& "$root\.venv\Scripts\Activate.ps1"

if (-not $env:LS_EMAIL)    { $env:LS_EMAIL    = "admin@local.dev" }
if (-not $env:LS_PASSWORD) { $env:LS_PASSWORD = "Annotate123!" }

$env:LABEL_STUDIO_BASE_DATA_DIR = "$root\data\.ls-data"

Write-Host "==> [1/4] Creating Label Studio account + API token (no browser)..." -ForegroundColor Cyan
& "$root\.venv\Scripts\python.exe" "$root\scripts\create_account.py"

Write-Host "==> [2/4] Launching Label Studio (8090), ML backend (9090), dashboard (9091)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit","-Command",". '$root\scripts\env.ps1'; & '$root\scripts\start_label_studio.ps1'"
Start-Process powershell -ArgumentList "-NoExit","-Command",". '$root\scripts\env.ps1'; & '$root\scripts\start_ml_backend.ps1'"
Start-Process powershell -ArgumentList "-NoExit","-Command",". '$root\scripts\env.ps1'; & '$root\scripts\start_dashboard.ps1'"

function Wait-Health($url, $name, $timeoutSec = 180) {
    Write-Host "    waiting for $name ($url) ..."
    for ($i = 0; $i -lt $timeoutSec; $i += 2) {
        try {
            if ((Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 3).StatusCode -eq 200) {
                Write-Host "    $name is up." -ForegroundColor Green; return
            }
        } catch {}
        Start-Sleep 2
    }
    throw "$name did not become healthy within $timeoutSec s. Check its window."
}

Write-Host "==> [3/4] Waiting for services..." -ForegroundColor Cyan
Wait-Health "http://localhost:8090/health" "Label Studio"
Wait-Health "http://localhost:9090/health" "ML backend"

Write-Host "==> [4/4] Bootstrapping project (create + import images + connect model)..." -ForegroundColor Cyan
. "$root\scripts\env.ps1"
& "$root\.venv\Scripts\python.exe" "$root\scripts\bootstrap_project.py"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " DONE - zero browser steps needed." -ForegroundColor Green
Write-Host "   Open:     http://localhost:8090" -ForegroundColor Green
Write-Host "   Login:    $($env:LS_EMAIL)  /  $($env:LS_PASSWORD)" -ForegroundColor Green
Write-Host "   Project:  Construction Site Detection (158 images imported)" -ForegroundColor Green
Write-Host "   Dashboard: http://localhost:9091" -ForegroundColor Green
Write-Host "   The model retrains automatically every 25 annotation events." -ForegroundColor Green
Write-Host "   Servers run in the two new windows; close them to stop." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
