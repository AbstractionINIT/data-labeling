# Shared environment for all services. Dot-source this in every terminal:
#     . .\scripts\env.ps1
#
# !!! EDIT THIS: paste your Label Studio access token below.
# Get it from Label Studio  ->  Account & Settings  ->  Access Token  (or Legacy Token).

$env:LABEL_STUDIO_URL     = "http://localhost:8080"
$env:LABEL_STUDIO_API_KEY = "PASTE_YOUR_TOKEN_HERE"
$env:ML_BACKEND_URL       = "http://localhost:9090"

# Retrain cadence and custom-model knobs (from-scratch detector)
$env:RETRAIN_EVERY = "25"
$env:DET_VARIANT  = "small"   # tiny | small | medium  (architecture size)
$env:DET_IMG_SIZE = "512"     # network input (square, letterboxed)
$env:DET_BATCH    = "8"       # fits 8 GB VRAM at 512
$env:DET_EPOCHS   = "0"       # 0 = auto-scale epochs by dataset size
$env:DET_LR       = "2e-3"

# Let the ML backend resolve local-storage images that LS serves.
$env:LABEL_STUDIO_HOST = $env:LABEL_STUDIO_URL

Write-Host "Env loaded. LS=$($env:LABEL_STUDIO_URL)  ML=$($env:ML_BACKEND_URL)  retrain every $($env:RETRAIN_EVERY)."
if ($env:LABEL_STUDIO_API_KEY -eq "PASTE_YOUR_TOKEN_HERE") {
    Write-Warning "Set LABEL_STUDIO_API_KEY in scripts\env.ps1 before running the bootstrap."
}
