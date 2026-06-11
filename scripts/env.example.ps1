# Shared environment for all services. Dot-source this in every terminal:
#     . .\scripts\env.ps1
#
# !!! EDIT THIS: paste your Label Studio access token below.
# Get it from Label Studio  ->  Account & Settings  ->  Access Token  (or Legacy Token).

# This project runs its OWN Label Studio instance on port 8081 with a separate
# database (data\.ls-data), so it never collides with any other LS you run on 8080.
$env:LABEL_STUDIO_URL     = "http://localhost:8090"
$env:LABEL_STUDIO_API_KEY = "PASTE_YOUR_TOKEN_HERE"
$env:ML_BACKEND_URL       = "http://localhost:9090"

# Retrain cadence and custom-model knobs (from-scratch detector)
$env:RETRAIN_EVERY = "25"
$env:DET_VARIANT  = "small"   # tiny | small | medium  (architecture size)
$env:DET_IMG_SIZE = "512"     # network input (square, letterboxed)
$env:DET_BATCH    = "8"       # fits 8 GB VRAM at 512
$env:DET_EPOCHS   = "0"       # 0 = auto-scale epochs by dataset size
$env:DET_LR       = "2e-3"

# NOTE: do NOT set LABEL_STUDIO_HOST here. It must equal the server's own
# host:port or the UI loads CSS/JS from the wrong port (blank/unstyled page).
# The start script sets it correctly for the 8081 instance.

Write-Host "Env loaded. LS=$($env:LABEL_STUDIO_URL)  ML=$($env:ML_BACKEND_URL)  retrain every $($env:RETRAIN_EVERY)."
if ($env:LABEL_STUDIO_API_KEY -eq "PASTE_YOUR_TOKEN_HERE") {
    Write-Warning "Set LABEL_STUDIO_API_KEY in scripts\env.ps1 before running the bootstrap."
}
