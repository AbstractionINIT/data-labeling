# Shared environment for the bash/Linux workflow. Source it in every terminal:
#     source scripts/env.sh
#
# !!! EDIT: paste your Label Studio access token below.
# Get it from Label Studio -> Account & Settings -> Access Token.

# This project runs its OWN Label Studio instance on port 8081 with a separate
# database (data/.ls-data), so it never collides with another LS on 8080.
export LABEL_STUDIO_URL="http://localhost:8090"
export LABEL_STUDIO_API_KEY="PASTE_YOUR_TOKEN_HERE"
export ML_BACKEND_URL="http://localhost:9090"

# Retrain cadence and custom-model knobs (from-scratch detector)
export RETRAIN_EVERY="25"
export DET_VARIANT="small"     # tiny | small | medium
export DET_IMG_SIZE="512"
export DET_BATCH="8"
export DET_EPOCHS="0"          # 0 = auto-scale by dataset size
export DET_LR="2e-3"
export DET_WORKERS="0"         # DataLoader workers (0 = main thread; raise to 4
                              # to speed up loading once the image cache is warm)

# SAHI-style sliced inference — recommended for the large panoramas (~7571x2619):
# detects on overlapping tiles so small objects aren't lost in the 512px downscale.
export DET_SLICED="1"          # 1 = on, 0 = single-pass whole-image inference
export DET_SLICE="1024"        # tile size in original pixels
export DET_SLICE_OVERLAP="0.2" # fractional tile overlap (0..1)

# Force a device if auto-detect is wrong:  cuda | dml | cpu
# export FORCE_DEVICE="cuda"

# Local-file serving (so images/ is served without re-upload)
export LOCAL_FILES_SERVING_ENABLED="true"
export LOCAL_FILES_DOCUMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Env loaded. LS=$LABEL_STUDIO_URL ML=$ML_BACKEND_URL retrain every $RETRAIN_EVERY."
if [ "$LABEL_STUDIO_API_KEY" = "PASTE_YOUR_TOKEN_HERE" ]; then
  echo "WARNING: set LABEL_STUDIO_API_KEY in scripts/env.sh before bootstrapping."
fi
