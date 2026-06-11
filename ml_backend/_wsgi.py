"""
WSGI entrypoint for the YOLO Label Studio ML backend.

Run (from the ml_backend/ directory, with the venv active):
    python _wsgi.py            # dev server on http://localhost:9090

Label Studio talks to this server for predictions (predict) and training (fit).
Environment variables it reads:
    LABEL_STUDIO_URL       e.g. http://localhost:8080
    LABEL_STUDIO_API_KEY   access token from LS > Account & Settings
    RETRAIN_EVERY          default 25
    YOLO_BASE_MODEL        default yolo11s.pt
    YOLO_EPOCHS / YOLO_IMGSZ / YOLO_BATCH   training overrides
"""
import os

from label_studio_ml.api import init_app

from model import ScratchDetBackend

# model_dir is where label-studio-ml stores per-job working dirs; it must be a
# real path or the webhook/training job manager crashes (_job_dir on None).
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ml_jobs"),
)
os.makedirs(MODEL_DIR, exist_ok=True)

app = init_app(model_class=ScratchDetBackend, model_dir=MODEL_DIR)

if __name__ == "__main__":
    port = int(os.getenv("ML_PORT", "9090"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
