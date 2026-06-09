"""Print the current training state (counter, active model, last metrics)."""
import json
from pathlib import Path

state_path = Path(__file__).resolve().parent.parent / "data" / "state.json"
if not state_path.exists():
    print("No training has happened yet (data/state.json missing).")
else:
    s = json.loads(state_path.read_text())
    print(json.dumps(s, indent=2))
    seen = s.get("annotations_seen", 0)
    every = 25
    print(f"\nAnnotated images: {seen}")
    print(f"Train runs:       {s.get('train_runs', 0)}")
    print(f"Architecture:     {s.get('variant', '?')}  (active: {s.get('active_weights')})")
    print(f"Next retrain at:  {(seen // every + 1) * every}")
    if s.get("last_metrics"):
        m = s["last_metrics"]
        print(f"Last run: best_val_loss={m.get('best_val_loss'):.4f} "
              f"on {m.get('images')} images, {m.get('epochs')} epochs, "
              f"{m.get('duration_s')}s")
