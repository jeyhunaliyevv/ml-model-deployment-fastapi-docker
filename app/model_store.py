"""Lazily load only a model artifact created locally by the training script."""
from functools import lru_cache
import os
from pathlib import Path
import joblib

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"

def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

@lru_cache(maxsize=1)
def get_model():
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found at {path}. Run the training script first.")
    # joblib uses pickle; load trusted locally generated artifacts ONLY.
    obj = joblib.load(path)
    if not callable(getattr(obj, "predict_proba", None)):
        raise ValueError("Model artifact does not support predict_proba")
    return obj
