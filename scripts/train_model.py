"""Rebuild serving model with the same CatBoost hyperparameters as the source repo.

The source project's model was selected by 5-fold training CV. This deployment
stage trains that already-selected algorithm on ALL available source data.
It does not claim a fresh holdout score or tune models on the test set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from src.pipeline import engineer, make_target, preprocessor, validate_raw
from src.compat import CatBoostAdapter

ROOT = Path(__file__).resolve().parents[1]

def train(data_path: Path, output_path: Path):
    raw = validate_raw(pd.read_csv(data_path), require_target=True)
    features = engineer(raw)
    labels = make_target(raw["charges"])
    if labels.nunique() != 2:
        raise ValueError("Training data must contain both target classes")
    model = Pipeline([
        ("preprocessor", preprocessor()),
        ("classifier", CatBoostAdapter(iterations=180, depth=4, learning_rate=0.05, random_seed=42)),
    ])
    model.fit(features, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Trained CatBoost on {len(raw)} rows and saved to {output_path}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "insurance.csv"))
    parser.add_argument("--output", default=str(ROOT / "models" / "best_model.joblib"))
    args = parser.parse_args()
    train(Path(args.data), Path(args.output))

if __name__ == "__main__":
    main()
