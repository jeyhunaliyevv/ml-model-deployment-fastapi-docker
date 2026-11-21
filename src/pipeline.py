"""Leakage-aware preprocessing, feature creation, and transparent cost label."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
# Explicit, illustrative policy threshold chosen BEFORE inspection/splitting.
# Not a medical/actuarial standard or a claim-paid amount.
HIGH_COST_USD = 20_000.0
RAW_FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
CAT = ["sex", "smoker", "region"]
NUM = ["age", "bmi", "children", "obese", "smoker_obese", "age_smoker"]
ALLOWED = {"sex": {"male", "female"}, "smoker": {"yes", "no"},
           "region": {"northeast", "northwest", "southeast", "southwest"}}

def validate_raw(df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    missing = set(RAW_FEATURES + (["charges"] if require_target else [])) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    clean = df.copy()
    for col in ["age", "bmi", "children"] + (["charges"] if require_target else []):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    if clean[["age", "bmi", "children"]].isna().any().any():
        raise ValueError("Missing or nonnumeric numeric input")
    if ((clean.age < 0) | (clean.age > 120) | (clean.bmi <= 0) |
            (clean.children < 0) | (clean.children % 1 != 0)).any():
        raise ValueError("Age, BMI, or children outside valid ranges")
    for col, choices in ALLOWED.items():
        if clean[col].isna().any() or not clean[col].isin(choices).all():
            raise ValueError(f"Invalid category in {col}")
    if require_target and (clean.charges.isna().any() or (clean.charges < 0).any()):
        raise ValueError("charges must be nonnegative, finite numeric values")
    return clean

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    x = validate_raw(df)[RAW_FEATURES].copy()
    x["obese"] = (x.bmi >= 30).astype(int)
    x["smoker_obese"] = ((x.smoker == "yes") & (x.bmi >= 30)).astype(int)
    x["age_smoker"] = x.age * (x.smoker == "yes").astype(int)
    return x

def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUM),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ], sparse_threshold=0)

def make_target(charges: pd.Series, threshold: float = HIGH_COST_USD) -> pd.Series:
    return (charges >= threshold).astype(int)
