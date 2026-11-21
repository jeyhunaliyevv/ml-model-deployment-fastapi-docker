"""Small production-shaped API demonstrating model serving, not clinical deployment."""
import logging
import math
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from .schemas import Customer, Prediction, Health
from . import model_store
from src.pipeline import engineer, HIGH_COST_USD

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Readiness is reported via /health, rather than hiding a missing artifact.
    try:
        model_store.get_model()
    except Exception as exc:
        logger.warning("ML model not ready: %s", type(exc).__name__)
    yield

app = FastAPI(title="Medical Insurance High-Cost Risk API", version="1.0.0", lifespan=lifespan)

@app.get("/health", response_model=Health, responses={503: {"description": "Model not ready"}})
def health():
    try:
        model_store.get_model()
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not_ready", "model_loaded": False})
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=Prediction, responses={503: {"description": "Model unavailable"}})
def predict(customer: Customer):
    try:
        model = model_store.get_model()
    except Exception as exc:
        logger.error("Model loading failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Model unavailable. Train the model before starting the service.") from exc
    try:
        frame = engineer(pd.DataFrame([customer.model_dump()]))
        probability = float(model.predict_proba(frame)[0, 1])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid model probability")
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return Prediction(high_cost_probability=round(probability, 4),
                      predicted_high_cost=probability >= 0.5,
                      threshold_usd=HIGH_COST_USD, decision_threshold=0.5)
