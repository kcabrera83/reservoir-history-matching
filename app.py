"""FastAPI web server for reservoir history matching."""

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from reservoir_history_matching.models.history_matcher import HistoryMatcher
from reservoir_history_matching.models.production_forecaster import ProductionForecaster
from reservoir_history_matching.utils.preprocessor import ReservoirPreprocessor

app = FastAPI(
    title="Reservoir History Matching",
    description="Reservoir production matching and future production forecasting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "outputs", "models")
models: dict[str, Any] = {}


@app.on_event("startup")
async def load_models():
    try:
        matcher_path = os.path.join(MODEL_DIR, "history_matcher.pkl")
        forecaster_path = os.path.join(MODEL_DIR, "production_forecaster.pkl")
        preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
        if os.path.exists(matcher_path):
            models["matcher"] = HistoryMatcher.load(matcher_path)
        if os.path.exists(forecaster_path):
            models["forecaster"] = ProductionForecaster.load(forecaster_path)
        if os.path.exists(preprocessor_path):
            models["preprocessor"] = ReservoirPreprocessor.load(preprocessor_path)
    except Exception as e:
        print(f"  Error loading models: {e}")


class PredictRequest(BaseModel):
    features: list[float]


class ForecastRequest(BaseModel):
    recent_data: list[dict]
    n_steps: int = 10


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "history_matcher": "matcher" in models and models["matcher"].is_trained,
            "production_forecaster": "forecaster" in models and models["forecaster"].is_trained,
        },
    }


@app.get("/api/models")
async def models_info():
    info = {}
    if "matcher" in models and models["matcher"].is_trained:
        m = models["matcher"]
        info["history_matcher"] = {
            "type": m.model_type,
            "metrics": m.metrics,
            "targets": m.targets,
        }
    if "forecaster" in models and models["forecaster"].is_trained:
        f = models["forecaster"]
        info["production_forecaster"] = {
            "lookback": f.lookback,
            "metrics": f.metrics,
            "targets": f.targets,
        }
    return info


@app.post("/api/predict")
async def predict(request: PredictRequest):
    if "matcher" not in models or not models["matcher"].is_trained:
        raise HTTPException(status_code=503, detail="History matcher not trained")
    if not request.features:
        raise HTTPException(status_code=400, detail="Missing features in request body")
    try:
        features = np.array(request.features).reshape(1, -1)
        features_scaled = models["preprocessor"].transform(features)
        prediction = models["matcher"].predict(features_scaled)
        result = prediction.iloc[0].to_dict()
        return {
            "prediction": {
                "oil_rate_bopd": round(float(result["oil_rate_bopd"]), 2),
                "water_rate_bwpd": round(float(result["water_rate_bwpd"]), 2),
                "gas_rate_mscfd": round(float(result["gas_rate_mscfd"]), 2),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/forecast")
async def forecast(request: ForecastRequest):
    if "forecaster" not in models or not models["forecaster"].is_trained:
        raise HTTPException(status_code=503, detail="Production forecaster not trained")
    if not request.recent_data:
        raise HTTPException(status_code=400, detail="Missing recent_data in request body")
    try:
        recent = pd.DataFrame(request.recent_data)
        forecasts = models["forecaster"].forecast_future(recent, n_steps=request.n_steps)
        result = forecasts.to_dict(orient="records")
        for r in result:
            for k in r:
                r[k] = round(float(r[k]), 2)
        return {"forecast": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5006)

