
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Dict, Optional, List
import numpy as np
import joblib, os
from dataclasses import dataclass

API_KEY = os.getenv("API_KEY", "dev")
api_key_header = APIKeyHeader(name="X-API-Key")
app = FastAPI(title="Reservoir History Matching")

@dataclass
class ModelWrapper:
    model: object
    scaler: Optional[object]
    features: List[str]
    target: str

class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: float
    model: str

REGISTRY: Dict[str, ModelWrapper] = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        data = joblib.load(os.path.join("outputs/models", f))
        REGISTRY[f.replace(".pkl", "")] = ModelWrapper(
            model=data["model"],
            scaler=data.get("scaler"),
            features=data.get("feature_names", []),
            target=data.get("target_name", "target"),
        )

def verify(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    return key

@app.get("/")
def root():
    return {"service": "Reservoir History Matching", "models": list(REGISTRY.keys())}

@app.post("/predict/{model_name}", response_model=PredictionResponse)
def predict(model_name: str, req: PredictionRequest, _=Security(verify)):
    if model_name not in REGISTRY:
        raise HTTPException(404, f"Model {model_name} not found")
    wrapper = REGISTRY[model_name]
    X = np.array([req.features.get(f, 0) for f in wrapper.features]).reshape(1, -1)
    if wrapper.scaler:
        X = wrapper.scaler.transform(X)
    pred = wrapper.model.predict(X)[0]
    return PredictionResponse(prediction=float(pred), model=model_name)
