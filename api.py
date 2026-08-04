"""
api.py
------
REST API for the Disease Prediction project (Level 8), built with FastAPI.

Run with:
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger docs, or:

    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"dataset": "diabetes", "features": {"Pregnancies": 2, "Glucose": 180, ...}}'

(Run `python main.py` at least once first so trained models exist in
outputs/models/)
"""

import json
import os
import pickle
from typing import Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_loader import DATASETS
from src.database import log_prediction
from src.constants import DATASET_LABELS, DISEASE_LABELS, DOCTOR_MAP, RECOMMENDATIONS
from src.utils import safe_model_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

app = FastAPI(
    title="Disease Prediction API",
    description="ML-powered disease prediction over breast cancer, heart disease, "
                 "and diabetes datasets.",
    version="1.0.0",
)

_model_cache = {}   # (dataset, safe_model_name) -> fitted model
_scaler_cache = {}  # dataset -> scaler
_feature_map_cache = {}  # dataset -> {"all_features": [...], "selected_features": [...]}


class PredictRequest(BaseModel):
    dataset: str = Field(..., description="One of: breast_cancer, heart, diabetes")
    model: Optional[str] = Field(
        None, description="Model display name, e.g. 'CatBoost'. Defaults to the "
                           "best model for this dataset by F1-Score if omitted."
    )
    features: Dict[str, float] = Field(..., description="feature_name -> value")


class PredictResponse(BaseModel):
    dataset: str
    model_used: str
    prediction: int
    disease: str
    confidence: Optional[float]
    suggested_doctor: Optional[str]
    recommendations: list


def _load_feature_map(dataset: str):
    if dataset not in _feature_map_cache:
        path = os.path.join(MODELS_DIR, f"{dataset}_feature_map.json")
        if not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail=f"No trained models for dataset '{dataset}'. Run `python main.py` first."
            )
        with open(path) as f:
            _feature_map_cache[dataset] = json.load(f)
    return _feature_map_cache[dataset]


def _load_scaler(dataset: str):
    if dataset not in _scaler_cache:
        path = os.path.join(MODELS_DIR, f"{dataset}_scaler.pkl")
        with open(path, "rb") as f:
            _scaler_cache[dataset] = pickle.load(f)
    return _scaler_cache[dataset]


def _default_model_name(dataset: str) -> str:
    """Best model for this dataset by F1-Score, from the training summary CSV."""
    summary_path = os.path.join(REPORTS_DIR, f"{dataset}_summary.csv")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail=f"No training summary for '{dataset}'.")
    summary_df = pd.read_csv(summary_path, index_col=0)
    return summary_df["F1-Score"].astype(float).idxmax()


def _load_model(dataset: str, model_name: str):
    key = (dataset, model_name)
    if key not in _model_cache:
        safe_name = safe_model_filename(model_name)
        path = os.path.join(MODELS_DIR, f"{dataset}_{safe_name}.pkl")
        if not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found for dataset '{dataset}'."
            )
        with open(path, "rb") as f:
            _model_cache[key] = pickle.load(f)
    return _model_cache[key]


@app.get("/")
def root():
    return {"message": "Disease Prediction API is running.", "docs": "/docs"}


@app.get("/datasets")
def list_datasets():
    """Lists available datasets and the exact feature names each expects."""
    result = {}
    for key in DATASETS:
        try:
            fm = _load_feature_map(key)
            result[key] = {
                "label": DATASET_LABELS.get(key, key),
                "all_features": fm["all_features"],
                "features_used_by_model": fm["selected_features"],
            }
        except HTTPException:
            result[key] = {"label": DATASET_LABELS.get(key, key), "status": "not trained yet"}
    return result


@app.get("/datasets/{dataset}/models")
def list_models(dataset: str):
    """Lists trained models available for a dataset, with their test metrics."""
    summary_path = os.path.join(REPORTS_DIR, f"{dataset}_summary.csv")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail=f"No trained models for '{dataset}'.")
    summary_df = pd.read_csv(summary_path, index_col=0)
    return json.loads(summary_df.reset_index().to_json(orient="records"))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.dataset not in DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{req.dataset}'. Choose one of: {list(DATASETS.keys())}"
        )

    feature_map = _load_feature_map(req.dataset)
    all_features = feature_map["all_features"]
    selected_features = feature_map["selected_features"]
    selected_positions = [all_features.index(f) for f in selected_features]

    missing = [f for f in all_features if f not in req.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features for '{req.dataset}': {missing}"
        )

    model_name = req.model or _default_model_name(req.dataset)
    model = _load_model(req.dataset, model_name)
    scaler = _load_scaler(req.dataset)

    input_df = pd.DataFrame([req.features])[all_features]
    input_scaled_full = scaler.transform(input_df)
    input_scaled = input_scaled_full[:, selected_positions]

    pred = int(model.predict(input_scaled)[0])
    proba = float(model.predict_proba(input_scaled)[0][1]) if hasattr(model, "predict_proba") else None

    disease_name = DISEASE_LABELS.get(req.dataset, "Disease")
    predicted_label = disease_name if pred == 1 else "No Disease"

    log_prediction(
        dataset=req.dataset, model_name=model_name, feature_values=req.features,
        predicted_disease=predicted_label, prediction=pred, confidence=proba,
    )

    return PredictResponse(
        dataset=req.dataset,
        model_used=model_name,
        prediction=pred,
        disease=predicted_label,
        confidence=proba,
        suggested_doctor=DOCTOR_MAP.get(req.dataset) if pred == 1 else None,
        recommendations=RECOMMENDATIONS.get(req.dataset, []) if pred == 1 else [],
    )
