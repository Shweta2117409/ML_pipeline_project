"""
api.py
------
FastAPI application for the Heart Disease Prediction service.

Endpoints:
    GET  /          → health check / welcome message
    GET  /health    → detailed health check
    POST /predict   → return 0/1 heart disease prediction
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

# Frontend dir (project root / frontend) when running from project root
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ---------------------------------------------------------------------------
# Feature column order must exactly match training order
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal",
]

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and release on shutdown."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file '{MODEL_PATH}' not found. "
            "Run `python train_model.py` first."
        )
    app_state["model"] = joblib.load(MODEL_PATH)
    print(f"[API] Model loaded successfully from '{MODEL_PATH}'.")
    yield
    app_state.clear()
    print("[API] Model unloaded.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "An ML-powered REST API that predicts the likelihood of heart disease "
        "based on clinical features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PatientFeatures(BaseModel):
    age:      float = Field(..., ge=1,   le=120, description="Age in years")
    sex:      int   = Field(..., ge=0,   le=1,   description="Sex (1=male, 0=female)")
    cp:       int   = Field(..., ge=0,   le=3,   description="Chest pain type (0–3)")
    trestbps: float = Field(..., ge=80,  le=220, description="Resting blood pressure (mm Hg)")
    chol:     float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs:      int   = Field(..., ge=0,   le=1,   description="Fasting blood sugar > 120 (1=true)")
    restecg:  int   = Field(..., ge=0,   le=2,   description="Resting ECG results (0–2)")
    thalach:  float = Field(..., ge=60,  le=220, description="Maximum heart rate achieved")
    exang:    int   = Field(..., ge=0,   le=1,   description="Exercise induced angina (1=yes)")
    oldpeak:  float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope:    int   = Field(..., ge=0,   le=2,   description="Slope of peak exercise ST segment")
    ca:       int   = Field(..., ge=0,   le=4,   description="Number of major vessels (0–4)")
    thal:     int   = Field(..., ge=0,   le=3,   description="Thalassemia type (0–3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
                "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
                "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No heart disease, 1 = Heart disease")
    confidence: Optional[float] = Field(None, description="Model confidence (probability)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Avoid 404 when the browser requests a tab icon."""
    return Response(status_code=204)


@app.get("/", summary="Frontend")
def serve_frontend():
    """Serve the frontend UI so one server runs both API and app."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/welcome", summary="Welcome")
def root():
    return {"message": "Heart Disease Prediction API"}


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health():
    model = app_state.get("model")
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "none",
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict Heart Disease")
def predict(patient: PatientFeatures):
    """
    Predict whether a patient has heart disease.

    - **prediction**: 0 = No heart disease, 1 = Heart disease present
    - **confidence**: Probability of the positive class (if model supports it)
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Build DataFrame in the exact column order used during training
    data = {col: [getattr(patient, col)] for col in FEATURE_COLUMNS}
    X = pd.DataFrame(data)

    prediction = int(model.predict(X)[0])

    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = round(float(proba[prediction]), 4)

    return {"prediction": prediction, "confidence": confidence}
