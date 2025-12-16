"""
Aurora Prediction API

Endpoints:
- POST /predict - Get aurora probability for a location
- GET /health - Health check
"""

import os
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .astronomy import calculate_astronomy
from .features import FEATURE_ORDER, engineer_features
from .space_weather import get_space_weather
from .weather import get_weather

app = FastAPI(
    title="Aurora Prediction API",
    description="Predicts aurora borealis visibility probability",
    version="1.0.0",
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# Load models on startup
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    gb_model = joblib.load(os.path.join(MODEL_DIR, "models", "gb.joblib"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "models", "xgb.joblib"))
except FileNotFoundError:
    # Fallback for running from inference directory
    gb_model = joblib.load("../models/gb.joblib")
    xgb_model = joblib.load("../models/xgb.joblib")


class PredictRequest(BaseModel):
    latitude: float
    longitude: float
    # Optional: if not provided, uses current time
    timestamp: Optional[str] = None  # ISO format: "2024-12-11T22:00:00Z"


class PredictResponse(BaseModel):
    probability: float
    confidence: str  # "high", "moderate", "low"
    gb_probability: float
    xgb_probability: float
    conditions: dict


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        lat = request.latitude
        lon = request.longitude

        # 1. Get weather data
        weather = await get_weather(lat, lon)

        # 2. Get space weather data
        space = await get_space_weather()

        # 3. Calculate astronomical positions
        astro = calculate_astronomy(lat, lon, request.timestamp)

        # 4. Build feature vector
        features = engineer_features(lat, lon, weather, space, astro)

        # 5. Ensure correct feature order
        import pandas as pd

        X = pd.DataFrame([features])[FEATURE_ORDER]

        # 6. Get predictions from both models
        gb_prob = float(gb_model.predict_proba(X)[0][1])
        xgb_prob = float(xgb_model.predict_proba(X)[0][1])

        # 7. Calculate combined probability and confidence
        avg_prob = (gb_prob + xgb_prob) / 2
        diff = abs(gb_prob - xgb_prob)

        if diff < 0.1:
            confidence = "high"
        elif diff < 0.2:
            confidence = "moderate"
        else:
            confidence = "low"

        # 8. Build conditions summary
        conditions = {
            "is_dark": bool(astro["sun_altitude"] < -6),
            "cloud_cover": weather["cloudcover"],
            "kp_index": space["kp_index"],
            "geomagnetic_storm": bool(space["kp_index"] >= 5),
            "moon_interference": bool(
                astro["moon_altitude"] > 0 and astro["moon_illumination"] > 0.5
            ),
        }

        return PredictResponse(
            probability=round(avg_prob, 3),
            confidence=confidence,
            gb_probability=round(gb_prob, 3),
            xgb_probability=round(xgb_prob, 3),
            conditions=conditions,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=2260)
