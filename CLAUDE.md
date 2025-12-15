# Aurora ML Backend

ML backend for aurora borealis visibility predictions in the Cosmofy iOS app.

## Structure

```
aurora/
├── inference/           # Production API
│   ├── app.py           # FastAPI endpoints
│   ├── weather.py       # WeatherKit + Open-Meteo fallback
│   ├── space_weather.py # NOAA SWPC (free)
│   ├── astronomy.py     # Sun/moon calculations
│   └── features.py      # Feature engineering (30 features)
├── models/
│   ├── gb.joblib        # Gradient Boosting model
│   └── xgb.joblib       # XGBoost model
├── training/
│   └── train.py         # Retrain models
├── data/
│   └── training.csv     # Training dataset (5293 samples)
└── requirements.txt
```

## Commands

### Run API
```bash
uvicorn inference.app:app --host 0.0.0.0 --port 8080
```

### Retrain Models
```bash
python training/train.py
```

## API

### Endpoints
- `GET /health` - Health check
- `POST /predict` - Aurora prediction

### Example
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 64.1, "longitude": -21.9}'
```

### Response
```json
{
  "probability": 0.335,
  "confidence": "high",
  "gb_probability": 0.346,
  "xgb_probability": 0.325,
  "conditions": {
    "is_dark": true,
    "cloud_cover": 89,
    "kp_index": 4.0,
    "geomagnetic_storm": false,
    "moon_interference": true
  }
}
```

## Model

Dual model system (Gradient Boosting + XGBoost):
- Both agree within 10% → high confidence
- Disagree 10-20% → moderate confidence
- Disagree >20% → low confidence

Test accuracy: 95.2% (averaged), AUC: 0.99

## WeatherKit Setup

Set environment variables:
```bash
export WEATHERKIT_KEY_ID="your_key_id"
export WEATHERKIT_TEAM_ID="your_team_id"
export WEATHERKIT_SERVICE_ID="com.yourapp.weatherkit"
export WEATHERKIT_PRIVATE_KEY="/path/to/AuthKey.p8"
```

Falls back to Open-Meteo (free) if not configured.
