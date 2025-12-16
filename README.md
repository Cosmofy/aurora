  # Aurora
  
  ML backend for real-time aurora borealis visibility predictions.
  
  ## Live API
  
  **Endpoint:** `https://aurora.arryan.xyz/predict`
  
  ### Request
  
  ```bash
  curl -X POST https://aurora.arryan.xyz/predict \
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
  
  ### Parameters
  
  | Field | Type | Description |
  |-------|------|-------------|
  | latitude | float | Location latitude (-90 to 90) |
  | longitude | float | Location longitude (-180 to 180) |
  
  ### Response Fields
  
  | Field | Description |
  |-------|-------------|
  | probability | Aurora visibility probability (0-1) |
  | confidence | Model agreement level (high, moderate, low) |
  | gb_probability | Gradient Boosting model prediction |
  | xgb_probability | XGBoost model prediction |
  | conditions | Current environmental conditions |
  
  ## Model
  
  Dual-model ensemble using Gradient Boosting and XGBoost trained on 5,293 samples.
  
  - Test accuracy: 95.2%
  - AUC: 0.99
  
  Confidence scoring based on model agreement:
  - High: models agree within 10%
  - Moderate: models disagree 10-20%
  - Low: models disagree more than 20%
  
  ## Data Sources
  
  - **Weather:** Apple WeatherKit (with Open-Meteo fallback)
  - **Space Weather:** NOAA Space Weather Prediction Center
  - **Astronomy:** Solar/lunar position calculations
  
  ## Infrastructure
  
  - FastAPI backend
  - Kubernetes with Horizontal Pod Autoscaler (2-20 pods)
  - Redis caching for NOAA data
  - CI/CD via GitHub Actions and Ansible
  
  ## Local Development
  
  ```bash
  # Install dependencies
  pip install -r requirements.txt
  
  # Run API
  uvicorn inference.app:app --host 0.0.0.0 --port 8080
  
  # Retrain models
  python training/train.py
  ```
  
  ## License
  
  Proprietary. Part of the Cosmofy iOS app.
