from fastapi import FastAPI
from scripts.inference.forecasting_600mw import Forecasting600MW
from pydantic import BaseModel
from typing import Dict
import pandas as pd

app = FastAPI(
    title="AI_SCADA API",
    description="AI service layer for the AI_SCADA smart power system",
    version="1.0.0",
)
forecasting_engine = Forecasting600MW()

class ForecastRequest(BaseModel):
    features: Dict[str, float]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AI_SCADA",
    }

@app.get("/models")
def models():
    return {
        "forecasting": {
            "model_family": "600 MW Power Output Forecasting",
            "horizons": [
                "2 minutes",
                "10 minutes",
                "30 minutes",
            ],
        },
        "anomaly_detection": {
            "model": "HAI 23.05 Candidate C Isolation Forest",
            "features": 118,
        },
    }

@app.post("/forecast")
def forecast(request: ForecastRequest):

    data = pd.DataFrame([request.features])

    predictions = forecasting_engine.predict_dict(data)

    return {
        "status": "success",
        "forecast": predictions
    }
