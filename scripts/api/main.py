# ============================================================
# AI_SCADA FASTAPI APPLICATION
# ============================================================

# ============================================================
# 1. IMPORTS
# ============================================================

from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.project_paths import FEATURES_DIR

from scripts.database.postgres_storage import PostgreSQLStorage

from scripts.decision_support.decision_engine import (
    DecisionInput,
    DecisionSupportEngine,
)

from scripts.inference.forecasting_600mw import (
    Forecasting600MW,
)

from scripts.inference.hai_attack_classifier_inference import (
    HAIAttackClassifierInference,
)

from scripts.inference.hai_attack_classifier_runtime import (
    HAIHistoricalFeatureBuilder,
)

from scripts.inference.hai_candidate_C_inference import (
    HAICandidateCInference,
)


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI_SCADA API",
    description="AI service layer for the AI_SCADA smart power system",
    version="1.0.0",
)


# ============================================================
# 3. CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 4. MODEL / ENGINE INITIALIZATION
# ============================================================


# ------------------------------------------------------------
# 4.1 600 MW FORECASTING MODEL
# ------------------------------------------------------------

forecasting_engine = Forecasting600MW()


# ------------------------------------------------------------
# 4.2 HAI CANDIDATE C ANOMALY MODEL
# ------------------------------------------------------------

HAI_FINAL_DIR = (
    FEATURES_DIR
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
    / "final_candidate"
)

HAI_MODEL_PATH = (
    HAI_FINAL_DIR
    / "hai_2305_candidate_C_isolation_forest.joblib"
)

HAI_MANIFEST_PATH = (
    HAI_FINAL_DIR
    / "hai_2305_candidate_C_feature_manifest.csv"
)

hai_engine = HAICandidateCInference(
    str(HAI_MODEL_PATH),
    str(HAI_MANIFEST_PATH),
)


# ------------------------------------------------------------
# 4.3 HAI ATTACK CLASSIFIER
# ------------------------------------------------------------

attack_feature_builder = HAIHistoricalFeatureBuilder()

attack_classifier = HAIAttackClassifierInference()


# ------------------------------------------------------------
# 4.4 DECISION SUPPORT ENGINE
# ------------------------------------------------------------

decision_engine = DecisionSupportEngine()


# ------------------------------------------------------------
# 4.5 POSTGRESQL STORAGE
# ------------------------------------------------------------

postgres_storage = PostgreSQLStorage()


# ============================================================
# 5. REQUEST MODELS
# ============================================================


class ForecastRequest(BaseModel):
    features: Dict[str, float]


class AnomalyRequest(BaseModel):
    features: Dict[str, float]


class DecisionRequest(BaseModel):
    features: Dict[str, float]
    persistent_anomaly_count: int = 0


# ============================================================
# 6. HEALTH & SYSTEM INFORMATION
# ============================================================


# ------------------------------------------------------------
# 6.1 HEALTH CHECK
# ------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "AI_SCADA",
    }


# ------------------------------------------------------------
# 6.2 MODEL INFORMATION
# ------------------------------------------------------------

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
            "domain": "HAI 23.05",
            "features": 118,
            "supports_600mw": False,
        },

        "attack_classification": {
            "model": "HAI 23.05 Temporal Multilabel GPU Classifier",
            "domain": "HAI 23.05",
            "features": 408,
            "target_mechanisms": 39,
        },

        "decision_support": {
            "supported_domains": [
                "600 MW",
                "HAI 23.05",
            ],
            "600mw_anomaly_state": "UNAVAILABLE",
            "levels": [
                "LEVEL_1",
                "LEVEL_2",
                "LEVEL_3",
            ],
        },
    }


# ============================================================
# 7. AI INFERENCE ENDPOINTS
# ============================================================


# ------------------------------------------------------------
# 7.1 600 MW FORECAST
# ------------------------------------------------------------

@app.post("/forecast")
def forecast(request: ForecastRequest):

    data = pd.DataFrame(
        [request.features]
    )

    predictions = forecasting_engine.predict_dict(
        data
    )

    return {
        "status": "success",
        "forecast": predictions,
    }


# ------------------------------------------------------------
# 7.2 HAI ANOMALY DETECTION
# ------------------------------------------------------------

@app.post("/anomaly")
def anomaly(request: AnomalyRequest):

    data = pd.DataFrame(
        [request.features]
    )

    result = hai_engine.predict_stream(
        data
    )

    row = result.iloc[0]

    return {
        "status": "success",

        "anomaly": {
            "anomaly_score": (
                None
                if pd.isna(row["anomaly_score"])
                else float(row["anomaly_score"])
            ),

            "prediction": (
                None
                if pd.isna(row["prediction"])
                else int(row["prediction"])
            ),

            "state": row["status"],
        },
    }


# ------------------------------------------------------------
# 7.3 DECISION SUPPORT
# ------------------------------------------------------------

@app.post("/decision")
def decision(request: DecisionRequest):

    data = pd.DataFrame(
        [request.features]
    )

    feature_names = set(
        request.features
    )

    # --------------------------------------------------------
    # DETERMINE MODEL DOMAIN
    # --------------------------------------------------------

    missing_600mw_features = [
        feature
        for feature in forecasting_engine.features
        if feature not in feature_names
    ]

    missing_hai_features = [
        feature
        for feature in hai_engine.original_features
        if feature not in feature_names
    ]

    if not missing_600mw_features:

        model_domain = "600MW"

    elif not missing_hai_features:

        model_domain = "HAI_23_05"

    else:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "The feature schema is not supported by the "
                    "600 MW forecasting or HAI 23.05 anomaly "
                    "components. Supply the required feature set "
                    "for one supported domain."
                ),

                "received_feature_count": len(
                    feature_names
                ),

                "supported_schemas": {
                    "600MW": {
                        "required_feature_count": len(
                            forecasting_engine.features
                        ),
                    },

                    "HAI_23_05": {
                        "required_feature_count": len(
                            hai_engine.original_features
                        ),

                        "classifier_feature_count": len(
                            attack_feature_builder.original_features
                        ),
                    },
                },
            },
        )

    # --------------------------------------------------------
    # INITIALIZE RESULTS
    # --------------------------------------------------------

    forecast_result = None
    anomaly_score = None
    classifier_result = None
    anomaly_unavailable_reason = None


    # --------------------------------------------------------
    # RUN DOMAIN-SPECIFIC MODELS
    # --------------------------------------------------------

    if model_domain == "600MW":

        forecast_result = (
            forecasting_engine.predict_dict(
                data
            )
        )

        anomaly_state = "UNAVAILABLE"

        anomaly_unavailable_reason = (
            "The 600 MW stream has forecasting support, "
            "but no compatible anomaly detector is available."
        )

    else:

        # Candidate C only receives data with
        # its HAI source schema.

        anomaly_result = (
            hai_engine.predict_stream(
                data
            )
        )

        anomaly_row = anomaly_result.iloc[0]

        anomaly_score = (
            None
            if pd.isna(
                anomaly_row["anomaly_score"]
            )
            else float(
                anomaly_row["anomaly_score"]
            )
        )

        anomaly_state = str(
            anomaly_row["status"]
        )

        # ----------------------------------------------------
        # HAI ATTACK CLASSIFIER
        # ----------------------------------------------------

        if set(
            attack_feature_builder.original_features
        ).issubset(feature_names):

            classifier_features = (
                attack_feature_builder.update(
                    request.features
                )
            )

            if classifier_features is not None:

                classifier_result = (
                    attack_classifier.predict(
                        classifier_features
                    )
                )


    # --------------------------------------------------------
    # BUILD DECISION INPUT
    # --------------------------------------------------------

    decision_input = DecisionInput(

        forecast_2min=(
            None
            if forecast_result is None
            else forecast_result[
                "power_2min"
            ]
        ),

        forecast_10min=(
            None
            if forecast_result is None
            else forecast_result[
                "power_10min"
            ]
        ),

        forecast_30min=(
            None
            if forecast_result is None
            else forecast_result[
                "power_30min"
            ]
        ),

        anomaly_score=anomaly_score,

        anomaly_state=anomaly_state,

        anomaly_unavailable_reason=(
            anomaly_unavailable_reason
        ),

        persistent_anomaly_count=(
            request.persistent_anomaly_count
        ),

        predicted_attack_mechanism=(
            None
            if classifier_result is None
            else classifier_result[
                "predicted_attack_mechanism"
            ]
        ),

        classifier_score=(
            None
            if classifier_result is None
            else classifier_result[
                "classifier_score"
            ]
        ),
    )


    # --------------------------------------------------------
    # EVALUATE DECISION
    # --------------------------------------------------------

    decision_result = (
        decision_engine.evaluate(
            decision_input
        )
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "status": "success",

        "model_domain": model_domain,

        "forecast": forecast_result,

        "decision": {
            "decision_level": (
                decision_result.decision_level
            ),

            "action": (
                decision_result.action
            ),

            "reason": (
                decision_result.reason
            ),

            "predicted_attack_mechanism": (
                decision_result.predicted_attack_mechanism
            ),

            "classifier_score": (
                decision_result.classifier_score
            ),

            "anomaly_score": (
                decision_result.anomaly_score
            ),

            "anomaly_state": anomaly_state,
        },
    }


# ============================================================
# 8. RUNTIME / DATABASE ENDPOINTS
# ============================================================


# ------------------------------------------------------------
# 8.1 LATEST RUNTIME MEASUREMENT
# ------------------------------------------------------------

@app.get("/runtime/latest")
def runtime_latest():

    measurement = (
        postgres_storage.get_latest_measurement(
            source_id=1,
            equipment_id=3,
            parameter_id=1,
        )
    )

    return {
        "status": "success",
        "measurement": measurement,
    }


# ------------------------------------------------------------
# 8.2 LATEST FORECAST PREDICTIONS
# ------------------------------------------------------------

@app.get("/predictions/latest")
def predictions_latest():

    predictions = (
        postgres_storage.get_latest_predictions(
            source_id=1,
            equipment_id=3,
            prediction_type="power_forecast",
        )
    )

    return {
        "status": "success",
        "predictions": predictions,
    }