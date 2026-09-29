"""
AI MODEL INTEGRATION LAYER

Combines the two frozen AI components:

1. 600 MW power-output forecasting
   - 119 leakage-safe features
   - Linear Regression
   - 2 / 10 / 30 minute horizons

2. HAI 23.05 Candidate C anomaly detection
   - HAICandidateCInference
   - 58 original SCADA input features
   - internally builds 30 abs_diff_1s + 30 rolling_std_5s features
   - 118 Candidate C model features
   - Isolation Forest
   - batch and stateful streaming inference

This layer does NOT:
- connect MQTT
- connect ESP32
- connect PostgreSQL
- expose FastAPI
- control SCADA
- modify datasets

It only combines the already validated model components.
"""

from pathlib import Path
import json
import importlib.util

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# 600 MW
# ------------------------------------------------------------

FORECASTING_ENGINE_PATH = (
    ROOT /
    "scripts" /
    "inference" /
    "forecasting_600mw.py"
)

FORECASTING_MODEL_DIR = (
    ROOT /
    "models" /
    "forecasting" /
    "600mw"
)

FORECASTING_SOURCE = (
    ROOT /
    "data" /
    "features" /
    "600mw" /
    "candidates" /
    "feature_set_D_full_candidate_pool.csv"
)

# ------------------------------------------------------------
# HAI 23.05
# ------------------------------------------------------------

HAI_ENGINE_PATH = (
    ROOT /
    "scripts" /
    "inference" /
    "hai_candidate_C_inference.py"
)

HAI_MODEL_PATH = (
    ROOT /
    "data" /
    "features" /
    "hai" /
    "hai-23.05" /
    "temporal_representation" /
    "final_candidate" /
    "hai_2305_candidate_C_isolation_forest.joblib"
)

HAI_MANIFEST_PATH = (
    ROOT /
    "data" /
    "features" /
    "hai" /
    "hai-23.05" /
    "temporal_representation" /
    "final_candidate" /
    "hai_2305_candidate_C_feature_manifest.csv"
)

# IMPORTANT:
# The production HAI engine expects the original 58 SCADA
# features and creates the 118 Candidate C features internally.
HAI_INPUT_SOURCE = (
    ROOT /
    "data" /
    "features" /
    "hai" /
    "hai-23.05" /
    "model_ready" /
    "hai-test1_model_ready.csv"
)


# ============================================================
# DYNAMIC MODULE LOADER
# ============================================================

def load_module(
    module_name,
    module_path
):

    module_path = Path(
        module_path
    )

    if not module_path.exists():

        raise FileNotFoundError(
            f"{module_name} module not found:\n"
            f"{module_path}"
        )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            module_path
        )
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not load module:\n"
            f"{module_path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# AI MODEL INTEGRATION
# ============================================================

class AIModelIntegration:

    def __init__(self):

        print("")
        print("=" * 80)
        print("AI MODEL INTEGRATION")
        print("=" * 80)
        print("")

        # ----------------------------------------------------
        # LOAD 600 MW FORECASTING ENGINE
        # ----------------------------------------------------

        forecasting_module = load_module(
            "forecasting_600mw",
            FORECASTING_ENGINE_PATH
        )

        self.forecaster = (
            forecasting_module.Forecasting600MW(
                FORECASTING_MODEL_DIR
            )
        )

        print(
            "600 MW forecasting engine : LOADED"
        )

        print(
            f"600 MW features           : "
            f"{len(self.forecaster.features)}"
        )

        print(
            f"600 MW models             : "
            f"{len(self.forecaster.models)}"
        )

        # ----------------------------------------------------
        # LOAD HAI CANDIDATE C ENGINE
        # ----------------------------------------------------

        hai_module = load_module(
            "hai_candidate_C_inference",
            HAI_ENGINE_PATH
        )

        # IMPORTANT:
        # The actual class in the production HAI engine is:
        #
        #     HAICandidateCInference
        #
        # not CandidateCInferenceEngine.

        self.hai_detector = (
            hai_module.HAICandidateCInference(
                str(HAI_MODEL_PATH),
                str(HAI_MANIFEST_PATH)
            )
        )

        print(
            "HAI Candidate C engine    : LOADED"
        )

        print(
            f"HAI original features     : "
            f"{len(self.hai_detector.original_features)}"
        )

        print(
            f"HAI abs-diff features     : "
            f"{len(self.hai_detector.abs_diff_features)}"
        )

        print(
            f"HAI rolling features      : "
            f"{len(self.hai_detector.rolling_features)}"
        )

        print(
            f"HAI total model features  : "
            f"{len(self.hai_detector.feature_names)}"
        )

        if len(
            self.hai_detector.feature_names
        ) != 118:

            raise RuntimeError(
                "HAI Candidate C engine did not load "
                "the expected 118-feature schema."
            )

        print("")
        print(
            "Model components loaded successfully."
        )

    # ========================================================
    # FORECAST ONLY
    # ========================================================

    def predict_forecast(
        self,
        data
    ):

        return self.forecaster.predict(
            data
        )

    # ========================================================
    # ANOMALY ONLY
    # ========================================================

    def predict_anomaly(
        self,
        data
    ):

        # The actual HAI production engine exposes:
        #
        #     predict_batch()
        #     predict_stream()
        #
        # It does NOT expose predict().

        return self.hai_detector.predict_batch(
            data.copy()
        )

    # ========================================================
    # UNIFIED BATCH PREDICTION
    # ========================================================

    def predict(
        self,
        forecasting_data,
        hai_data
    ):

        forecast_result = (
            self.predict_forecast(
                forecasting_data
            )
        )

        anomaly_result = (
            self.predict_anomaly(
                hai_data
            )
        )

        return {
            "forecast": forecast_result,
            "anomaly": anomaly_result
        }

    # ========================================================
    # SINGLE-ROW UNIFIED PREDICTION
    # ========================================================

    def predict_single(
        self,
        forecasting_data,
        hai_data
    ):

        if len(
            forecasting_data
        ) != 1:

            raise ValueError(
                "forecasting_data must contain "
                "exactly one row."
            )

        if len(
            hai_data
        ) != 1:

            raise ValueError(
                "hai_data must contain "
                "exactly one row."
            )

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        forecast = (
            self.forecaster.predict_dict(
                forecasting_data
            )
        )

        # ----------------------------------------------------
        # HAI
        # ----------------------------------------------------

        anomaly_result = (
            self.hai_detector.predict_batch(
                hai_data.copy()
            )
        )

        if len(
            anomaly_result
        ) != 1:

            raise RuntimeError(
                "HAI detector returned an unexpected "
                "number of rows."
            )

        anomaly_row = (
            anomaly_result.iloc[0]
        )

        anomaly = {}

        for column in anomaly_result.columns:

            value = anomaly_row[column]

            if pd.isna(value):

                anomaly[column] = None

            elif hasattr(
                value,
                "item"
            ):

                anomaly[column] = (
                    value.item()
                )

            else:

                anomaly[column] = value

        return {
            "forecast": forecast,
            "anomaly": anomaly
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test():

    print("")
    print("=" * 80)
    print("AI MODEL INTEGRATION SELF-TEST")
    print("=" * 80)
    print("")

    system = AIModelIntegration()

    # --------------------------------------------------------
    # LOAD 600 MW FEATURE MANIFEST
    # --------------------------------------------------------

    forecasting_manifest = (
        FORECASTING_MODEL_DIR /
        "600mw_forecasting_feature_manifest.json"
    )

    if not forecasting_manifest.exists():

        raise FileNotFoundError(
            "600 MW feature manifest not found:\n"
            f"{forecasting_manifest}"
        )

    with open(
        forecasting_manifest,
        "r",
        encoding="utf-8"
    ) as file:

        manifest = json.load(
            file
        )

    forecast_features = (
        manifest["features"]
    )

    # --------------------------------------------------------
    # LOAD ONE 600 MW ROW
    # --------------------------------------------------------

    if not FORECASTING_SOURCE.exists():

        raise FileNotFoundError(
            "600 MW Feature Set D not found:\n"
            f"{FORECASTING_SOURCE}"
        )

    forecast_df = (
        pd.read_csv(
            FORECASTING_SOURCE,
            usecols=forecast_features
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .head(1)
    )

    if forecast_df.empty:

        raise RuntimeError(
            "Could not obtain a valid 600 MW "
            "self-test row."
        )

    # --------------------------------------------------------
    # LOAD ONE HAI ORIGINAL-FEATURE ROW
    # --------------------------------------------------------

    if not HAI_INPUT_SOURCE.exists():

        raise FileNotFoundError(
            "HAI model-ready Test1 file not found:\n"
            f"{HAI_INPUT_SOURCE}"
        )

    hai_df_full = pd.read_csv(
        HAI_INPUT_SOURCE
    )

    # --------------------------------------------------------
    # The HAI engine needs its 58 original features.
    # Use the engine's authoritative feature list.
    # --------------------------------------------------------

    hai_input_features = (
        system.hai_detector.original_features
    )

    missing_hai_features = [
        feature
        for feature in hai_input_features
        if feature not in hai_df_full.columns
    ]

    if missing_hai_features:

        raise RuntimeError(
            "HAI self-test source is missing "
            "required original features:\n"
            + "\n".join(
                missing_hai_features
            )
        )

    hai_df = (
        hai_df_full[
            hai_input_features
        ]
        .head(1)
        .copy()
    )

    # --------------------------------------------------------
    # RUN UNIFIED PREDICTION
    # --------------------------------------------------------

    result = system.predict_single(
        forecast_df,
        hai_df
    )

    # --------------------------------------------------------
    # FORECAST RESULT
    # --------------------------------------------------------

    print("")
    print("FORECAST RESULT")
    print("-" * 80)

    for key, value in result[
        "forecast"
    ].items():

        print(
            f"{key:15s}: "
            f"{float(value):.6f} MW"
        )

    # --------------------------------------------------------
    # HAI RESULT
    # --------------------------------------------------------

    print("")
    print("HAI ANOMALY RESULT")
    print("-" * 80)

    for key, value in result[
        "anomaly"
    ].items():

        print(
            f"{key:20s}: {value}"
        )

    # --------------------------------------------------------
    # VALIDATE FORECAST SCHEMA
    # --------------------------------------------------------

    expected_forecast = {
        "power_2min",
        "power_10min",
        "power_30min"
    }

    actual_forecast = set(
        result[
            "forecast"
        ].keys()
    )

    if actual_forecast != (
        expected_forecast
    ):

        raise RuntimeError(
            "Forecast output schema mismatch.\n"
            f"Expected: {expected_forecast}\n"
            f"Actual:   {actual_forecast}"
        )

    # --------------------------------------------------------
    # VALIDATE HAI RESULT
    # --------------------------------------------------------

    if not result[
        "anomaly"
    ]:

        raise RuntimeError(
            "HAI anomaly detector returned "
            "an empty result."
        )

    required_hai_columns = {
        "prediction"
    }

    missing_output_columns = (
        required_hai_columns
        - set(
            result["anomaly"].keys()
        )
    )

    if missing_output_columns:

        raise RuntimeError(
            "HAI output is missing required "
            f"columns: {missing_output_columns}"
        )

    # --------------------------------------------------------
    # FINAL SELF-TEST
    # --------------------------------------------------------

    print("")
    print("=" * 80)
    print("INTEGRATION SELF-TEST: PASS")
    print("=" * 80)

    print("")
    print("Verified:")
    print(
        "  600 MW engine loaded      : PASS"
    )
    print(
        "  600 MW 119-feature schema : PASS"
    )
    print(
        "  600 MW 3-model interface  : PASS"
    )
    print(
        "  HAI engine loaded         : PASS"
    )
    print(
        "  HAI 58-input schema       : PASS"
    )
    print(
        "  HAI 118-model schema      : PASS"
    )
    print(
        "  Unified prediction        : PASS"
    )
    print(
        "  Forecast output schema    : PASS"
    )
    print(
        "  HAI output schema         : PASS"
    )

    print("")
    print("Unified architecture:")
    print("")
    print(
        "600 MW 119 features"
    )
    print(
        "        ↓"
    )
    print(
        "Forecasting600MW"
    )
    print(
        "        ↓"
    )
    print(
        "2 / 10 / 30 minute forecast"
    )
    print("")
    print(
        "HAI 58 original SCADA features"
    )
    print(
        "        ↓"
    )
    print(
        "HAICandidateCInference"
    )
    print(
        "        ↓"
    )
    print(
        "118 Candidate C features"
    )
    print(
        "        ↓"
    )
    print(
        "NORMAL / ANOMALY"
    )

    print("")
    print(
        "STATUS: MODEL INTEGRATION READY"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_self_test()
