"""
600 MW Forecasting Inference Engine

Loads the frozen 600 MW forecasting models and the authoritative
119-feature manifest.

Expected input:
    pandas DataFrame containing the 119 model features.

Output:
    pandas DataFrame containing:
        power_2min
        power_10min
        power_30min
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


class Forecasting600MW:

    def __init__(self, model_dir=None):

        # ----------------------------------------------------
        # MODEL DIRECTORY
        # ----------------------------------------------------

        if model_dir is None:

            model_dir = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "forecasting"
                / "600mw"
            )

        self.model_dir = Path(model_dir)

        # ----------------------------------------------------
        # LOAD FEATURE MANIFEST
        # ----------------------------------------------------

        manifest_path = (
            self.model_dir
            / "600mw_forecasting_feature_manifest.json"
        )

        if not manifest_path.exists():

            raise FileNotFoundError(
                "600 MW feature manifest not found:\n"
                f"{manifest_path}"
            )

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.manifest = json.load(file)

        # ----------------------------------------------------
        # LOAD MODEL METADATA
        # ----------------------------------------------------

        metadata_path = (
            self.model_dir
            / "600mw_forecasting_model_metadata.json"
        )

        if not metadata_path.exists():

            raise FileNotFoundError(
                "600 MW model metadata not found:\n"
                f"{metadata_path}"
            )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        # ----------------------------------------------------
        # FEATURE SCHEMA
        # ----------------------------------------------------

        self.features = self.manifest[
            "features"
        ]

        expected_count = self.manifest[
            "feature_count"
        ]

        if expected_count != 119:

            raise ValueError(
                "Invalid 600 MW manifest. "
                f"Expected 119 features, got {expected_count}."
            )

        if len(self.features) != 119:

            raise ValueError(
                "Feature manifest contains an unexpected "
                f"number of features: {len(self.features)}"
            )

        # ----------------------------------------------------
        # LOAD FINAL MODELS
        # ----------------------------------------------------

        self.models = {}

        for target in self.manifest[
            "targets"
        ]:

            horizon = target.replace(
                "target_power_",
                ""
            )

            model_path = (
                self.model_dir
                / f"600mw_{horizon}_final.joblib"
            )

            if not model_path.exists():

                raise FileNotFoundError(
                    "600 MW model not found:\n"
                    f"{model_path}"
                )

            self.models[target] = joblib.load(
                model_path
            )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def validate_input(
        self,
        data
    ):

        if not isinstance(
            data,
            pd.DataFrame
        ):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if data.empty:

            raise ValueError(
                "Input DataFrame is empty."
            )

        # ----------------------------------------------------
        # CHECK REQUIRED FEATURES
        # ----------------------------------------------------

        missing = [
            feature
            for feature in self.features
            if feature not in data.columns
        ]

        if missing:

            raise ValueError(
                "Missing required 600 MW features:\n"
                + "\n".join(missing)
            )

        # ----------------------------------------------------
        # CHECK NUMERIC FEATURES
        # ----------------------------------------------------

        non_numeric = [
            feature
            for feature in self.features
            if not pd.api.types.is_numeric_dtype(
                data[feature]
            )
        ]

        if non_numeric:

            raise TypeError(
                "The following required features "
                "are not numeric:\n"
                + "\n".join(non_numeric)
            )

        # ----------------------------------------------------
        # CHECK MISSING VALUES
        # ----------------------------------------------------

        missing_values = (
            data[
                self.features
            ]
            .isna()
            .sum()
            .sum()
        )

        if missing_values > 0:

            raise ValueError(
                "Input contains "
                f"{missing_values} missing feature values."
            )

        # ----------------------------------------------------
        # CHECK INFINITE / INVALID VALUES
        # ----------------------------------------------------

        numeric_block = data[
            self.features
        ]

        if not np.isfinite(
            numeric_block.to_numpy(dtype=float)
        ).all():

            raise ValueError(
                "Input contains infinite or invalid "
                "numeric feature values."
            )

        return True

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        data
    ):

        self.validate_input(
            data
        )

        X = data[
            self.features
        ].copy()

        predictions = pd.DataFrame(
            index=data.index
        )

        # ----------------------------------------------------
        # 2-MINUTE FORECAST
        # ----------------------------------------------------

        predictions[
            "power_2min"
        ] = self.models[
            "target_power_2min"
        ].predict(X)

        # ----------------------------------------------------
        # 10-MINUTE FORECAST
        # ----------------------------------------------------

        predictions[
            "power_10min"
        ] = self.models[
            "target_power_10min"
        ].predict(X)

        # ----------------------------------------------------
        # 30-MINUTE FORECAST
        # ----------------------------------------------------

        predictions[
            "power_30min"
        ] = self.models[
            "target_power_30min"
        ].predict(X)

        return predictions

    # ========================================================
    # SINGLE-ROW PREDICTION
    # ========================================================

    def predict_dict(
        self,
        data
    ):

        if len(data) != 1:

            raise ValueError(
                "predict_dict() expects exactly one row."
            )

        result = self.predict(
            data
        )

        row = result.iloc[0]

        return {
            "power_2min": float(
                row["power_2min"]
            ),
            "power_10min": float(
                row["power_10min"]
            ),
            "power_30min": float(
                row["power_30min"]
            )
        }
