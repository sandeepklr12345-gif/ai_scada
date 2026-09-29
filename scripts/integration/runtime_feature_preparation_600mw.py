from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "600mw"
)

MANIFEST_PATH = (
    MODEL_DIR
    / "600mw_forecasting_feature_manifest.json"
)


# ============================================================
# RUNTIME FEATURE PREPARATION
# ============================================================

class RuntimeFeaturePreparation600MW:

    def __init__(self, manifest_path):

        self.manifest_path = Path(manifest_path)

        self.features = self._load_manifest()

        print(
            f"600 MW runtime feature schema loaded: "
            f"{len(self.features)} features"
        )

    # --------------------------------------------------------
    # Load authoritative model feature manifest
    # --------------------------------------------------------

    def _load_manifest(self):

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                f"Model feature manifest not found:\n"
                f"{self.manifest_path}"
            )

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as f:

            manifest = json.load(f)

        if "features" in manifest:

            features = manifest["features"]

        elif "feature_names" in manifest:

            features = manifest["feature_names"]

        else:

            raise ValueError(
                "Could not find feature list in "
                "600 MW model manifest."
            )

        return [str(x) for x in features]

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def validate_source_data(self, data):

        if not isinstance(data, pd.DataFrame):

            raise TypeError(
                "Runtime input must be a pandas DataFrame."
            )

        if data.empty:

            raise ValueError(
                "Runtime input contains no rows."
            )

        if "timestamp" not in data.columns:

            raise ValueError(
                "Runtime input must contain "
                "'timestamp'."
            )

        # Timestamp validation

        timestamps = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

        if timestamps.isna().any():

            raise ValueError(
                "Runtime input contains invalid timestamps."
            )

        # Numeric validation

        metadata_columns = {
            "timestamp",
            "replay_source",
            "replay_step",
        }

        numeric_columns = [
            column
            for column in data.columns
            if column not in metadata_columns
        ]

        if numeric_columns:

            numeric_block = data[
                numeric_columns
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            if numeric_block.isna().any().any():

                raise ValueError(
                    "Runtime input contains "
                    "non-numeric values."
                )

            values = numeric_block.to_numpy(
                dtype=float
            )

            if not np.isfinite(values).all():

                raise ValueError(
                    "Runtime input contains "
                    "NaN or infinite values."
                )

        return True

    # ========================================================
    # SCHEMA VALIDATION
    # ========================================================

    def validate_model_features(self, data):

        missing = [
            feature
            for feature in self.features
            if feature not in data.columns
        ]

        extra = [
            column
            for column in data.columns
            if column != "timestamp"
            and column not in self.features
        ]

        if missing:

            return {
                "status": "INCOMPLETE",
                "missing_features": missing,
                "extra_features": extra,
                "feature_count": len(
                    data.columns
                ) - 1
            }

        return {
            "status": "READY",
            "missing_features": [],
            "extra_features": extra,
            "feature_count": len(
                self.features
            )
        }

    # ========================================================
    # PREPARE MODEL INPUT
    # ========================================================

    def prepare(self, data):

        self.validate_source_data(data)

        validation = (
            self.validate_model_features(data)
        )

        if validation["status"] != "READY":

            missing = validation[
                "missing_features"
            ]

            raise RuntimeError(
                "600 MW runtime feature preparation "
                "cannot continue.\n\n"
                f"Missing model features: "
                f"{len(missing)}\n"
                f"Expected: {len(self.features)}\n\n"
                "A runtime SCADA-to-model mapping "
                "must be established before inference."
            )

        prepared = data[
            ["timestamp"] + self.features
        ].copy()

        # --------------------------------------------------------
        # Convert all model features to actual numeric dtypes.
        #
        # validate_source_data() confirms that conversion is
        # possible, while this step actually performs it.
        # --------------------------------------------------------

        prepared[self.features] = (
            prepared[self.features]
            .apply(
                pd.to_numeric,
                errors="raise"
            )
        )

        # --------------------------------------------------------
        # Final numeric validation
        # --------------------------------------------------------

        if not all(
            pd.api.types.is_numeric_dtype(
                prepared[feature]
            )
            for feature in self.features
        ):
            raise TypeError(
                "Runtime model features are not numeric "
                "after conversion."
            )

        return prepared


# ============================================================
# SELF TEST
# ============================================================

def main():

    print("=" * 70)
    print("600 MW RUNTIME FEATURE PREPARATION")
    print("=" * 70)

    builder = RuntimeFeaturePreparation600MW(
        MANIFEST_PATH
    )

    print("\nExpected model features:")
    print(
        len(builder.features)
    )

    # --------------------------------------------------------
    # Deliberately use a small simulated SCADA input.
    #
    # This is NOT intended to satisfy the 119-feature model.
    # It verifies that the layer correctly detects an
    # incomplete runtime mapping.
    # --------------------------------------------------------

    simulated_scada = pd.DataFrame({

        "timestamp": [
            "2026-09-30T10:00:00",
            "2026-09-30T10:00:02",
        ],

        "power_output": [
            346.75,
            347.10,
        ],

        "voltage": [
            415.0,
            415.2,
        ],

        "current": [
            520.0,
            521.0,
        ],

        "frequency": [
            50.0,
            50.0,
        ],

        "steam_pressure": [
            16.2,
            16.25,
        ],

        "steam_temperature": [
            540.0,
            540.3,
        ],

        "turbine_speed": [
            3000.0,
            3001.0,
        ],

        "vibration": [
            2.3,
            2.31,
        ],
    })

    print("\nSimulated SCADA rows:")
    print(
        len(simulated_scada)
    )

    builder.validate_source_data(
        simulated_scada
    )

    print(
        "Source validation: PASS"
    )

    validation = (
        builder.validate_model_features(
            simulated_scada
        )
    )

    print("\nMODEL SCHEMA VALIDATION")

    print(
        f"Status           : "
        f"{validation['status']}"
    )

    print(
        f"Expected features: "
        f"{len(builder.features)}"
    )

    print(
        f"Missing features  : "
        f"{len(validation['missing_features'])}"
    )

    # --------------------------------------------------------
    # Expected result:
    # INCOMPLETE
    #
    # This is intentional.
    # --------------------------------------------------------

    if validation["status"] == "INCOMPLETE":

        print(
            "\nEXPECTED RESULT:"
        )

        print(
            "Runtime feature mapping is not complete."
        )

        print(
            "The builder correctly prevents "
            "invalid inference."
        )

    else:

        raise RuntimeError(
            "Unexpected result. "
            "The simulated SCADA input should "
            "not satisfy the 119-feature model."
        )

    print("\n" + "=" * 70)
    print(
        "600 MW RUNTIME FEATURE PREPARATION: "
        "INTERFACE VALIDATION PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()