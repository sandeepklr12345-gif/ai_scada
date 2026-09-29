"""
AI MODEL INTEGRATION BATCH VALIDATION

Purpose:
    Validate that the unified AI integration layer preserves the
    behavior of both already-frozen model components.

Validation:
    1. Load 600 MW forecasting engine.
    2. Load HAI Candidate C inference engine.
    3. Run a multi-row 600 MW batch.
    4. Run the same 600 MW batch directly through the standalone
       forecasting engine and through the integration layer.
    5. Run a multi-row HAI sequence through the standalone engine.
    6. Run the same HAI sequence through the integration layer.
    7. Compare predictions and scores.
    8. Confirm HAI history initialization behaves as expected.

This script does NOT modify models or datasets.
It does NOT connect to MQTT, SCADA, PostgreSQL, ESP32, FastAPI,
or React.
"""

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INTEGRATION_PATH = (
    ROOT /
    "scripts" /
    "integration" /
    "ai_model_integration.py"
)

FORECAST_ENGINE_PATH = (
    ROOT /
    "scripts" /
    "inference" /
    "forecasting_600mw.py"
)

HAI_ENGINE_PATH = (
    ROOT /
    "scripts" /
    "inference" /
    "hai_candidate_C_inference.py"
)

FORECAST_MODEL_DIR = (
    ROOT /
    "models" /
    "forecasting" /
    "600mw"
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

FORECAST_SOURCE = (
    ROOT /
    "data" /
    "features" /
    "600mw" /
    "candidates" /
    "feature_set_D_full_candidate_pool.csv"
)

HAI_SOURCE = (
    ROOT /
    "data" /
    "features" /
    "hai" /
    "hai-23.05" /
    "model_ready" /
    "hai-test1_model_ready.csv"
)

BATCH_SIZE = 100


# ============================================================
# MODULE LOADER
# ============================================================

def load_module(
    name,
    path
):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Required module not found:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not load module:\n{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# HELPERS
# ============================================================

def compare_numeric_columns(
    left,
    right,
    columns,
    tolerance=1e-12
):

    max_difference = 0.0
    mismatch_count = 0

    for column in columns:

        a = pd.to_numeric(
            left[column],
            errors="coerce"
        ).to_numpy()

        b = pd.to_numeric(
            right[column],
            errors="coerce"
        ).to_numpy()

        if len(a) != len(b):

            raise RuntimeError(
                f"Row count mismatch for {column}: "
                f"{len(a)} vs {len(b)}"
            )

        both_nan = (
            np.isnan(a) &
            np.isnan(b)
        )

        valid = ~both_nan

        if valid.any():

            difference = np.abs(
                a[valid] -
                b[valid]
            )

            max_difference = max(
                max_difference,
                float(difference.max())
            )

            mismatch_count += int(
                np.sum(
                    difference > tolerance
                )
            )

        mismatch_count += int(
            np.sum(
                np.isnan(a) !=
                np.isnan(b)
            )
        )

    return (
        max_difference,
        mismatch_count
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("")
    print("=" * 80)
    print("AI MODEL INTEGRATION BATCH VALIDATION")
    print("=" * 80)
    print("")

    # --------------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------------

    required_paths = [
        INTEGRATION_PATH,
        FORECAST_ENGINE_PATH,
        HAI_ENGINE_PATH,
        FORECAST_SOURCE,
        HAI_SOURCE,
        HAI_MODEL_PATH,
        HAI_MANIFEST_PATH
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required integration artifact not found:\n{path}"
            )

    # --------------------------------------------------------
    # LOAD MODULES
    # --------------------------------------------------------

    integration_module = load_module(
        "ai_model_integration",
        INTEGRATION_PATH
    )

    forecast_module = load_module(
        "forecasting_600mw_direct",
        FORECAST_ENGINE_PATH
    )

    hai_module = load_module(
        "hai_candidate_C_direct",
        HAI_ENGINE_PATH
    )

    # --------------------------------------------------------
    # LOAD INTEGRATED SYSTEM
    # --------------------------------------------------------

    integrated = (
        integration_module.AIModelIntegration()
    )

    # --------------------------------------------------------
    # LOAD STANDALONE ENGINES
    # --------------------------------------------------------

    direct_forecaster = (
        forecast_module.Forecasting600MW(
            FORECAST_MODEL_DIR
        )
    )

    direct_hai = (
        hai_module.HAICandidateCInference(
            str(HAI_MODEL_PATH),
            str(HAI_MANIFEST_PATH)
        )
    )

    # ========================================================
    # 600 MW BATCH VALIDATION
    # ========================================================

    print("")
    print("=" * 80)
    print("600 MW BATCH VALIDATION")
    print("=" * 80)
    print("")

    forecast_features = (
        integrated.forecaster.features
    )

    forecast_df = pd.read_csv(
        FORECAST_SOURCE,
        usecols=forecast_features
    )

    forecast_df = (
        forecast_df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .head(BATCH_SIZE)
        .reset_index(drop=True)
    )

    if len(forecast_df) != BATCH_SIZE:

        raise RuntimeError(
            "Could not obtain the required "
            f"{BATCH_SIZE} valid 600 MW rows."
        )

    print(
        f"Rows tested: {len(forecast_df)}"
    )

    # Direct prediction
    direct_forecast = (
        direct_forecaster.predict(
            forecast_df
        )
        .reset_index(drop=True)
    )

    # Integrated prediction
    integrated_forecast = (
        integrated.predict_forecast(
            forecast_df
        )
        .reset_index(drop=True)
    )

    forecast_columns = [
        "power_2min",
        "power_10min",
        "power_30min"
    ]

    (
        forecast_max_diff,
        forecast_mismatches
    ) = compare_numeric_columns(
        direct_forecast,
        integrated_forecast,
        forecast_columns
    )

    print(
        f"Maximum forecast difference : "
        f"{forecast_max_diff:.15f}"
    )

    print(
        f"Forecast mismatches          : "
        f"{forecast_mismatches}"
    )

    if forecast_mismatches != 0:

        raise RuntimeError(
            "600 MW integration changed "
            "forecast predictions."
        )

    print(
        "600 MW integration check    : PASS"
    )

    # ========================================================
    # HAI BATCH VALIDATION
    # ========================================================

    print("")
    print("=" * 80)
    print("HAI CANDIDATE C BATCH VALIDATION")
    print("=" * 80)
    print("")

    # --------------------------------------------------------
    # Read the authoritative 58 original features from the
    # production HAI engine.
    # --------------------------------------------------------

    hai_features = (
        integrated.hai_detector.original_features
    )

    hai_full = pd.read_csv(
        HAI_SOURCE
    )

    missing_features = [
        feature
        for feature in hai_features
        if feature not in hai_full.columns
    ]

    if missing_features:

        raise RuntimeError(
            "HAI source is missing required original "
            "features:\n"
            + "\n".join(
                missing_features
            )
        )

    hai_df = (
        hai_full[
            hai_features
        ]
        .head(BATCH_SIZE)
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"Rows tested: {len(hai_df)}"
    )

    print(
        f"Input features: {len(hai_features)}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use separate engine instances so both start with empty
    # temporal history. This makes the comparison fair.
    # --------------------------------------------------------

    direct_hai = (
        hai_module.HAICandidateCInference(
            str(HAI_MODEL_PATH),
            str(HAI_MANIFEST_PATH)
        )
    )

    integrated_hai = (
        hai_module.HAICandidateCInference(
            str(HAI_MODEL_PATH),
            str(HAI_MANIFEST_PATH)
        )
    )

    # Direct standalone batch
    direct_hai_result = (
        direct_hai.predict_batch(
            hai_df.copy()
        )
        .reset_index(drop=True)
    )

    # Integration-layer batch
    integrated_hai_result = (
        integrated.predict_anomaly(
            hai_df.copy()
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Compare common output columns
    # --------------------------------------------------------

    hai_output_columns = [
        column
        for column in direct_hai_result.columns
        if column in integrated_hai_result.columns
    ]

    print("")
    print(
        "HAI output columns:"
    )

    for column in hai_output_columns:

        print(
            f"  {column}"
        )

    (
        hai_max_diff,
        hai_mismatches
    ) = compare_numeric_columns(
        direct_hai_result,
        integrated_hai_result,
        hai_output_columns
    )

    print("")
    print(
        f"Maximum HAI output difference : "
        f"{hai_max_diff:.15f}"
    )

    print(
        f"HAI output mismatches         : "
        f"{hai_mismatches}"
    )

    if hai_mismatches != 0:

        raise RuntimeError(
            "HAI integration changed "
            "batch inference results."
        )

    print(
        "HAI integration check       : PASS"
    )

    # ========================================================
    # HAI HISTORY VALIDATION
    # ========================================================

    print("")
    print("=" * 80)
    print("HAI TEMPORAL HISTORY VALIDATION")
    print("=" * 80)
    print("")

    if "status" not in integrated_hai_result.columns:

        raise RuntimeError(
            "HAI output does not contain "
            "the expected status column."
        )

    insufficient_history_count = int(
        (
            integrated_hai_result["status"]
            == "INSUFFICIENT_HISTORY"
        ).sum()
    )

    valid_count = int(
        len(integrated_hai_result)
        -
        insufficient_history_count
    )

    print(
        f"Rows tested              : "
        f"{len(integrated_hai_result)}"
    )

    print(
        f"Insufficient-history rows : "
        f"{insufficient_history_count}"
    )

    print(
        f"Rows with inference      : "
        f"{valid_count}"
    )

    # The Candidate C engine has HISTORY_SIZE = 4.
    # Therefore the first 4 rows require history and do not
    # receive a model prediction.
    expected_initial_history = (
        direct_hai.HISTORY_SIZE
    )

    if insufficient_history_count != (
        expected_initial_history
    ):

        raise RuntimeError(
            "Unexpected HAI history initialization behavior.\n"
            f"Expected: {expected_initial_history}\n"
            f"Actual:   {insufficient_history_count}"
        )

    print(
        "Temporal history check   : PASS"
    )

    # --------------------------------------------------------
    # Validate actual prediction rows
    # --------------------------------------------------------

    valid_rows = (
        integrated_hai_result[
            integrated_hai_result["status"]
            != "INSUFFICIENT_HISTORY"
        ]
    )

    if valid_rows.empty:

        raise RuntimeError(
            "No HAI rows reached actual model inference."
        )

    if not valid_rows[
        "prediction"
    ].isin([0, 1]).all():

        raise RuntimeError(
            "HAI prediction contains values "
            "other than 0 or 1."
        )

    print(
        f"Valid HAI predictions      : "
        f"{len(valid_rows)}"
    )

    print(
        f"Predicted NORMAL            : "
        f"{int((valid_rows['prediction'] == 0).sum())}"
    )

    print(
        f"Predicted ANOMALY           : "
        f"{int((valid_rows['prediction'] == 1).sum())}"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("")
    print("=" * 80)
    print("INTEGRATION BATCH VALIDATION: PASS")
    print("=" * 80)

    print("")
    print("Verified:")
    print(
        "  600 MW standalone vs integrated : PASS"
    )
    print(
        "  600 MW prediction preservation  : PASS"
    )
    print(
        "  HAI standalone vs integrated    : PASS"
    )
    print(
        "  HAI temporal history            : PASS"
    )
    print(
        "  HAI actual model predictions    : PASS"
    )

    print("")
    print(
        "STATUS: MODEL INTEGRATION VALIDATED"
    )


if __name__ == "__main__":
    main()
