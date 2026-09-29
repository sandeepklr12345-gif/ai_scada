from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPLAY_FILE = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_scada_replay.csv"
)

RUNTIME_FEATURE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "integration"
    / "runtime_feature_preparation_600mw.py"
)

FORECASTING_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "inference"
    / "forecasting_600mw.py"
)

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "600mw_runtime_forecast_results_optimized.csv"
)


# ============================================================
# MODULE LOADER
# ============================================================

def load_module(path, module_name):

    spec = importlib.util.spec_from_file_location(
        module_name,
        path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module:\n{path}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("600 MW BATCH-OPTIMIZED FORECAST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load replay
    # --------------------------------------------------------

    replay = pd.read_csv(
        REPLAY_FILE
    )

    print(
        f"\nReplay rows : {len(replay)}"
    )

    # --------------------------------------------------------
    # Load runtime feature preparation
    # --------------------------------------------------------

    runtime_module = load_module(
        RUNTIME_FEATURE_SCRIPT,
        "runtime_feature_preparation_600mw"
    )

    RuntimeFeaturePreparation600MW = (
        runtime_module.RuntimeFeaturePreparation600MW
    )

    runtime_builder = (
        RuntimeFeaturePreparation600MW(
            MANIFEST_PATH
        )
    )

    # --------------------------------------------------------
    # Prepare runtime data
    # --------------------------------------------------------

    print(
        "\nPreparing runtime model inputs..."
    )

    prepared = runtime_builder.prepare(
        replay
    )

    feature_columns = (
        runtime_builder.features
    )

    print(
        f"Prepared rows    : {len(prepared)}"
    )

    print(
        f"Prepared features: {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Validate ONCE
    # --------------------------------------------------------

    print(
        "\nValidating complete model input..."
    )

    model_input = prepared[
        feature_columns
    ].copy()

    # Numeric conversion ONCE
    model_input = model_input.apply(
        pd.to_numeric,
        errors="raise"
    )

    # Numeric check
    if not all(
        pd.api.types.is_numeric_dtype(
            model_input[column]
        )
        for column in feature_columns
    ):
        raise TypeError(
            "Non-numeric feature detected."
        )

    # Missing-value check
    if model_input.isna().any().any():

        raise ValueError(
            "Model input contains missing values."
        )

    # Infinite-value check
    numeric_values = model_input.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            "Model input contains "
            "NaN or infinite values."
        )

    print(
        "Validation: PASS"
    )

    # --------------------------------------------------------
    # Load forecasting engine
    # --------------------------------------------------------

    forecasting_module = load_module(
        FORECASTING_SCRIPT,
        "forecasting_600mw"
    )

    ForecastingClass = getattr(
        forecasting_module,
        "Forecasting600MW"
    )

    forecaster = ForecastingClass(
        model_dir=MODEL_DIR
    )

    print(
        "Forecasting engine: LOADED"
    )

    # --------------------------------------------------------
    # BATCH MODEL PREDICTION
    # --------------------------------------------------------

    print(
        "\nRunning BATCH inference..."
    )

    predictions = {}

    predictions[
        "forecast_2min"
    ] = forecaster.models[
        "target_power_2min"
    ].predict(
        model_input
    )

    predictions[
        "forecast_10min"
    ] = forecaster.models[
        "target_power_10min"
    ].predict(
        model_input
    )

    predictions[
        "forecast_30min"
    ] = forecaster.models[
        "target_power_30min"
    ].predict(
        model_input
    )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    results_df = pd.DataFrame({

        "timestamp":
            prepared["timestamp"].values,

        "replay_step":
            replay["replay_step"].values,

        "actual_power":
            prepared[
                "Power output\n（MW）"
            ].astype(float).values,

        "forecast_2min":
            np.asarray(
                predictions["forecast_2min"],
                dtype=float
            ),

        "forecast_10min":
            np.asarray(
                predictions["forecast_10min"],
                dtype=float
            ),

        "forecast_30min":
            np.asarray(
                predictions["forecast_30min"],
                dtype=float
            ),
    })

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    prediction_columns = [
        "forecast_2min",
        "forecast_10min",
        "forecast_30min"
    ]

    if not np.isfinite(
        results_df[
            prediction_columns
        ].to_numpy()
    ).all():

        raise RuntimeError(
            "Batch inference produced "
            "NaN or infinite values."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nOutput saved:\n{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nFORECAST SUMMARY")

    for column in prediction_columns:

        print(
            f"\n{column}"
        )

        print(
            f"  Mean : "
            f"{results_df[column].mean():.4f} MW"
        )

        print(
            f"  Min  : "
            f"{results_df[column].min():.4f} MW"
        )

        print(
            f"  Max  : "
            f"{results_df[column].max():.4f} MW"
        )

    print("\nFIRST 5 RESULTS")

    print(
        results_df.head(5).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("600 MW BATCH FORECAST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()