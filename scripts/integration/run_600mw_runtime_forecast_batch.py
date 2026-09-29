from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np
import time


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
    / "600mw_runtime_forecast_results_batch.csv"
)


# ============================================================
# DYNAMIC MODULE LOADER
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

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.perf_counter()

    print("=" * 70)
    print("600 MW BATCH RUNTIME → AI FORECAST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load replay
    # --------------------------------------------------------

    if not REPLAY_FILE.exists():

        raise FileNotFoundError(
            f"Replay file not found:\n{REPLAY_FILE}"
        )

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
        "runtime_feature_preparation_600mw_batch"
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

    if len(feature_columns) != 119:

        raise RuntimeError(
            f"Expected 119 runtime features, "
            f"got {len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Build complete model input matrix
    # --------------------------------------------------------

    print(
        "\nBuilding batch model input..."
    )

    model_input = prepared[
        feature_columns
    ].copy()

    # Convert all features to numeric once
    model_input = model_input.apply(
        pd.to_numeric,
        errors="raise"
    )

    # --------------------------------------------------------
    # Validate model input
    # --------------------------------------------------------

    print(
        "\nMODEL INPUT VALIDATION"
    )

    print(
        f"Total features : {len(feature_columns)}"
    )

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            model_input[column]
        )
    ]

    print(
        f"Non-numeric    : {len(non_numeric)}"
    )

    if non_numeric:

        print(
            "\nNon-numeric features:"
        )

        for column in non_numeric:

            print(
                f"  {column}: "
                f"{model_input[column].dtype}"
            )

        raise RuntimeError(
            "Runtime model input contains "
            "non-numeric features."
        )

    numeric_values = model_input.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise RuntimeError(
            "Runtime model input contains "
            "NaN or infinite values."
        )

    print(
        "Input validation: PASS"
    )

    # --------------------------------------------------------
    # Load forecasting engine
    # --------------------------------------------------------

    forecasting_module = load_module(
        FORECASTING_SCRIPT,
        "forecasting_600mw_batch"
    )

    ForecastingClass = getattr(
        forecasting_module,
        "Forecasting600MW",
        None
    )

    if ForecastingClass is None:

        raise AttributeError(
            "Forecasting600MW class was not found "
            "in forecasting_600mw.py"
        )

    forecaster = ForecastingClass(
        model_dir=MODEL_DIR
    )

    print(
        "600 MW forecasting engine: LOADED"
    )

    # --------------------------------------------------------
    # Batch inference
    # --------------------------------------------------------

    print(
        "\nRunning BATCH inference..."
    )

    inference_start = time.perf_counter()

    predictions = {}

    predictions["forecast_2min"] = (
        forecaster.models[
            "target_power_2min"
        ].predict(
            model_input
        )
    )

    predictions["forecast_10min"] = (
        forecaster.models[
            "target_power_10min"
        ].predict(
            model_input
        )
    )

    predictions["forecast_30min"] = (
        forecaster.models[
            "target_power_30min"
        ].predict(
            model_input
        )
    )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # --------------------------------------------------------
    # Construct output
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
            predictions[
                "forecast_2min"
            ],

        "forecast_10min":
            predictions[
                "forecast_10min"
            ],

        "forecast_30min":
            predictions[
                "forecast_30min"
            ],
    })

    # --------------------------------------------------------
    # Validate predictions
    # --------------------------------------------------------

    prediction_columns = [
        "forecast_2min",
        "forecast_10min",
        "forecast_30min"
    ]

    numeric_predictions = results_df[
        prediction_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_predictions
    ).all():

        raise RuntimeError(
            "Inference produced NaN or "
            "infinite prediction values."
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\nFORECAST SUMMARY"
    )

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

    total_time = (
        time.perf_counter()
        - start_time
    )

    rows_per_second = (
        len(results_df)
        / total_time
    )

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    print(
        "\nFIRST 5 RESULTS"
    )

    print(
        results_df.head(5).to_string(
            index=False
        )
    )

    print(
        "\nLAST 5 RESULTS"
    )

    print(
        results_df.tail(5).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    print(
        "\nPERFORMANCE"
    )

    print(
        f"  Total runtime : "
        f"{total_time:.4f} sec"
    )

    print(
        f"  Inference time: "
        f"{inference_time:.4f} sec"
    )

    print(
        f"  Rows/sec      : "
        f"{rows_per_second:.2f}"
    )

    print(
        f"  Time/row      : "
        f"{(total_time / len(results_df)) * 1000:.4f} ms"
    )

    print(
        f"\nOutput saved:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print(
        "600 MW BATCH RUNTIME → AI FORECAST: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()