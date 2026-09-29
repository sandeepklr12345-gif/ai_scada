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
    / "600mw_runtime_forecast_results.csv"
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

    print("=" * 70)
    print("600 MW RUNTIME → AI FORECAST VALIDATION")
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
    # Validate and prepare runtime data
    # --------------------------------------------------------

    print(
        "\nPreparing runtime model inputs..."
    )

    prepared = runtime_builder.prepare(
        replay
    )

    print(
        f"Prepared rows    : {len(prepared)}"
    )

    print(
        f"Prepared features: "
        f"{len(runtime_builder.features)}"
    )

    # --------------------------------------------------------
    # Load frozen forecasting engine
    # --------------------------------------------------------

    forecasting_module = load_module(
        FORECASTING_SCRIPT,
        "forecasting_600mw"
    )

    # --------------------------------------------------------
    # Detect forecasting class
    # --------------------------------------------------------

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
    # Run inference
    # --------------------------------------------------------

    print(
        "\nRunning chronological inference..."
    )

    results = []

    feature_columns = (
        runtime_builder.features
    )

    for index, row in prepared.iterrows():

        model_input = row[
            feature_columns
        ].to_frame().T

        # --------------------------------------------------------
        # Ensure model input is explicitly numeric
        # --------------------------------------------------------

        model_input = model_input.copy()

        model_input[feature_columns] = (
            model_input[feature_columns]
            .apply(
                pd.to_numeric,
                errors="raise"
            )
        )

        # Diagnostic for first row
        if index == 0:

            print("\nMODEL INPUT DTYPE CHECK")

            non_numeric = [
                column
                for column in feature_columns
                if not pd.api.types.is_numeric_dtype(
                    model_input[column]
                )
            ]

            print(
                f"Total features : {len(feature_columns)}"
            )

            print(
                f"Non-numeric    : {len(non_numeric)}"
            )

            if non_numeric:

                print(
                    "Non-numeric features:"
                )

                for column in non_numeric:
                    print(
                        f"  {column}: "
                        f"{model_input[column].dtype}"
                    )

            print(
                "\nFirst 5 dtypes:"
            )

            print(
                model_input[
                    feature_columns[:5]
                ].dtypes
            )

        prediction = forecaster.predict(
            model_input
        )

        result_row = {
            "timestamp":
                row["timestamp"],

            "replay_step":
                replay.iloc[index][
                    "replay_step"
                ],

            "actual_power":
                float(
                    row[
                        "Power output\n（MW）"
                    ]
                ),

            "forecast_2min":
                float(
                    prediction[
                        "power_2min"
                    ].iloc[0]
                ),

            "forecast_10min":
                float(
                    prediction[
                        "power_10min"
                    ].iloc[0]
                ),

            "forecast_30min":
                float(
                    prediction[
                        "power_30min"
                    ].iloc[0]
                ),
        }

        results.append(
            result_row
        )

        if (
            (index + 1) % 500 == 0
            or index == 0
            or index == len(prepared) - 1
        ):

            print(
                f"Processed "
                f"{index + 1}/{len(prepared)}"
            )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Validate predictions
    # --------------------------------------------------------

    prediction_columns = [
        "forecast_2min",
        "forecast_10min",
        "forecast_30min"
    ]

    numeric_values = results_df[
        prediction_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise RuntimeError(
            "Inference produced NaN or "
            "infinite prediction values."
        )

    # --------------------------------------------------------
    # Basic forecast statistics
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

    # --------------------------------------------------------
    # Show samples
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

    print(
        f"\nOutput saved:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print(
        "600 MW RUNTIME → AI FORECAST: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()