from pathlib import Path
import sys
import pandas as pd


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

RUNTIME_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "integration"
    / "runtime_feature_preparation_600mw.py"
)


# ============================================================
# LOAD RUNTIME FEATURE PREPARATION
# ============================================================

def load_runtime_builder():

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "runtime_feature_preparation_600mw",
        RUNTIME_SCRIPT
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module.RuntimeFeaturePreparation600MW


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("600 MW RUNTIME REPLAY VALIDATION")
    print("=" * 70)

    if not REPLAY_FILE.exists():

        raise FileNotFoundError(
            f"Replay file not found:\n{REPLAY_FILE}"
        )

    # --------------------------------------------------------
    # Load replay
    # --------------------------------------------------------

    replay = pd.read_csv(
        REPLAY_FILE
    )

    print(
        f"\nReplay rows    : {len(replay)}"
    )

    print(
        f"Replay columns : {len(replay.columns)}"
    )

    # --------------------------------------------------------
    # Load runtime builder
    # --------------------------------------------------------

    RuntimeFeaturePreparation600MW = (
        load_runtime_builder()
    )

    builder = RuntimeFeaturePreparation600MW(
        builder_manifest_path
        if False else (
            PROJECT_ROOT
            / "models"
            / "forecasting"
            / "600mw"
            / "600mw_forecasting_feature_manifest.json"
        )
    )

    # --------------------------------------------------------
    # Validate source
    # --------------------------------------------------------

    print(
        "\nSOURCE VALIDATION"
    )

    builder.validate_source_data(
        replay
    )

    print(
        "Source validation : PASS"
    )

    # --------------------------------------------------------
    # Validate model schema
    # --------------------------------------------------------

    print(
        "\nMODEL SCHEMA VALIDATION"
    )

    validation = (
        builder.validate_model_features(
            replay
        )
    )

    print(
        f"Status             : "
        f"{validation['status']}"
    )

    print(
        f"Expected features  : "
        f"{len(builder.features)}"
    )

    print(
        f"Missing features   : "
        f"{len(validation['missing_features'])}"
    )

    print(
        f"Extra columns      : "
        f"{len(validation['extra_features'])}"
    )

    if validation["missing_features"]:

        print(
            "\nMissing features:"
        )

        for feature in validation[
            "missing_features"
        ]:

            print(
                f"  {feature}"
            )

        raise RuntimeError(
            "Replay does not satisfy "
            "the 600 MW model schema."
        )

    if validation["status"] != "READY":

        raise RuntimeError(
            "Runtime schema validation failed."
        )

    print(
        "Model schema validation : PASS"
    )

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    print(
        "\nPREPARING MODEL INPUT"
    )

    prepared = builder.prepare(
        replay
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_columns = [
        "timestamp"
    ] + builder.features

    actual_columns = list(
        prepared.columns
    )

    if actual_columns != expected_columns:

        raise RuntimeError(
            "Prepared feature order does not "
            "match the authoritative model schema."
        )

    if prepared.isna().any().any():

        raise RuntimeError(
            "Prepared model input contains "
            "missing values."
        )

    numeric_features = prepared[
        builder.features
    ]

    if not numeric_features.apply(
        lambda column: pd.api.types.is_numeric_dtype(
            column
        )
    ).all():

        raise RuntimeError(
            "One or more model features "
            "are not numeric."
        )

    print(
        "Feature order validation : PASS"
    )

    print(
        "Missing-value validation : PASS"
    )

    print(
        "Numeric-feature validation: PASS"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\nRUNTIME FEATURE OUTPUT"
    )

    print(
        f"Rows                : "
        f"{len(prepared)}"
    )

    print(
        f"Model features      : "
        f"{len(builder.features)}"
    )

    print(
        f"Output columns      : "
        f"{len(prepared.columns)}"
    )

    print(
        f"First timestamp     : "
        f"{prepared['timestamp'].iloc[0]}"
    )

    print(
        f"Last timestamp      : "
        f"{prepared['timestamp'].iloc[-1]}"
    )

    print(
        "\nFirst prepared row:"
    )

    print(
        prepared.iloc[0][
            builder.features[:10]
        ].to_string()
    )

    print("\n" + "=" * 70)
    print(
        "600 MW RUNTIME REPLAY VALIDATION: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()