from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "600mw"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "600mw"
)

INPUT_FILE = (
    FEATURE_DIR
    / "candidates"
    / "feature_set_D_full_candidate_pool.csv"
)

MANIFEST_FILE = (
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
    / "600mw_scada_replay.csv"
)


# ============================================================
# LOAD FINAL MODEL FEATURES
# ============================================================

def load_model_features():

    with open(
        MANIFEST_FILE,
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("600 MW HISTORICAL SCADA REPLAY SIMULATOR")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load authoritative model schema
    # --------------------------------------------------------

    model_features = load_model_features()

    print(
        f"\nFinal model features : "
        f"{len(model_features)}"
    )

    # --------------------------------------------------------
    # Load historical candidate dataset
    # --------------------------------------------------------

    print(
        "\nLoading 600 MW historical data..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Historical rows      : {len(df)}"
    )

    print(
        f"Historical columns   : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Locate timestamp
    # --------------------------------------------------------

    timestamp_column = None

    for candidate in [
        "Time",
        "timestamp",
        "Timestamp"
    ]:

        if candidate in df.columns:

            timestamp_column = candidate
            break

    if timestamp_column is None:

        raise ValueError(
            "No timestamp column found."
        )

    # --------------------------------------------------------
    # Validate model feature availability
    # --------------------------------------------------------

    missing = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing:

        print(
            "\nMissing model features:"
        )

        for feature in missing:
            print(
                f"  {feature}"
            )

        raise ValueError(
            f"{len(missing)} model features "
            "are missing from the replay source."
        )

    print(
        "\nAll 119 model features found."
    )

    # --------------------------------------------------------
    # Build replay dataset
    # --------------------------------------------------------

    replay = df[
        [timestamp_column] + model_features
    ].copy()

    replay = replay.rename(
        columns={
            timestamp_column: "timestamp"
        }
    )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    replay["timestamp"] = pd.to_datetime(
        replay["timestamp"],
        errors="coerce"
    )

    if replay["timestamp"].isna().any():

        raise ValueError(
            "Replay dataset contains invalid timestamps."
        )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in replay.columns
        if column != "timestamp"
    ]

    replay[numeric_columns] = (
        replay[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    if replay[numeric_columns].isna().any().any():

        raise ValueError(
            "Replay dataset contains missing/non-numeric "
            "model input values."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    replay = replay.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Validate duplicate timestamps
    # --------------------------------------------------------

    duplicate_timestamps = (
        replay["timestamp"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate timestamps : "
        f"{duplicate_timestamps}"
    )

    if duplicate_timestamps:

        raise ValueError(
            "Replay source contains duplicate timestamps."
        )

    # --------------------------------------------------------
    # Validate sampling interval
    # --------------------------------------------------------

    intervals = (
        replay["timestamp"]
        .diff()
        .dropna()
    )

    if not intervals.empty:

        print(
            "\nSampling interval distribution:"
        )

        print(
            intervals
            .value_counts()
            .sort_index()
            .to_string()
        )

    # --------------------------------------------------------
    # Add replay metadata
    # --------------------------------------------------------

    replay.insert(
        1,
        "replay_source",
        "600MW_HISTORICAL_REPLAY"
    )

    replay.insert(
        2,
        "replay_step",
        range(1, len(replay) + 1)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    replay.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\nREPLAY DATASET")

    print(
        f"Rows                : {len(replay)}"
    )

    print(
        f"Columns             : {len(replay.columns)}"
    )

    print(
        f"Model input columns : {len(model_features)}"
    )

    print(
        f"First timestamp     : "
        f"{replay['timestamp'].iloc[0]}"
    )

    print(
        f"Last timestamp      : "
        f"{replay['timestamp'].iloc[-1]}"
    )

    print(
        f"\nOutput saved:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print(
        "600 MW SCADA REPLAY SIMULATOR: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()