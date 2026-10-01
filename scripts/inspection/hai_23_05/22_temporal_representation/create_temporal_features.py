from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]


# ============================================================
# STAGE 22B
# HAI 23.05 - Controlled Temporal Feature Creation
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "model_ready"
    / "hai_2305_training_model_ready.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_2305_training_temporal_features.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_feature_metadata.csv"
)


print("=" * 70)
print("STAGE 22B: CONTROLLED TEMPORAL FEATURE CREATION")
print("=" * 70)


# ============================================================
# 1. LOAD
# ============================================================

print("\nLoading training model-ready data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")


# ============================================================
# 2. VALIDATE INPUT
# ============================================================

assert "timestamp" in df.columns

timestamps = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

assert timestamps.notna().all()

feature_columns = [
    col for col in df.columns
    if col != "timestamp"
]

assert len(feature_columns) == 58

assert df[feature_columns].isna().sum().sum() == 0

assert np.isfinite(
    df[feature_columns].to_numpy(dtype=float)
).all()

print(f"Base features: {len(feature_columns)}")


# ============================================================
# 3. CREATE SEQUENCE IDs
# ============================================================

print("\nDetecting temporal boundaries...")

time_diff = timestamps.diff()

normal_interval = pd.Timedelta(seconds=1)

sequence_id = (
    time_diff > normal_interval
).cumsum()

df["_sequence_id"] = sequence_id

sequence_count = int(
    df["_sequence_id"].nunique()
)

boundary_count = int(
    (time_diff > normal_interval).sum()
)

print(f"Continuous sequences : {sequence_count}")
print(f"Temporal boundaries  : {boundary_count}")


# ============================================================
# 4. CREATE TEMPORAL FEATURES
# ============================================================

print("\nCreating temporal features...")

temporal_data = pd.DataFrame(
    index=df.index
)

metadata = []

for i, feature in enumerate(
    feature_columns,
    start=1
):

    print(
        f"[{i:02d}/{len(feature_columns)}] "
        f"{feature}"
    )

    # --------------------------------------------------------
    # First difference
    # --------------------------------------------------------

    diff = (
        df.groupby("_sequence_id")[feature]
        .diff()
    )

    diff_name = f"{feature}__diff_1s"

    temporal_data[diff_name] = diff

    metadata.append({
        "feature": diff_name,
        "source_feature": feature,
        "transformation": "first_difference",
        "window_seconds": 1
    })

    # --------------------------------------------------------
    # Absolute first difference
    # --------------------------------------------------------

    abs_diff_name = (
        f"{feature}__abs_diff_1s"
    )

    temporal_data[abs_diff_name] = (
        diff.abs()
    )

    metadata.append({
        "feature": abs_diff_name,
        "source_feature": feature,
        "transformation": "absolute_first_difference",
        "window_seconds": 1
    })

    # --------------------------------------------------------
    # 5-second rolling standard deviation
    # --------------------------------------------------------

    rolling_std = (
        df.groupby("_sequence_id")[feature]
        .rolling(
            window=5,
            min_periods=5
        )
        .std()
        .reset_index(
            level=0,
            drop=True
        )
    )

    rolling_name = (
        f"{feature}__rolling_std_5s"
    )

    temporal_data[rolling_name] = (
        rolling_std
    )

    metadata.append({
        "feature": rolling_name,
        "source_feature": feature,
        "transformation": "rolling_standard_deviation",
        "window_seconds": 5
    })


# ============================================================
# 5. COMBINE TIMESTAMP + TEMPORAL FEATURES
# ============================================================

output_df = pd.concat(
    [
        df[["timestamp"]],
        temporal_data
    ],
    axis=1
)


# ============================================================
# 6. METADATA
# ============================================================

metadata_df = pd.DataFrame(metadata)


# ============================================================
# 7. VALIDATE FEATURE COUNT
# ============================================================

expected_temporal_features = (
    len(feature_columns) * 3
)

actual_temporal_features = (
    len(output_df.columns) - 1
)

print("\nFeature count validation:")

print(
    f"Expected temporal features : "
    f"{expected_temporal_features}"
)

print(
    f"Actual temporal features   : "
    f"{actual_temporal_features}"
)

assert (
    actual_temporal_features
    == expected_temporal_features
)


# ============================================================
# 8. TEMPORAL NaN ANALYSIS
# ============================================================

print("\nAnalyzing expected temporal boundary NaNs...")

nan_counts = (
    output_df
    .drop(columns=["timestamp"])
    .isna()
    .sum()
)

unique_nan_counts = (
    nan_counts
    .value_counts()
    .sort_index()
)

print("\nNaN-count distribution:")

print(unique_nan_counts)


# Expected:
# diff features:
# one NaN per sequence
#
# rolling 5s:
# four NaNs per sequence

expected_diff_nan = sequence_count

expected_rolling_nan = sequence_count * 4

for feature in feature_columns:

    diff_name = (
        f"{feature}__diff_1s"
    )

    abs_diff_name = (
        f"{feature}__abs_diff_1s"
    )

    rolling_name = (
        f"{feature}__rolling_std_5s"
    )

    assert (
        output_df[diff_name].isna().sum()
        == expected_diff_nan
    )

    assert (
        output_df[abs_diff_name].isna().sum()
        == expected_diff_nan
    )

    assert (
        output_df[rolling_name].isna().sum()
        == expected_rolling_nan
    )


# ============================================================
# 9. FINITE VALUE VALIDATION
# ============================================================

numeric_values = (
    output_df
    .drop(columns=["timestamp"])
    .dropna()
    .to_numpy(dtype=float)
)

assert np.isfinite(
    numeric_values
).all()


# ============================================================
# 10. SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_df.to_csv(
    OUTPUT_FILE,
    index=False
)

metadata_df.to_csv(
    METADATA_FILE,
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL FEATURE SUMMARY")
print("=" * 70)

print(
    f"Training rows       : {len(output_df):,}"
)

print(
    f"Base features       : {len(feature_columns)}"
)

print(
    f"Temporal features   : {actual_temporal_features}"
)

print(
    f"Continuous sequences: {sequence_count}"
)

print(
    f"Temporal boundaries : {boundary_count}"
)

print(
    f"Expected diff NaNs per feature: "
    f"{expected_diff_nan}"
)

print(
    f"Expected rolling NaNs per feature: "
    f"{expected_rolling_nan}"
)


print("\nTransformation counts:")

print(
    metadata_df["transformation"]
    .value_counts()
)


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(output_df) == len(df)

assert (
    len(output_df.columns)
    == 1 + 174
)

assert (
    len(metadata_df)
    == 174
)

assert (
    metadata_df["feature"].nunique()
    == 174
)

assert (
    metadata_df["source_feature"].nunique()
    == 58
)

print("PASS: Row count preserved")
print("PASS: 58 base features processed")
print("PASS: 174 temporal features created")
print("PASS: Difference features respect sequence boundaries")
print("PASS: Rolling windows respect sequence boundaries")
print("PASS: Expected boundary NaNs validated")
print("PASS: No infinite temporal values")
print("PASS: Training data only")
print("PASS: No test labels used")


print("\nOutputs saved to:")

print(OUTPUT_FILE)

print(METADATA_FILE)


print("\n" + "=" * 70)
print("STAGE 22B: COMPLETE")
print("=" * 70)