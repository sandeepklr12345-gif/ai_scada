from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STAGE 22A
# HAI 23.05 - Temporal Feature Feasibility Analysis
# Boundary-Aware Version
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"
    r"\data\features\hai\hai-23.05\model_ready"
    r"\hai_2305_training_model_ready.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"
    r"\data\features\hai\hai-23.05"
    r"\temporal_representation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_2305_training_temporal_feasibility.csv"
)

BOUNDARY_FILE = (
    OUTPUT_DIR
    / "hai_2305_training_temporal_boundaries.csv"
)


print("=" * 70)
print("STAGE 22A: TEMPORAL FEATURE FEASIBILITY ANALYSIS")
print("Boundary-Aware Analysis")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading training model-ready data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

assert "timestamp" in df.columns, "Missing timestamp column"

feature_columns = [
    col for col in df.columns
    if col != "timestamp"
]

print(f"Features   : {len(feature_columns)}")

assert len(feature_columns) == 58, (
    f"Expected 58 features, found {len(feature_columns)}"
)

timestamps = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

assert timestamps.notna().all(), (
    "Invalid timestamps detected"
)

assert df[feature_columns].isna().sum().sum() == 0, (
    "Missing feature values detected"
)

assert np.isfinite(
    df[feature_columns].to_numpy(dtype=float)
).all(), (
    "Infinite feature values detected"
)


# ============================================================
# 3. TIMESTAMP GAP ANALYSIS
# ============================================================

print("\nAnalyzing timestamp continuity...")

time_diff = timestamps.diff()

# Normal HAI sampling interval
NORMAL_INTERVAL = pd.Timedelta(seconds=1)

# First row has no previous timestamp
valid_diff = time_diff.iloc[1:]

normal_intervals = (
    valid_diff == NORMAL_INTERVAL
)

gap_mask = (
    valid_diff > NORMAL_INTERVAL
)

unexpected_mask = (
    valid_diff < NORMAL_INTERVAL
)

normal_count = int(normal_intervals.sum())
gap_count = int(gap_mask.sum())
unexpected_count = int(unexpected_mask.sum())

print("\nTimestamp continuity summary:")
print(f"1-second intervals : {normal_count:,}")
print(f"Temporal gaps      : {gap_count:,}")
print(f"Unexpected intervals: {unexpected_count:,}")

if gap_count > 0:

    boundary_indices = (
        np.where(gap_mask.to_numpy())[0] + 1
    )

    boundary_records = []

    for idx in boundary_indices:

        previous_time = timestamps.iloc[idx - 1]
        current_time = timestamps.iloc[idx]

        gap_duration = (
            current_time - previous_time
        )

        boundary_records.append({
            "row_index": idx,
            "previous_timestamp": previous_time,
            "current_timestamp": current_time,
            "gap_duration": gap_duration,
            "gap_seconds": gap_duration.total_seconds()
        })

    boundary_df = pd.DataFrame(
        boundary_records
    )

else:

    boundary_df = pd.DataFrame(
        columns=[
            "row_index",
            "previous_timestamp",
            "current_timestamp",
            "gap_duration",
            "gap_seconds"
        ]
    )


# Save boundary information
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

boundary_df.to_csv(
    BOUNDARY_FILE,
    index=False
)

print(
    f"\nTemporal boundaries recorded: "
    f"{len(boundary_df):,}"
)

print(
    f"Boundary file:\n{BOUNDARY_FILE}"
)


# ============================================================
# 4. CREATE CONTINUOUS SEQUENCE IDs
# ============================================================

print("\nCreating continuous temporal sequences...")

# Every gap creates a new sequence.
sequence_id = (
    (time_diff > NORMAL_INTERVAL)
    .cumsum()
)

df["_sequence_id"] = sequence_id

sequence_count = int(
    df["_sequence_id"].nunique()
)

print(
    f"Continuous sequences: {sequence_count:,}"
)


# ============================================================
# 5. TEMPORAL FEATURE ANALYSIS
# ============================================================

print("\nCalculating temporal statistics...")

results = []

for i, feature in enumerate(
    feature_columns,
    start=1
):

    print(
        f"[{i:02d}/{len(feature_columns)}] "
        f"Analyzing {feature}"
    )

    series = df[feature].astype(float)

    # --------------------------------------------------------
    # First difference within each continuous sequence
    # --------------------------------------------------------

    diff = (
        df.groupby("_sequence_id")[feature]
        .diff()
    )

    # The first row of every sequence has no valid difference
    diff = diff.dropna()

    abs_diff = diff.abs()

    # --------------------------------------------------------
    # Temporal activity
    # --------------------------------------------------------

    changed_points = int(
        (abs_diff > 0).sum()
    )

    total_valid_changes = len(diff)

    change_rate = (
        changed_points / total_valid_changes
        if total_valid_changes > 0
        else 0
    )

    # --------------------------------------------------------
    # Rolling variability
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
        .dropna()
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    results.append({

        "feature": feature,

        "unique_values": int(
            series.nunique()
        ),

        "minimum": series.min(),

        "maximum": series.max(),

        "change_rate": change_rate,

        "mean_absolute_change": (
            abs_diff.mean()
        ),

        "median_absolute_change": (
            abs_diff.median()
        ),

        "p95_absolute_change": (
            abs_diff.quantile(0.95)
        ),

        "maximum_absolute_change": (
            abs_diff.max()
        ),

        "first_difference_std": (
            diff.std()
        ),

        "rolling_5sec_std_mean": (
            rolling_std.mean()
        ),

        "rolling_5sec_std_median": (
            rolling_std.median()
        ),

        "rolling_5sec_std_p95": (
            rolling_std.quantile(0.95)
        ),

        "rolling_5sec_std_max": (
            rolling_std.max()
        )
    })


result_df = pd.DataFrame(results)


# ============================================================
# 6. TEMPORAL ACTIVITY CLASSIFICATION
# ============================================================

def classify_activity(change_rate):

    if change_rate == 0:
        return "STATIC"

    if change_rate < 0.01:
        return "VERY_LOW_ACTIVITY"

    if change_rate < 0.10:
        return "LOW_ACTIVITY"

    if change_rate < 0.50:
        return "MODERATE_ACTIVITY"

    return "HIGH_ACTIVITY"


result_df["temporal_activity"] = (
    result_df["change_rate"]
    .apply(classify_activity)
)


result_df["temporal_signal_present"] = (
    result_df["change_rate"] > 0
)


# ============================================================
# 7. SORT RESULTS
# ============================================================

result_df = (
    result_df
    .sort_values(
        by=[
            "change_rate",
            "p95_absolute_change"
        ],
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL FEASIBILITY SUMMARY")
print("=" * 70)

print(
    f"\nContinuous sequences : "
    f"{sequence_count:,}"
)

print(
    f"Temporal boundaries  : "
    f"{len(boundary_df):,}"
)

print(
    f"Valid 1-sec changes  : "
    f"{len(df) - sequence_count:,}"
)

print("\nTemporal activity categories:")

activity_counts = (
    result_df["temporal_activity"]
    .value_counts()
)

for category, count in activity_counts.items():

    print(
        f"{category:20s}: {count}"
    )


static_count = (
    result_df["temporal_activity"]
    == "STATIC"
).sum()

temporal_count = (
    result_df["temporal_signal_present"]
).sum()


print(
    f"\nStatic features:"
    f" {static_count}"
)

print(
    f"Features with temporal variation:"
    f" {temporal_count}"
)


# ============================================================
# 10. TOP TEMPORALLY ACTIVE FEATURES
# ============================================================

print("\nTop 15 temporally active features:")

print(
    result_df[
        [
            "feature",
            "change_rate",
            "median_absolute_change",
            "p95_absolute_change",
            "rolling_5sec_std_p95",
            "temporal_activity"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 11. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(result_df) == 58

assert (
    result_df["feature"].nunique()
    == 58
)

assert (
    result_df["change_rate"]
    .notna()
    .all()
)

assert (
    np.isfinite(
        result_df.select_dtypes(
            include=np.number
        ).to_numpy()
    ).all()
)

assert (
    len(boundary_df)
    == gap_count
)

assert (
    "_sequence_id" in df.columns
)

print("PASS: 58 features analyzed")
print("PASS: Timestamp gaps handled as sequence boundaries")
print("PASS: No temporal differences calculated across gaps")
print("PASS: Temporal statistics are finite")
print("PASS: Activity classification complete")
print("PASS: Boundary report created")
print("PASS: Training data only")
print("PASS: No test labels used")


print("\nOutputs saved to:")

print(OUTPUT_FILE)

print(BOUNDARY_FILE)


print("\n" + "=" * 70)
print("STAGE 22A: COMPLETE")
print("=" * 70)