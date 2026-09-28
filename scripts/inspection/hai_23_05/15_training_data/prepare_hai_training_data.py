from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hai"
    / "hai-23.05"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
)

OUTPUT_DIR = (
    FEATURE_DIR
    / "training"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING FILES
# ============================================================

TRAIN_FILES = [
    "hai-train1.csv",
    "hai-train2.csv",
    "hai-train3.csv",
    "hai-train4.csv",
]


# ============================================================
# EXPECTED HAI SCADA FEATURES
# ============================================================

# The processed HAI test datasets established that HAI 23.05
# contains 86 SCADA variables plus timestamp and label.
#
# Training data should initially retain ALL 86 variables.
# Feature selection will happen later using training data only.


# ============================================================
# PROCESS EACH TRAINING DATASET
# ============================================================

all_training_data = []

print("=" * 70)
print("HAI 23.05 TRAINING DATA PREPARATION")
print("=" * 70)

for filename in TRAIN_FILES:

    input_file = RAW_DIR / filename

    print()
    print("-" * 70)
    print(f"PROCESSING: {filename}")
    print("-" * 70)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    if "timestamp" not in df.columns:
        raise ValueError(
            f"{filename}: timestamp column missing."
        )

    # Training files should NOT contain an anomaly label.
    if "label" in df.columns:
        raise ValueError(
            f"{filename}: unexpected label column found."
        )

    feature_columns = [
        c for c in df.columns
        if c != "timestamp"
    ]

    print(
        f"SCADA features : {len(feature_columns)}"
    )

    if len(feature_columns) != 86:
        raise ValueError(
            f"{filename}: expected 86 SCADA features, "
            f"found {len(feature_columns)}."
        )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_timestamps = timestamps.isna().sum()

    print(
        f"Invalid timestamps : {invalid_timestamps:,}"
    )

    if invalid_timestamps != 0:
        raise ValueError(
            f"{filename}: invalid timestamps detected."
        )

    df["timestamp"] = timestamps

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    print(
        f"Duplicate rows : {duplicate_rows:,}"
    )

    if duplicate_rows != 0:
        raise ValueError(
            f"{filename}: duplicate rows detected."
        )

    # --------------------------------------------------------
    # Duplicate timestamps
    # --------------------------------------------------------

    duplicate_timestamps = (
        df["timestamp"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate timestamps : {duplicate_timestamps:,}"
    )

    if duplicate_timestamps != 0:
        raise ValueError(
            f"{filename}: duplicate timestamps detected."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = df.isna().sum().sum()

    print(
        f"Missing values : {missing_values:,}"
    )

    if missing_values != 0:
        raise ValueError(
            f"{filename}: missing values detected."
        )

    # --------------------------------------------------------
    # Numeric feature validation
    # --------------------------------------------------------

    non_numeric = [
        c
        for c in feature_columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    if non_numeric:
        raise ValueError(
            f"{filename}: non-numeric SCADA features:\n"
            + "\n".join(non_numeric)
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    infinite_values = (
        df[feature_columns]
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(
        f"Infinite values : {infinite_values:,}"
    )

    if infinite_values != 0:
        raise ValueError(
            f"{filename}: infinite values detected."
        )

    # --------------------------------------------------------
    # Timestamp ordering
    # --------------------------------------------------------

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            f"{filename}: timestamps are not sorted."
        )

    # --------------------------------------------------------
    # Sampling interval
    # --------------------------------------------------------

    intervals = (
        df["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    if len(intervals) > 0:

        unique_intervals = sorted(
            intervals.unique()
        )

        print(
            f"Sampling intervals : {unique_intervals}"
        )

        if unique_intervals != [1.0]:
            raise ValueError(
                f"{filename}: expected 1-second sampling."
            )

    # --------------------------------------------------------
    # Save validated training dataset
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / filename

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved : {output_file}"
    )

    all_training_data.append(df)


# ============================================================
# COMBINED TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("COMBINING TRAINING DATA")
print("=" * 70)

combined = pd.concat(
    all_training_data,
    ignore_index=True
)

print(
    f"Combined rows : {len(combined):,}"
)

print(
    f"Combined columns : {len(combined.columns)}"
)


# ============================================================
# COMBINED VALIDATION
# ============================================================

if len(combined.columns) != 87:
    raise ValueError(
        "Combined dataset should contain "
        "1 timestamp + 86 SCADA features."
    )

if combined["timestamp"].isna().any():
    raise ValueError(
        "Combined dataset contains invalid timestamps."
    )

if combined.iloc[:, 1:].isna().any().any():
    raise ValueError(
        "Combined dataset contains missing values."
    )

# Check duplicate complete rows only.
combined_duplicate_rows = combined.duplicated().sum()

print(
    f"Combined duplicate rows : "
    f"{combined_duplicate_rows:,}"
)

# Do NOT reject duplicate timestamps across different
# training files automatically. Different training runs may
# have overlapping timestamp ranges.
#
# Each individual training file was already validated for
# unique timestamps.


# ============================================================
# SAVE COMBINED DATASET
# ============================================================

combined_file = (
    OUTPUT_DIR
    / "hai_2305_training_all.csv"
)

combined.to_csv(
    combined_file,
    index=False
)


# ============================================================
# TRAINING-ONLY FEATURE VARIABILITY ANALYSIS
# ============================================================

feature_columns = [
    c
    for c in combined.columns
    if c != "timestamp"
]

training_feature_summary = []

for feature in feature_columns:

    series = combined[feature]

    unique_values = series.nunique(
        dropna=False
    )

    training_feature_summary.append(
        {
            "feature": feature,
            "unique_values": unique_values,
            "constant_training": unique_values == 1,
            "minimum": series.min(),
            "maximum": series.max(),
        }
    )


training_feature_summary = pd.DataFrame(
    training_feature_summary
)

training_feature_summary = (
    training_feature_summary
    .sort_values("feature")
    .reset_index(drop=True)
)


summary_file = (
    OUTPUT_DIR
    / "hai_2305_training_feature_variability.csv"
)

training_feature_summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

constant_count = int(
    training_feature_summary[
        "constant_training"
    ].sum()
)

variable_count = (
    len(training_feature_summary)
    - constant_count
)


print()
print("=" * 70)
print("TRAINING FEATURE VARIABILITY")
print("=" * 70)

print(
    f"Total SCADA features : "
    f"{len(training_feature_summary)}"
)

print(
    f"Constant in training : "
    f"{constant_count}"
)

print(
    f"Variable in training : "
    f"{variable_count}"
)

print()
print("OUTPUT FILES")
print("-" * 70)

print(
    f"Individual training datasets:"
)

for filename in TRAIN_FILES:
    print(
        OUTPUT_DIR / filename
    )

print()
print(
    f"Combined dataset:"
)

print(combined_file)

print()
print(
    f"Training feature variability:"
)

print(summary_file)

print()
print("=" * 70)
print("HAI 23.05 TRAINING DATA PREPARATION COMPLETE")
print("=" * 70)