from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STAGE 21A: TRAINING FEATURE DISTRIBUTION & SCALING VALIDATION
# ============================================================

print("=" * 70)
print("STAGE 21A: TRAINING FEATURE DISTRIBUTION & SCALING VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAINING_FILE = (
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
    / "anomaly_detection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_2305_training_feature_distribution.csv"
)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("\nLoading model-ready training data...")

training = pd.read_csv(TRAINING_FILE)

print("Training data loaded successfully.")


# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

assert "timestamp" in training.columns

features = [
    column
    for column in training.columns
    if column != "timestamp"
]

assert len(features) == 58

print("\nTraining rows     :", f"{len(training):,}")
print("ML features       :", len(features))


# ------------------------------------------------------------
# NUMERIC VALIDATION
# ------------------------------------------------------------

non_numeric = [
    feature
    for feature in features
    if not pd.api.types.is_numeric_dtype(
        training[feature]
    )
]

assert not non_numeric, (
    f"Non-numeric features found: {non_numeric}"
)

print("PASS: All 58 features are numeric")


# ------------------------------------------------------------
# BUILD DISTRIBUTION TABLE
# ------------------------------------------------------------

rows = []

for feature in features:

    series = training[feature]

    minimum = series.min()
    maximum = series.max()
    mean = series.mean()
    std = series.std()
    median = series.median()

    unique_values = series.nunique()

    missing = series.isna().sum()

    infinite = np.isinf(
        series.to_numpy()
    ).sum()

    zero_count = (
        series == 0
    ).sum()

    rows.append({
        "feature": feature,
        "unique_values": unique_values,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "std": std,
        "median": median,
        "missing_values": missing,
        "infinite_values": infinite,
        "zero_count": zero_count,
    })


distribution = pd.DataFrame(rows)


# ------------------------------------------------------------
# RANGE / VARIABILITY FLAGS
# ------------------------------------------------------------

distribution["zero_variance"] = (
    distribution["std"] == 0
)

distribution["very_low_variance"] = (
    distribution["std"] > 0
) & (
    distribution["std"] < 1e-8
)

distribution["large_range"] = (
    distribution["maximum"]
    - distribution["minimum"]
) > 1000


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DISTRIBUTION VALIDATION")
print("=" * 70)


zero_variance_count = (
    distribution["zero_variance"]
).sum()

very_low_variance_count = (
    distribution["very_low_variance"]
).sum()

missing_count = (
    distribution["missing_values"]
).sum()

infinite_count = (
    distribution["infinite_values"]
).sum()


print(
    "Zero-variance features      :",
    zero_variance_count
)

print(
    "Very-low-variance features  :",
    very_low_variance_count
)

print(
    "Total missing values       :",
    missing_count
)

print(
    "Total infinite values      :",
    infinite_count
)


assert zero_variance_count == 0
assert missing_count == 0
assert infinite_count == 0


print("PASS: No zero-variance features")
print("PASS: No missing values")
print("PASS: No infinite values")


# ------------------------------------------------------------
# FEATURE SCALE SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE SCALE SUMMARY")
print("=" * 70)

scale_summary = distribution[
    [
        "feature",
        "minimum",
        "maximum",
        "mean",
        "std",
        "unique_values",
    ]
].copy()

scale_summary = scale_summary.sort_values(
    by="std",
    ascending=False
)

print(
    scale_summary.head(20).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# SCALE COMPARISON
# ------------------------------------------------------------

std_min = distribution["std"].min()
std_max = distribution["std"].max()

if std_min > 0:
    scale_ratio = std_max / std_min
else:
    scale_ratio = np.inf

print("\n" + "=" * 70)
print("SCALE COMPARISON")
print("=" * 70)

print(
    "Smallest feature standard deviation:",
    std_min
)

print(
    "Largest feature standard deviation :",
    std_max
)

print(
    "Std-dev ratio                      :",
    scale_ratio
)


# ------------------------------------------------------------
# SCALING DECISION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SCALING DECISION")
print("=" * 70)

print(
    "Decision: Standardization will be evaluated before "
    "Isolation Forest training."
)

print(
    "Reason: HAI variables represent different physical/"
    "control quantities and may have substantially different "
    "numerical ranges."
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

distribution.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nOutput saved to:")
print(OUTPUT_FILE)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(distribution) == 58
assert distribution["feature"].is_unique

print("PASS: 58 feature distributions recorded")
print("PASS: No duplicate feature names")
print("PASS: All features numeric")
print("PASS: No missing values")
print("PASS: No infinite values")
print("PASS: No zero-variance features")

print("\n" + "=" * 70)
print("STAGE 21A: COMPLETE")
print("=" * 70)