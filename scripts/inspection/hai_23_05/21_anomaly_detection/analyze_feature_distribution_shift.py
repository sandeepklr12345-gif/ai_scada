from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STAGE 21D: FEATURE DISTRIBUTION SHIFT ANALYSIS
# ============================================================

print("=" * 70)
print("STAGE 21D: FEATURE DISTRIBUTION SHIFT ANALYSIS")
print("=" * 70)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

BASE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
)

TRAINING_FILE = (
    BASE / "model_ready" / "hai_2305_training_model_ready.csv"
)

TEST1_FILE = (
    BASE / "model_ready" / "hai-test1_model_ready.csv"
)

TEST2_FILE = (
    BASE / "model_ready" / "hai-test2_model_ready.csv"
)

OUTPUT_DIR = (
    BASE / "anomaly_detection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_feature_distribution_shift.csv"
)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("\nLoading datasets...")

training = pd.read_csv(TRAINING_FILE)
test1 = pd.read_csv(TEST1_FILE)
test2 = pd.read_csv(TEST2_FILE)

print("Files loaded successfully.")


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

features = [
    c for c in training.columns
    if c != "timestamp"
]

assert len(features) == 58

print("\nTraining rows:", f"{len(training):,}")
print("Test 1 rows  :", f"{len(test1):,}")
print("Test 2 rows  :", f"{len(test2):,}")
print("Features     :", len(features))


# ------------------------------------------------------------
# TRAINING REFERENCE STATISTICS
# ------------------------------------------------------------

rows = []

for feature in features:

    train = training[feature]
    t1 = test1[feature]
    t2 = test2[feature]

    train_mean = train.mean()
    train_std = train.std()

    train_median = train.median()

    train_min = train.min()
    train_max = train.max()

    train_q01 = train.quantile(0.01)
    train_q05 = train.quantile(0.05)
    train_q25 = train.quantile(0.25)
    train_q50 = train.quantile(0.50)
    train_q75 = train.quantile(0.75)
    train_q95 = train.quantile(0.95)
    train_q99 = train.quantile(0.99)

    # --------------------------------------------------------
    # TEST STATISTICS
    # --------------------------------------------------------

    t1_mean = t1.mean()
    t1_std = t1.std()
    t1_median = t1.median()

    t2_mean = t2.mean()
    t2_std = t2.std()
    t2_median = t2.median()

    # --------------------------------------------------------
    # STANDARDIZED MEAN SHIFT
    # --------------------------------------------------------

    if train_std > 0:

        test1_mean_z = (
            t1_mean - train_mean
        ) / train_std

        test2_mean_z = (
            t2_mean - train_mean
        ) / train_std

        test1_median_z = (
            t1_median - train_median
        ) / train_std

        test2_median_z = (
            t2_median - train_median
        ) / train_std

    else:

        test1_mean_z = np.nan
        test2_mean_z = np.nan
        test1_median_z = np.nan
        test2_median_z = np.nan

    # --------------------------------------------------------
    # TRAINING RANGE COVERAGE
    # --------------------------------------------------------

    test1_below = (
        t1 < train_min
    ).mean() * 100

    test1_above = (
        t1 > train_max
    ).mean() * 100

    test2_below = (
        t2 < train_min
    ).mean() * 100

    test2_above = (
        t2 > train_max
    ).mean() * 100

    # --------------------------------------------------------
    # QUANTILE SHIFT
    # --------------------------------------------------------

    test1_q50 = t1.quantile(0.50)
    test2_q50 = t2.quantile(0.50)

    test1_q95 = t1.quantile(0.95)
    test2_q95 = t2.quantile(0.95)

    test1_q99 = t1.quantile(0.99)
    test2_q99 = t2.quantile(0.99)

    rows.append({
        "feature": feature,

        "training_mean": train_mean,
        "training_std": train_std,
        "training_median": train_median,

        "training_min": train_min,
        "training_max": train_max,

        "training_q01": train_q01,
        "training_q05": train_q05,
        "training_q25": train_q25,
        "training_q50": train_q50,
        "training_q75": train_q75,
        "training_q95": train_q95,
        "training_q99": train_q99,

        "test1_mean": t1_mean,
        "test1_std": t1_std,
        "test1_median": t1_median,

        "test1_mean_shift_z": test1_mean_z,
        "test1_median_shift_z": test1_median_z,

        "test1_below_training_min_pct": test1_below,
        "test1_above_training_max_pct": test1_above,

        "test1_q50": test1_q50,
        "test1_q95": test1_q95,
        "test1_q99": test1_q99,

        "test2_mean": t2_mean,
        "test2_std": t2_std,
        "test2_median": t2_median,

        "test2_mean_shift_z": test2_mean_z,
        "test2_median_shift_z": test2_median_z,

        "test2_below_training_min_pct": test2_below,
        "test2_above_training_max_pct": test2_above,

        "test2_q50": test2_q50,
        "test2_q95": test2_q95,
        "test2_q99": test2_q99,
    })


shift = pd.DataFrame(rows)


# ------------------------------------------------------------
# SHIFT FLAGS
# ------------------------------------------------------------

shift["test1_large_mean_shift"] = (
    shift["test1_mean_shift_z"].abs() >= 2
)

shift["test2_large_mean_shift"] = (
    shift["test2_mean_shift_z"].abs() >= 2
)

shift["test1_range_escape"] = (
    shift["test1_below_training_min_pct"]
    + shift["test1_above_training_max_pct"]
) > 1

shift["test2_range_escape"] = (
    shift["test2_below_training_min_pct"]
    + shift["test2_above_training_max_pct"]
) > 1


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DISTRIBUTION SHIFT SUMMARY")
print("=" * 70)

print(
    "Test 1 features with |mean shift| >= 2 std:",
    int(shift["test1_large_mean_shift"].sum())
)

print(
    "Test 2 features with |mean shift| >= 2 std:",
    int(shift["test2_large_mean_shift"].sum())
)

print(
    "Test 1 features with >1% outside training range:",
    int(shift["test1_range_escape"].sum())
)

print(
    "Test 2 features with >1% outside training range:",
    int(shift["test2_range_escape"].sum())
)


# ------------------------------------------------------------
# TOP SHIFTS
# ------------------------------------------------------------

print("\nTop Test 1 mean shifts:")

print(
    shift[
        [
            "feature",
            "test1_mean_shift_z",
            "test1_below_training_min_pct",
            "test1_above_training_max_pct",
        ]
    ]
    .assign(
        abs_shift=lambda x:
        x["test1_mean_shift_z"].abs()
    )
    .sort_values(
        "abs_shift",
        ascending=False
    )
    .head(15)
    .drop(columns=["abs_shift"])
    .to_string(index=False)
)


print("\nTop Test 2 mean shifts:")

print(
    shift[
        [
            "feature",
            "test2_mean_shift_z",
            "test2_below_training_min_pct",
            "test2_above_training_max_pct",
        ]
    ]
    .assign(
        abs_shift=lambda x:
        x["test2_mean_shift_z"].abs()
    )
    .sort_values(
        "abs_shift",
        ascending=False
    )
    .head(15)
    .drop(columns=["abs_shift"])
    .to_string(index=False)
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

shift.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(shift) == 58
assert shift["feature"].is_unique

numeric_columns = shift.select_dtypes(
    include="number"
).columns

assert np.isfinite(
    shift[numeric_columns].to_numpy()
).all()

print("PASS: 58 features analyzed")
print("PASS: Training used as reference")
print("PASS: Test labels not used")
print("PASS: Distribution statistics finite")
print("PASS: Mean-shift analysis complete")
print("PASS: Training-range coverage analyzed")

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STAGE 21D: COMPLETE")
print("=" * 70)