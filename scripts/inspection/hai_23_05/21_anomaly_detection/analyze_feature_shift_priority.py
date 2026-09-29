from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STAGE 21E: FEATURE SHIFT PRIORITY ANALYSIS
# ============================================================

print("=" * 70)
print("STAGE 21E: FEATURE SHIFT PRIORITY ANALYSIS")
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

OUTPUT_DIR = BASE / "anomaly_detection"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_feature_shift_priority.csv"
)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("\nLoading datasets...")

training = pd.read_csv(TRAINING_FILE)
test1 = pd.read_csv(TEST1_FILE)
test2 = pd.read_csv(TEST2_FILE)

print("Files loaded successfully.")


features = [
    c for c in training.columns
    if c != "timestamp"
]

assert len(features) == 58


# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------

rows = []

for feature in features:

    tr = training[feature]
    t1 = test1[feature]
    t2 = test2[feature]

    tr_mean = tr.mean()
    tr_std = tr.std()
    tr_min = tr.min()
    tr_max = tr.max()

    # Avoid division by zero
    if tr_std > 0:
        t1_mean_shift = abs(
            (t1.mean() - tr_mean) / tr_std
        )

        t2_mean_shift = abs(
            (t2.mean() - tr_mean) / tr_std
        )

        t1_median_shift = abs(
            (t1.median() - tr.median()) / tr_std
        )

        t2_median_shift = abs(
            (t2.median() - tr.median()) / tr_std
        )

    else:
        t1_mean_shift = 0
        t2_mean_shift = 0
        t1_median_shift = 0
        t2_median_shift = 0

    # Range violations
    t1_below = (
        t1 < tr_min
    ).mean() * 100

    t1_above = (
        t1 > tr_max
    ).mean() * 100

    t2_below = (
        t2 < tr_min
    ).mean() * 100

    t2_above = (
        t2 > tr_max
    ).mean() * 100

    total_t1_range_violation = (
        t1_below + t1_above
    )

    total_t2_range_violation = (
        t2_below + t2_above
    )

    # Quantile shifts
    train_q05 = tr.quantile(0.05)
    train_q50 = tr.quantile(0.50)
    train_q95 = tr.quantile(0.95)

    test1_q05 = t1.quantile(0.05)
    test1_q50 = t1.quantile(0.50)
    test1_q95 = t1.quantile(0.95)

    test2_q05 = t2.quantile(0.05)
    test2_q50 = t2.quantile(0.50)
    test2_q95 = t2.quantile(0.95)

    # Normalized quantile displacement
    if tr_std > 0:

        t1_q05_shift = abs(
            (test1_q05 - train_q05)
            / tr_std
        )

        t1_q50_shift = abs(
            (test1_q50 - train_q50)
            / tr_std
        )

        t1_q95_shift = abs(
            (test1_q95 - train_q95)
            / tr_std
        )

        t2_q05_shift = abs(
            (test2_q05 - train_q05)
            / tr_std
        )

        t2_q50_shift = abs(
            (test2_q50 - train_q50)
            / tr_std
        )

        t2_q95_shift = abs(
            (test2_q95 - train_q95)
            / tr_std
        )

    else:

        t1_q05_shift = 0
        t1_q50_shift = 0
        t1_q95_shift = 0

        t2_q05_shift = 0
        t2_q50_shift = 0
        t2_q95_shift = 0

    # Overall shift priority
    shift_priority = max(
        t1_mean_shift,
        t2_mean_shift,
        t1_median_shift,
        t2_median_shift,
        t1_q05_shift,
        t1_q50_shift,
        t1_q95_shift,
        t2_q05_shift,
        t2_q50_shift,
        t2_q95_shift,
    )

    range_priority = max(
        total_t1_range_violation,
        total_t2_range_violation,
    )

    rows.append({
        "feature": feature,

        "training_mean": tr_mean,
        "training_std": tr_std,
        "training_min": tr_min,
        "training_max": tr_max,

        "test1_mean_shift_z": t1_mean_shift,
        "test2_mean_shift_z": t2_mean_shift,

        "test1_median_shift_z": t1_median_shift,
        "test2_median_shift_z": t2_median_shift,

        "test1_below_training_min_pct": t1_below,
        "test1_above_training_max_pct": t1_above,
        "test1_total_range_violation_pct":
            total_t1_range_violation,

        "test2_below_training_min_pct": t2_below,
        "test2_above_training_max_pct": t2_above,
        "test2_total_range_violation_pct":
            total_t2_range_violation,

        "test1_q05_shift_z": t1_q05_shift,
        "test1_q50_shift_z": t1_q50_shift,
        "test1_q95_shift_z": t1_q95_shift,

        "test2_q05_shift_z": t2_q05_shift,
        "test2_q50_shift_z": t2_q50_shift,
        "test2_q95_shift_z": t2_q95_shift,

        "shift_priority": shift_priority,
        "range_priority": range_priority,
    })


result = pd.DataFrame(rows)


# ------------------------------------------------------------
# RANKING
# ------------------------------------------------------------

result = result.sort_values(
    by=[
        "range_priority",
        "shift_priority",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(drop=True)


result["shift_rank"] = (
    np.arange(len(result)) + 1
)


# ------------------------------------------------------------
# PRINT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP FEATURE SHIFT PRIORITIES")
print("=" * 70)

print(
    result[
        [
            "shift_rank",
            "feature",
            "shift_priority",
            "range_priority",
            "test1_total_range_violation_pct",
            "test2_total_range_violation_pct",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(result) == 58
assert result["feature"].is_unique
assert result["shift_rank"].tolist() == list(
    range(1, 59)
)

numeric_columns = result.select_dtypes(
    include="number"
).columns

assert np.isfinite(
    result[numeric_columns].to_numpy()
).all()

print("PASS: 58 features analyzed")
print("PASS: Features ranked by distribution shift")
print("PASS: Range violations calculated")
print("PASS: Quantile shifts calculated")
print("PASS: No anomaly labels used")
print("PASS: All statistics finite")

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STAGE 21E: COMPLETE")
print("=" * 70)