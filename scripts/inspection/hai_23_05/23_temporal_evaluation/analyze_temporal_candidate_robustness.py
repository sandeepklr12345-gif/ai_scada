from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STAGE 23H
# HAI 23.05 - Temporal Candidate Robustness Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TEMPORAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
)

INPUT_FILE = (
    TEMPORAL_DIR
    / "reduced_candidates"
    / "hai_2305_temporal_reduced_candidate_results.csv"
)

OUTPUT_DIR = (
    TEMPORAL_DIR
    / "reduced_candidates"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_candidate_robustness.csv"
)

REFERENCE_FEATURE_COUNT = 232


print("=" * 70)
print("STAGE 23H: TEMPORAL CANDIDATE ROBUSTNESS ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD RESULTS
# ============================================================

print("\nLoading Stage 23G results...")

df = pd.read_csv(
    INPUT_FILE
)

assert len(df) == 6

print(
    f"Candidates loaded: {len(df)}"
)


# ============================================================
# 2. CALCULATE ROBUSTNESS METRICS
# ============================================================

df["mean_f1"] = (
    df[
        [
            "test1_f1",
            "test2_f1"
        ]
    ]
    .mean(axis=1)
)

df["f1_std"] = (
    df[
        [
            "test1_f1",
            "test2_f1"
        ]
    ]
    .std(
        axis=1,
        ddof=0
    )
)

df["f1_test_difference"] = (
    (
        df["test1_f1"]
        - df["test2_f1"]
    )
    .abs()
)

df["mean_precision"] = (
    df[
        [
            "test1_precision",
            "test2_precision"
        ]
    ]
    .mean(axis=1)
)

df["mean_recall"] = (
    df[
        [
            "test1_recall",
            "test2_recall"
        ]
    ]
    .mean(axis=1)
)

df["feature_reduction_percent"] = (
    1
    -
    (
        df["feature_count"]
        / REFERENCE_FEATURE_COUNT
    )
) * 100


# ============================================================
# 3. COMPARE AGAINST FULL REFERENCE
# ============================================================

reference = df[
    df["candidate"]
    == "full_temporal_reference"
].iloc[0]

df["test1_f1_delta_vs_full"] = (
    df["test1_f1"]
    - reference["test1_f1"]
)

df["test2_f1_delta_vs_full"] = (
    df["test2_f1"]
    - reference["test2_f1"]
)

df["mean_f1_delta_vs_full"] = (
    df["mean_f1"]
    - reference["mean_f1"]
)


# ============================================================
# 4. DISPLAY RESULTS
# ============================================================

display_columns = [
    "candidate",
    "feature_count",
    "feature_reduction_percent",
    "test1_f1",
    "test2_f1",
    "mean_f1",
    "f1_std",
    "f1_test_difference",
    "mean_precision",
    "mean_recall",
    "test1_f1_delta_vs_full",
    "test2_f1_delta_vs_full",
    "mean_f1_delta_vs_full"
]

print(
    "\n" + "=" * 70
)

print(
    "CANDIDATE ROBUSTNESS RESULTS"
)

print(
    "=" * 70
)

print(
    df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 5. TRAINING-ONLY FEATURE EFFICIENCY VIEW
# ============================================================

non_reference = df[
    df["candidate"]
    != "full_temporal_reference"
].copy()

non_reference[
    "mean_f1_per_100_features"
] = (
    non_reference["mean_f1"]
    /
    non_reference["feature_count"]
    * 100
)

print(
    "\n" + "=" * 70
)

print(
    "FEATURE-EFFICIENCY VIEW"
)

print(
    "=" * 70
)

print(
    non_reference[
        [
            "candidate",
            "feature_count",
            "mean_f1",
            "mean_f1_per_100_features"
        ]
    ]
    .sort_values(
        "feature_count"
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 6. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 7. VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23H FINAL VALIDATION"
)

print(
    "=" * 70
)

assert len(df) == 6

assert (
    df["feature_count"]
    .tolist()
    ==
    [
        78,
        98,
        118,
        88,
        118,
        232
    ]
)

required_columns = [
    "mean_f1",
    "f1_std",
    "f1_test_difference",
    "mean_precision",
    "mean_recall",
    "feature_reduction_percent",
    "test1_f1_delta_vs_full",
    "test2_f1_delta_vs_full",
    "mean_f1_delta_vs_full"
]

for column in required_columns:

    assert column in df.columns

assert np.isfinite(
    df[
        required_columns
    ].to_numpy()
).all()

assert Path(
    OUTPUT_FILE
).exists()

print(
    "PASS: All six candidates analyzed"
)

print(
    "PASS: Cross-test F1 stability calculated"
)

print(
    "PASS: Precision stability metrics calculated"
)

print(
    "PASS: Recall stability metrics calculated"
)

print(
    "PASS: Feature reduction calculated"
)

print(
    "PASS: Comparison against full reference calculated"
)

print(
    "PASS: No model retraining"
)

print(
    "PASS: No threshold tuning"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23H: COMPLETE"
)

print(
    "=" * 70
)