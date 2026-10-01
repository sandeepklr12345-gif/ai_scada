from pathlib import Path
import os
import re
import numpy as np
import pandas as pd


# ============================================================
# STAGE 25C: CANDIDATE C FEATURE CONSTRUCTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE_DIR = str(PROJECT_ROOT)

FINAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "hai",
    "hai-23.05",
    "temporal_representation",
    "final_candidate"
)

TEST1_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_test1.csv"
)

MANIFEST_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_feature_manifest.csv"
)

OUTPUT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_reconstructed_features_test1.csv"
)


print("=" * 70)
print("STAGE 25C: CANDIDATE C FEATURE CONSTRUCTION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Candidate C Test 1 data...")

df = pd.read_csv(TEST1_PATH)

print(
    f"Rows loaded: {len(df)}"
)


# ============================================================
# LOAD MANIFEST
# ============================================================

print("\nLoading Candidate C feature manifest...")

manifest = pd.read_csv(MANIFEST_PATH)

print(
    f"Manifest rows: {len(manifest)}"
)


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

all_features = [
    c for c in df.columns
    if c not in ["timestamp", "label"]
]

assert len(all_features) == 118

print(
    f"Candidate C feature count: {len(all_features)}"
)


# ============================================================
# SEPARATE ORIGINAL / TEMPORAL FEATURES
# ============================================================

original_features = [
    c for c in all_features
    if "__" not in c
]

abs_diff_features = [
    c for c in all_features
    if "__abs_diff_1s" in c
]

rolling_features = [
    c for c in all_features
    if "__rolling_std_5s" in c
]

diff_features = [
    c for c in all_features
    if "__diff_1s" in c
]


print("\nFeature composition:")
print(
    f"Original features       : {len(original_features)}"
)
print(
    f"Diff features           : {len(diff_features)}"
)
print(
    f"Abs-diff features       : {len(abs_diff_features)}"
)
print(
    f"Rolling-std features    : {len(rolling_features)}"
)


# ============================================================
# CANDIDATE C EXPECTATIONS
# ============================================================

assert len(original_features) == 58
assert len(diff_features) == 0
assert len(abs_diff_features) == 30
assert len(rolling_features) == 30

print(
    "PASS: Candidate C composition = "
    "58 + 30 + 30 = 118"
)


# ============================================================
# EXTRACT BASE FEATURE NAMES
# ============================================================

abs_base_features = [
    c.replace("__abs_diff_1s", "")
    for c in abs_diff_features
]

rolling_base_features = [
    c.replace("__rolling_std_5s", "")
    for c in rolling_features
]


# ============================================================
# VALIDATE BASE FEATURES
# ============================================================

for feature in abs_base_features:

    assert feature in original_features, (
        f"Missing original feature for "
        f"abs_diff feature: {feature}"
    )


for feature in rolling_base_features:

    assert feature in original_features, (
        f"Missing original feature for "
        f"rolling feature: {feature}"
    )


print(
    "PASS: All temporal features map to "
    "Candidate C original features"
)


# ============================================================
# CREATE RECONSTRUCTED FEATURES
# ============================================================

print("\nConstructing Candidate C temporal features...")


# Work on a copy of the original 58 features.
result = df[
    ["timestamp"] + original_features
].copy()


# ------------------------------------------------------------
# ABSOLUTE FIRST DIFFERENCE
# ------------------------------------------------------------

for feature in abs_base_features:

    output_name = (
        f"{feature}__abs_diff_1s"
    )

    result[output_name] = (
        result[feature]
        .diff()
        .abs()
    )


# ------------------------------------------------------------
# 5-SECOND ROLLING STANDARD DEVIATION
# ------------------------------------------------------------

for feature in rolling_base_features:

    output_name = (
        f"{feature}__rolling_std_5s"
    )

    result[output_name] = (
        result[feature]
        .rolling(
            window=5,
            min_periods=5
        )
        .std()
    )


# ============================================================
# REORDER TO CANDIDATE C ORDER
# ============================================================

candidate_columns = [
    "timestamp"
] + all_features

result = result[
    candidate_columns
]


# ============================================================
# VALIDATE COLUMN ORDER
# ============================================================

assert list(
    result.columns
) == candidate_columns

print(
    "PASS: Reconstructed feature ordering "
    "matches Candidate C"
)


# ============================================================
# VALIDATE SHAPE
# ============================================================

assert result.shape == (
    len(df),
    119
)

print(
    f"PASS: Reconstructed matrix shape = "
    f"{result.shape[0]} x {result.shape[1]}"
)


# ============================================================
# VALIDATE ORIGINAL FEATURES
# ============================================================

for feature in original_features:

    assert np.allclose(
        result[feature].to_numpy(),
        df[feature].to_numpy(),
        equal_nan=True
    )


print(
    "PASS: Original 58 features preserved exactly"
)


# ============================================================
# EXPECTED TEMPORAL NaNs
# ============================================================

abs_nan_count = (
    result[abs_diff_features]
    .isna()
    .sum()
    .sum()
)

rolling_nan_count = (
    result[rolling_features]
    .isna()
    .sum()
    .sum()
)

total_temporal_nan = (
    abs_nan_count +
    rolling_nan_count
)


print("\nTemporal boundary values:")
print(
    f"Abs-diff NaNs       : {abs_nan_count}"
)
print(
    f"Rolling-std NaNs    : {rolling_nan_count}"
)
print(
    f"Total temporal NaNs : {total_temporal_nan}"
)


# For one continuous Test 1 sequence:
#
# abs_diff:
#   first row of each feature = NaN
#   30 features -> 30 NaNs
#
# rolling std:
#   first four rows of each feature = NaN
#   30 features -> 120 NaNs
#
# total = 150 NaNs


assert abs_nan_count == 30
assert rolling_nan_count == 120
assert total_temporal_nan == 150

print(
    "PASS: Expected temporal boundary NaNs confirmed"
)


# ============================================================
# COMPARE AGAINST EXISTING CANDIDATE C DATASET
# ============================================================

print("\nComparing reconstruction with existing Candidate C...")


existing = pd.read_csv(
    TEST1_PATH
)

# Candidate C source contains label.
# Compare only feature columns.

existing_features = existing[
    ["timestamp"] + all_features
].copy()


# Compare timestamps
assert (
    result["timestamp"].astype(str).to_numpy()
    ==
    existing_features["timestamp"].astype(str).to_numpy()
).all()

print(
    "PASS: Timestamps match"
)


# Compare every feature.
#
# NaNs are allowed only at expected temporal boundaries.

for feature in all_features:

    a = result[feature].to_numpy(
        dtype=float
    )

    b = existing_features[feature].to_numpy(
        dtype=float
    )

    assert np.allclose(
        a,
        b,
        equal_nan=True,
        rtol=1e-10,
        atol=1e-10
    ), (
        f"Feature reconstruction mismatch: "
        f"{feature}"
    )


print(
    "PASS: Reconstructed features match "
    "existing Candidate C exactly"
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\nOutput rows: "
    f"{len(result)}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25C VALIDATION")
print("=" * 70)

assert len(original_features) == 58

print(
    "PASS: 58 original features"
)

assert len(abs_diff_features) == 30

print(
    "PASS: 30 abs_diff_1s features"
)

assert len(rolling_features) == 30

print(
    "PASS: 30 rolling_std_5s features"
)

assert len(diff_features) == 0

print(
    "PASS: Candidate C correctly excludes diff_1s"
)

assert len(all_features) == 118

print(
    "PASS: Total Candidate C features = 118"
)

assert np.isfinite(
    result[original_features]
    .to_numpy()
).all()

print(
    "PASS: Original features contain no NaN/infinite values"
)

assert os.path.exists(OUTPUT_PATH)

print(
    "PASS: Reconstructed feature dataset saved"
)

print(
    "PASS: No model retraining"
)

print(
    "PASS: No threshold tuning"
)

print(
    "PASS: No feature selection"
)


print("\nOutput:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STAGE 25C: COMPLETE")
print("=" * 70)
