from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# STAGE 25D: END-TO-END CANDIDATE C REPRODUCIBILITY
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

MODEL_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_isolation_forest.joblib"
)

TEST1_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_test1.csv"
)

EXISTING_INFERENCE_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_batch_inference_test.csv"
)

RECONSTRUCTED_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_reconstructed_features_test1.csv"
)

OUTPUT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_end_to_end_inference.csv"
)


print("=" * 70)
print("STAGE 25D: END-TO-END CANDIDATE C REPRODUCIBILITY")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Candidate C detector...")

model = joblib.load(MODEL_PATH)

print("PASS: Model loaded")


# ============================================================
# LOAD ORIGINAL TEST DATA
# ============================================================

print("\nLoading original Candidate C Test 1 data...")

source = pd.read_csv(TEST1_PATH)

print(
    f"Source rows: {len(source)}"
)


# ============================================================
# IDENTIFY 58 ORIGINAL FEATURES
# ============================================================

candidate_features = [
    c for c in source.columns
    if c not in ["timestamp", "label"]
]

assert len(candidate_features) == 118

original_features = [
    c for c in candidate_features
    if "__" not in c
]

assert len(original_features) == 58

print(
    f"Original SCADA features: {len(original_features)}"
)


# ============================================================
# BUILD TEMPORAL FEATURES FROM ORIGINAL DATA
# ============================================================

print("\nReconstructing Candidate C features...")

reconstructed = source[
    ["timestamp"] + original_features
].copy()


# ------------------------------------------------------------
# Extract required temporal feature bases
# ------------------------------------------------------------

abs_features = [
    c for c in candidate_features
    if "__abs_diff_1s" in c
]

rolling_features = [
    c for c in candidate_features
    if "__rolling_std_5s" in c
]

assert len(abs_features) == 30
assert len(rolling_features) == 30


# ------------------------------------------------------------
# Absolute first differences
# ------------------------------------------------------------

for feature in abs_features:

    base = feature.replace(
        "__abs_diff_1s",
        ""
    )

    reconstructed[feature] = (
        reconstructed[base]
        .diff()
        .abs()
    )


# ------------------------------------------------------------
# Five-second rolling standard deviation
# ------------------------------------------------------------

for feature in rolling_features:

    base = feature.replace(
        "__rolling_std_5s",
        ""
    )

    reconstructed[feature] = (
        reconstructed[base]
        .rolling(
            window=5,
            min_periods=5
        )
        .std()
    )


# ============================================================
# REORDER
# ============================================================

reconstructed = reconstructed[
    ["timestamp"] + candidate_features
]


# ============================================================
# VERIFY RECONSTRUCTION
# ============================================================

print(
    "\nValidating reconstructed Candidate C matrix..."
)

assert list(
    reconstructed.columns
) == ["timestamp"] + candidate_features

print(
    "PASS: Candidate C feature ordering"
)


# ============================================================
# VALID ROWS
# ============================================================

X = reconstructed[candidate_features]

valid_mask = X.notna().all(axis=1)

X_valid = X.loc[valid_mask].copy()

timestamps = reconstructed.loc[
    valid_mask,
    "timestamp"
].reset_index(drop=True)


print(
    f"Valid inference rows: {len(X_valid)}"
)

assert len(X_valid) == 53996

print(
    "PASS: Expected 53,996 valid rows"
)


# ============================================================
# RUN MODEL
# ============================================================

print("\nRunning end-to-end inference...")

scores = model.decision_function(
    X_valid.to_numpy()
)

predictions = (
    scores < 0
).astype(int)


print(
    "PASS: Inference completed"
)


# ============================================================
# CREATE OUTPUT
# ============================================================

output = pd.DataFrame({
    "timestamp": timestamps,
    "anomaly_score": scores,
    "prediction": predictions
})

output["status"] = np.where(
    output["prediction"] == 1,
    "ANOMALY",
    "NORMAL"
)


# ============================================================
# COMPARE AGAINST EXISTING 25B RESULTS
# ============================================================

print(
    "\nComparing against Stage 25B batch inference..."
)

existing = pd.read_csv(
    EXISTING_INFERENCE_PATH
)

assert len(existing) == len(output)

print(
    "PASS: Row counts match"
)


# ------------------------------------------------------------
# Timestamp comparison
# ------------------------------------------------------------

assert (
    output["timestamp"].astype(str).to_numpy()
    ==
    existing["timestamp"].astype(str).to_numpy()
).all()

print(
    "PASS: Timestamps match"
)


# ------------------------------------------------------------
# Score comparison
# ------------------------------------------------------------

score_difference = np.abs(
    output["anomaly_score"].to_numpy()
    -
    existing["anomaly_score"].to_numpy()
)

max_score_difference = (
    score_difference.max()
)

print(
    f"Maximum score difference: "
    f"{max_score_difference:.15f}"
)

assert max_score_difference <= 1e-12

print(
    "PASS: Anomaly scores reproduce exactly"
)


# ------------------------------------------------------------
# Prediction comparison
# ------------------------------------------------------------

assert (
    output["prediction"].to_numpy()
    ==
    existing["prediction"].to_numpy()
).all()

print(
    "PASS: Predictions reproduce exactly"
)


# ------------------------------------------------------------
# Status comparison
# ------------------------------------------------------------

assert (
    output["status"].to_numpy()
    ==
    existing["status"].to_numpy()
).all()

print(
    "PASS: Status values reproduce exactly"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("END-TO-END INFERENCE SUMMARY")
print("=" * 70)

print(
    f"Source rows          : {len(source)}"
)

print(
    f"Valid rows            : {len(output)}"
)

print(
    f"Candidate C features  : {len(candidate_features)}"
)

print(
    f"Normal predictions    : "
    f"{(output['prediction'] == 0).sum()}"
)

print(
    f"Anomaly predictions   : "
    f"{(output['prediction'] == 1).sum()}"
)

print(
    f"Anomaly rate          : "
    f"{(output['prediction'] == 1).mean():.6f}"
)

print(
    f"Mean anomaly score    : "
    f"{output['anomaly_score'].mean():.6f}"
)


# ============================================================
# SAVE
# ============================================================

output.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25D VALIDATION")
print("=" * 70)

assert len(candidate_features) == 118

print(
    "PASS: Candidate C has 118 features"
)

assert len(original_features) == 58

print(
    "PASS: 58 original features used"
)

assert len(abs_features) == 30

print(
    "PASS: 30 abs_diff_1s features reconstructed"
)

assert len(rolling_features) == 30

print(
    "PASS: 30 rolling_std_5s features reconstructed"
)

assert np.isfinite(
    scores
).all()

print(
    "PASS: All scores finite"
)

assert os.path.exists(
    OUTPUT_PATH
)

print(
    "PASS: End-to-end inference output saved"
)

print(
    "PASS: Existing 25B scores reproduced"
)

print(
    "PASS: Existing 25B predictions reproduced"
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
print("STAGE 25D: COMPLETE")
print("=" * 70)
