import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# STAGE 25F
# VALIDATE REUSABLE CANDIDATE C INFERENCE ENGINE
# ============================================================

BASE_DIR = r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"

INFERENCE_DIR = os.path.join(
    BASE_DIR,
    "scripts",
    "inspection",
    "hai_23_05",
    "25_inference"
)

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

MANIFEST_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_feature_manifest.csv"
)

TEST1_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_test1.csv"
)

REFERENCE_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_end_to_end_inference.csv"
)

OUTPUT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_engine_validation.csv"
)


print("=" * 70)
print("STAGE 25F: REUSABLE CANDIDATE C ENGINE VALIDATION")
print("=" * 70)


# ============================================================
# IMPORT ENGINE
# ============================================================

print("\nLoading reusable inference engine...")

sys.path.insert(0, INFERENCE_DIR)

from candidate_C_inference_engine import (
    CandidateCInferenceEngine
)

print("PASS: Engine imported")


# ============================================================
# CREATE ENGINE
# ============================================================

print("\nInitializing engine...")

engine = CandidateCInferenceEngine(
    MODEL_PATH,
    MANIFEST_PATH
)

print("PASS: Engine initialized")

print(
    f"Engine feature count: "
    f"{len(engine.feature_names)}"
)

assert len(engine.feature_names) == 118


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading Candidate C Test 1...")

test1 = pd.read_csv(
    TEST1_PATH
)

print(
    f"Test 1 rows: {len(test1)}"
)

assert len(test1) == 54000

print(
    "PASS: Test 1 row count"
)


# ============================================================
# RUN REUSABLE ENGINE
# ============================================================

print("\nRunning reusable inference engine...")

engine_output = engine.predict(
    test1
)

print(
    "PASS: Engine inference completed"
)


# ============================================================
# VALIDATE OUTPUT SHAPE
# ============================================================

print("\nValidating engine output...")

assert len(engine_output) == 54000

print(
    "PASS: Output row count = 54,000"
)

expected_columns = [
    "timestamp",
    "anomaly_score",
    "prediction",
    "status"
]

assert list(
    engine_output.columns
) == expected_columns

print(
    "PASS: Output schema"
)


# ============================================================
# VALID ROWS
# ============================================================

valid_output = engine_output[
    engine_output["prediction"].notna()
].copy()

print(
    f"Valid inference rows: {len(valid_output)}"
)

assert len(valid_output) == 53996

print(
    "PASS: 53,996 valid inference rows"
)


# ============================================================
# BOUNDARY ROWS
# ============================================================

boundary_output = engine_output[
    engine_output["prediction"].isna()
].copy()

print(
    f"Boundary rows: {len(boundary_output)}"
)

assert len(boundary_output) == 4

print(
    "PASS: 4 boundary rows preserved"
)

assert (
    boundary_output["status"]
    == "INSUFFICIENT_HISTORY"
).all()

print(
    "PASS: Boundary status correct"
)


# ============================================================
# LOAD REFERENCE
# ============================================================

print(
    "\nLoading proven Stage 25D reference output..."
)

reference = pd.read_csv(
    REFERENCE_PATH
)

print(
    f"Reference rows: {len(reference)}"
)

assert len(reference) == 53996

print(
    "PASS: Reference contains 53,996 valid rows"
)


# ============================================================
# COMPARE TIMESTAMPS
# ============================================================

print("\nComparing timestamps...")

engine_timestamps = (
    valid_output["timestamp"]
    .astype(str)
    .to_numpy()
)

reference_timestamps = (
    reference["timestamp"]
    .astype(str)
    .to_numpy()
)

assert np.array_equal(
    engine_timestamps,
    reference_timestamps
)

print(
    "PASS: Timestamps match exactly"
)


# ============================================================
# COMPARE SCORES
# ============================================================

print("\nComparing anomaly scores...")

engine_scores = (
    valid_output["anomaly_score"]
    .to_numpy()
)

reference_scores = (
    reference["anomaly_score"]
    .to_numpy()
)

score_difference = np.abs(
    engine_scores - reference_scores
)

max_difference = (
    score_difference.max()
)

print(
    f"Maximum score difference: "
    f"{max_difference:.15f}"
)

assert max_difference <= 1e-12

print(
    "PASS: Anomaly scores match exactly"
)


# ============================================================
# COMPARE PREDICTIONS
# ============================================================

print("\nComparing predictions...")

engine_predictions = (
    valid_output["prediction"]
    .astype(int)
    .to_numpy()
)

reference_predictions = (
    reference["prediction"]
    .astype(int)
    .to_numpy()
)

assert np.array_equal(
    engine_predictions,
    reference_predictions
)

print(
    "PASS: Predictions match exactly"
)


# ============================================================
# COMPARE STATUS
# ============================================================

print("\nComparing status...")

engine_status = (
    valid_output["status"]
    .astype(str)
    .to_numpy()
)

reference_status = (
    reference["status"]
    .astype(str)
    .to_numpy()
)

assert np.array_equal(
    engine_status,
    reference_status
)

print(
    "PASS: Status values match exactly"
)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

normal_count = (
    engine_predictions == 0
).sum()

anomaly_count = (
    engine_predictions == 1
).sum()

anomaly_rate = (
    anomaly_count / len(engine_predictions)
)


print("\n" + "=" * 70)
print("ENGINE INFERENCE SUMMARY")
print("=" * 70)

print(
    f"Total input rows       : {len(test1)}"
)

print(
    f"Valid rows             : {len(valid_output)}"
)

print(
    f"Boundary rows          : {len(boundary_output)}"
)

print(
    f"Candidate C features   : "
    f"{len(engine.feature_names)}"
)

print(
    f"Normal predictions     : {normal_count}"
)

print(
    f"Anomaly predictions    : {anomaly_count}"
)

print(
    f"Anomaly rate           : {anomaly_rate:.6f}"
)


# ============================================================
# SAVE VALIDATION OUTPUT
# ============================================================

engine_output.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nValidation output saved:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25F VALIDATION")
print("=" * 70)

assert normal_count == 52091

print(
    "PASS: 52,091 normal predictions"
)

assert anomaly_count == 1905

print(
    "PASS: 1,905 anomaly predictions"
)

assert abs(
    anomaly_rate - 0.035280
) < 1e-6

print(
    "PASS: Anomaly rate = 0.035280"
)

assert max_difference <= 1e-12

print(
    "PASS: Scores reproduce Stage 25D"
)

print(
    "PASS: Predictions reproduce Stage 25D"
)

print(
    "PASS: Status values reproduce Stage 25D"
)

print(
    "PASS: Boundary handling verified"
)

print(
    "PASS: Reusable inference engine validated"
)

print("\n" + "=" * 70)
print("STAGE 25F: COMPLETE")
print("=" * 70)