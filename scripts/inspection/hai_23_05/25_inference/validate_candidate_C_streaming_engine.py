from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# STAGE 25H
# STATEFUL STREAMING ENGINE VALIDATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE_DIR = str(PROJECT_ROOT)

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
    "hai_2305_candidate_C_streaming_validation.csv"
)


print("=" * 70)
print("STAGE 25H: STATEFUL STREAMING ENGINE VALIDATION")
print("=" * 70)


# ============================================================
# IMPORT
# ============================================================

sys.path.insert(0, INFERENCE_DIR)

from candidate_C_streaming_engine import (
    CandidateCStreamingEngine
)


print("\nLoading stateful streaming engine...")


# ============================================================
# INITIALIZE
# ============================================================

engine = CandidateCStreamingEngine(
    MODEL_PATH,
    MANIFEST_PATH
)

print("PASS: Streaming engine initialized")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading Candidate C Test 1...")

data = pd.read_csv(
    TEST1_PATH
)

assert len(data) == 54000

print(
    "PASS: 54,000 Test 1 rows loaded"
)


# ============================================================
# LOAD REFERENCE
# ============================================================

print("\nLoading validated batch reference...")

reference = pd.read_csv(
    REFERENCE_PATH
)

assert len(reference) == 53996

print(
    "PASS: 53,996 reference inference rows loaded"
)


# ============================================================
# STREAMING PROCESSING
# ============================================================

print("\nProcessing Test 1 in streaming chunks...")

CHUNK_SIZE = 1000

results = []

for start in range(
    0,
    len(data),
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        len(data)
    )

    chunk = data.iloc[
        start:end
    ].copy()

    result = engine.predict_chunk(
        chunk
    )

    results.append(result)

    print(
        f"Processed rows "
        f"{start + 1:,} - {end:,}"
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

streaming_output = pd.concat(
    results,
    ignore_index=True
)


print(
    f"\nStreaming output rows: "
    f"{len(streaming_output)}"
)

assert len(streaming_output) == 54000

print(
    "PASS: Streaming output contains 54,000 rows"
)


# ============================================================
# VALID ROWS
# ============================================================

valid_output = streaming_output[
    streaming_output["prediction"].notna()
].copy()

print(
    f"Valid inference rows: "
    f"{len(valid_output)}"
)

assert len(valid_output) == 53996

print(
    "PASS: Exactly 53,996 valid rows"
)


# ============================================================
# BOUNDARY ROWS
# ============================================================

boundary_output = streaming_output[
    streaming_output["prediction"].isna()
].copy()

print(
    f"Boundary rows: {len(boundary_output)}"
)

assert len(boundary_output) == 4

print(
    "PASS: Exactly 4 boundary rows"
)

assert (
    boundary_output["status"]
    == "INSUFFICIENT_HISTORY"
).all()

print(
    "PASS: Boundary status is correct"
)


# ============================================================
# COMPARE TIMESTAMPS
# ============================================================

print("\nComparing timestamps with batch reference...")

assert np.array_equal(
    valid_output["timestamp"]
    .astype(str)
    .to_numpy(),

    reference["timestamp"]
    .astype(str)
    .to_numpy()
)

print(
    "PASS: Timestamps match exactly"
)


# ============================================================
# COMPARE SCORES
# ============================================================

print("\nComparing anomaly scores...")

stream_scores = (
    valid_output["anomaly_score"]
    .to_numpy()
)

reference_scores = (
    reference["anomaly_score"]
    .to_numpy()
)

score_difference = np.abs(
    stream_scores - reference_scores
)

max_difference = (
    score_difference.max()
)

mean_difference = (
    score_difference.mean()
)

print(
    f"Maximum score difference: "
    f"{max_difference:.15f}"
)

print(
    f"Mean score difference: "
    f"{mean_difference:.15f}"
)

assert max_difference <= 1e-12

print(
    "PASS: Streaming scores reproduce batch scores"
)


# ============================================================
# COMPARE PREDICTIONS
# ============================================================

print("\nComparing predictions...")

stream_predictions = (
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
    stream_predictions,
    reference_predictions
)

print(
    "PASS: Streaming predictions match batch predictions"
)


# ============================================================
# COMPARE STATUS
# ============================================================

print("\nComparing status values...")

stream_status = (
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
    stream_status,
    reference_status
)

print(
    "PASS: Streaming status values match batch status"
)


# ============================================================
# COUNTS
# ============================================================

normal_count = (
    stream_predictions == 0
).sum()

anomaly_count = (
    stream_predictions == 1
).sum()

anomaly_rate = (
    anomaly_count / len(stream_predictions)
)


print("\n" + "=" * 70)
print("STREAMING INFERENCE SUMMARY")
print("=" * 70)

print(
    f"Input rows            : {len(data)}"
)

print(
    f"Valid rows            : {len(valid_output)}"
)

print(
    f"Boundary rows         : {len(boundary_output)}"
)

print(
    f"Normal predictions    : {normal_count}"
)

print(
    f"Anomaly predictions   : {anomaly_count}"
)

print(
    f"Anomaly rate          : {anomaly_rate:.6f}"
)


# ============================================================
# SAVE
# ============================================================

streaming_output.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\nStreaming validation output saved:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25H VALIDATION")
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
    "PASS: Maximum score difference = 0"
)

print(
    "PASS: Stateful history preserved across chunks"
)

print(
    "PASS: Streaming inference reproduces batch inference"
)

print(
    "PASS: Candidate C is stream-consistent"
)

print("\n" + "=" * 70)
print("STAGE 25H: COMPLETE")
print("=" * 70)
