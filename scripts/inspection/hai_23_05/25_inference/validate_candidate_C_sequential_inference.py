from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# STAGE 25G
# SEQUENTIAL INFERENCE VALIDATION
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


print("=" * 70)
print("STAGE 25G: SEQUENTIAL INFERENCE VALIDATION")
print("=" * 70)


# ============================================================
# IMPORT ENGINE
# ============================================================

sys.path.insert(0, INFERENCE_DIR)

from candidate_C_inference_engine import (
    CandidateCInferenceEngine
)


# ============================================================
# INITIALIZE ENGINE
# ============================================================

print("\nInitializing inference engine...")

engine = CandidateCInferenceEngine(
    MODEL_PATH,
    MANIFEST_PATH
)

print("PASS: Engine initialized")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Test 1...")

data = pd.read_csv(
    TEST1_PATH
)

print(
    f"Rows loaded: {len(data)}"
)

assert len(data) == 54000

print(
    "PASS: 54,000 rows loaded"
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
    "PASS: Reference contains 53,996 valid rows"
)


# ============================================================
# SEQUENTIAL CHUNK PROCESSING
# ============================================================

print("\nProcessing data sequentially...")

CHUNK_SIZE = 1000

chunks = []

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

    # --------------------------------------------------------
    # IMPORTANT:
    # Each chunk is currently passed independently.
    # This intentionally tests whether temporal history
    # is preserved or lost between chunks.
    # --------------------------------------------------------

    result = engine.predict(
        chunk
    )

    chunks.append(result)

    print(
        f"Processed rows "
        f"{start + 1:,} - {end:,}"
    )


# ============================================================
# COMBINE
# ============================================================

sequential_output = pd.concat(
    chunks,
    ignore_index=True
)


print(
    f"\nSequential output rows: "
    f"{len(sequential_output)}"
)

assert len(sequential_output) == 54000

print(
    "PASS: Sequential output row count"
)


# ============================================================
# COUNT VALID RESULTS
# ============================================================

valid_sequential = sequential_output[
    sequential_output["prediction"].notna()
].copy()

print(
    f"Sequential valid rows: "
    f"{len(valid_sequential)}"
)


# ============================================================
# EXPECTED RESULT
# ============================================================

print("\nExpected valid rows from batch: 53,996")

if len(valid_sequential) == 53996:

    print(
        "PASS: Sequential valid-row count matches"
    )

else:

    print(
        "WARNING: Sequential valid-row count differs"
    )

    print(
        "This may indicate history is being reset "
        "between chunks."
    )


# ============================================================
# TEMPORAL HISTORY CHECK
# ============================================================

print("\nChecking chunk boundaries...")

expected_boundary_positions = [
    1000,
    2000,
    3000,
    4000,
    5000
]

for position in expected_boundary_positions:

    if position >= len(
        sequential_output
    ):
        continue

    row = sequential_output.iloc[position]

    print(
        f"Boundary row {position + 1}: "
        f"status={row['status']}"
    )


# ============================================================
# COMPARE AVAILABLE RESULTS
# ============================================================

print("\nComparing sequential results with batch reference...")

# Convert timestamps to strings
sequential_timestamps = (
    valid_sequential["timestamp"]
    .astype(str)
    .tolist()
)

reference_timestamps = (
    reference["timestamp"]
    .astype(str)
    .tolist()
)


# ------------------------------------------------------------
# Only compare the common valid region.
# ------------------------------------------------------------

common_count = min(
    len(valid_sequential),
    len(reference)
)

sequential_common = (
    valid_sequential
    .iloc[:common_count]
    .reset_index(drop=True)
)

reference_common = (
    reference
    .iloc[:common_count]
    .reset_index(drop=True)
)


# ============================================================
# SCORE DIFFERENCE
# ============================================================

if common_count > 0:

    score_diff = np.abs(
        sequential_common["anomaly_score"].to_numpy()
        -
        reference_common["anomaly_score"].to_numpy()
    )

    print(
        f"Common valid rows compared: "
        f"{common_count}"
    )

    print(
        f"Maximum score difference: "
        f"{score_diff.max():.15f}"
    )

    print(
        f"Mean score difference: "
        f"{score_diff.mean():.15f}"
    )


# ============================================================
# FINAL OBSERVATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25G RESULT")
print("=" * 70)

print(
    "This stage intentionally tests whether the current "
    "batch-oriented engine preserves temporal history "
    "across streaming chunks."
)

print(
    "\nIf chunk boundaries create additional "
    "'INSUFFICIENT_HISTORY' rows or score differences, "
    "that is expected and tells us the engine needs "
    "a persistent rolling history buffer before live deployment."
)

print(
    "\nDo NOT modify the engine based on this run yet."
)

print("\n" + "=" * 70)
print("STAGE 25G: TEST COMPLETE")
print("=" * 70)
