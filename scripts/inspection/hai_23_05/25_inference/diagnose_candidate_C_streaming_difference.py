import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# STAGE 25I
# DIAGNOSE STREAMING VS BATCH SCORE DIFFERENCES
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


print("=" * 70)
print("STAGE 25I: STREAMING DIFFERENCE DIAGNOSTICS")
print("=" * 70)


# ============================================================
# IMPORT
# ============================================================

sys.path.insert(0, INFERENCE_DIR)

from candidate_C_streaming_engine import (
    CandidateCStreamingEngine
)


# ============================================================
# LOAD
# ============================================================

print("\nLoading data...")

data = pd.read_csv(
    TEST1_PATH
)

reference = pd.read_csv(
    REFERENCE_PATH
)

print(
    f"Test rows: {len(data)}"
)

print(
    f"Reference rows: {len(reference)}"
)


# ============================================================
# ENGINE
# ============================================================

engine = CandidateCStreamingEngine(
    MODEL_PATH,
    MANIFEST_PATH
)

print(
    "PASS: Streaming engine initialized"
)


# ============================================================
# PROCESS CHUNKS
# ============================================================

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


streaming = pd.concat(
    results,
    ignore_index=True
)

streaming_valid = streaming[
    streaming["prediction"].notna()
].copy()

streaming_valid = streaming_valid.reset_index(
    drop=True
)

reference = reference.reset_index(
    drop=True
)


# ============================================================
# SCORE DIFFERENCE
# ============================================================

stream_scores = (
    streaming_valid["anomaly_score"]
    .to_numpy()
)

reference_scores = (
    reference["anomaly_score"]
    .to_numpy()
)

difference = np.abs(
    stream_scores - reference_scores
)

difference_signed = (
    stream_scores - reference_scores
)


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SCORE DIFFERENCE SUMMARY")
print("=" * 70)

print(
    f"Maximum absolute difference : "
    f"{difference.max():.15f}"
)

print(
    f"Mean absolute difference    : "
    f"{difference.mean():.15f}"
)

print(
    f"Median absolute difference  : "
    f"{np.median(difference):.15f}"
)

print(
    f"Rows with difference > 1e-12: "
    f"{(difference > 1e-12).sum()}"
)

print(
    f"Rows with difference > 1e-6 : "
    f"{(difference > 1e-6).sum()}"
)

print(
    f"Rows with difference > 1e-4 : "
    f"{(difference > 1e-4).sum()}"
)


# ============================================================
# FIRST DIFFERENCES
# ============================================================

different_indices = np.where(
    difference > 1e-12
)[0]

print("\n" + "=" * 70)
print("FIRST DIFFERING ROWS")
print("=" * 70)

for index in different_indices[:20]:

    print(
        f"Row {index}: "
        f"timestamp={streaming_valid.iloc[index]['timestamp']}, "
        f"stream={stream_scores[index]:.15f}, "
        f"batch={reference_scores[index]:.15f}, "
        f"diff={difference[index]:.15f}"
    )


# ============================================================
# LARGEST DIFFERENCES
# ============================================================

print("\n" + "=" * 70)
print("LARGEST DIFFERENCES")
print("=" * 70)

largest_indices = np.argsort(
    difference
)[-20:][::-1]

for index in largest_indices:

    print(
        f"Row {index}: "
        f"timestamp={streaming_valid.iloc[index]['timestamp']}, "
        f"stream={stream_scores[index]:.15f}, "
        f"batch={reference_scores[index]:.15f}, "
        f"diff={difference[index]:.15f}"
    )


# ============================================================
# CHUNK BOUNDARY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CHUNK BOUNDARY ANALYSIS")
print("=" * 70)

boundary_rows = []

for boundary in range(
    CHUNK_SIZE,
    len(data),
    CHUNK_SIZE
):

    # The first few rows after a chunk boundary
    # are examined because temporal history may be
    # reconstructed differently there.

    for offset in range(0, 10):

        position = boundary + offset - 4

        if position < 0:
            continue

        if position >= len(streaming_valid):
            continue

        boundary_rows.append(
            position
        )


boundary_rows = sorted(
    set(boundary_rows)
)

boundary_differences = difference[
    boundary_rows
]

non_boundary_mask = np.ones(
    len(difference),
    dtype=bool
)

non_boundary_mask[
    boundary_rows
] = False

non_boundary_differences = difference[
    non_boundary_mask
]


print(
    f"Boundary-related rows examined: "
    f"{len(boundary_rows)}"
)

print(
    f"Boundary max difference: "
    f"{boundary_differences.max():.15f}"
)

print(
    f"Boundary mean difference: "
    f"{boundary_differences.mean():.15f}"
)

print(
    f"Non-boundary max difference: "
    f"{non_boundary_differences.max():.15f}"
)

print(
    f"Non-boundary mean difference: "
    f"{non_boundary_differences.mean():.15f}"
)


# ============================================================
# PREDICTION COMPARISON
# ============================================================

stream_predictions = (
    streaming_valid["prediction"]
    .astype(int)
    .to_numpy()
)

reference_predictions = (
    reference["prediction"]
    .astype(int)
    .to_numpy()
)

prediction_difference = (
    stream_predictions
    !=
    reference_predictions
)

print("\n" + "=" * 70)
print("PREDICTION COMPARISON")
print("=" * 70)

print(
    f"Prediction mismatches: "
    f"{prediction_difference.sum()}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25I COMPLETE")
print("=" * 70)

print(
    "No model changes were made."
)

print(
    "No threshold tuning was performed."
)

print(
    "This stage only diagnoses the source of "
    "streaming-vs-batch differences."
)