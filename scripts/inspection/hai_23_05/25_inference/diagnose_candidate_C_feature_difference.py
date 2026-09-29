import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# STAGE 25J
# DIAGNOSE CANDIDATE C FEATURE DIFFERENCES
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


print("=" * 70)
print("STAGE 25J: CANDIDATE C FEATURE DIFFERENCE DIAGNOSTICS")
print("=" * 70)


# ============================================================
# IMPORT
# ============================================================

sys.path.insert(0, INFERENCE_DIR)

from candidate_C_inference_engine import (
    CandidateCInferenceEngine
)

from candidate_C_streaming_engine import (
    CandidateCStreamingEngine
)


# ============================================================
# LOAD
# ============================================================

print("\nLoading Test 1...")

data = pd.read_csv(
    TEST1_PATH
)

print(
    f"Rows: {len(data)}"
)


# ============================================================
# CREATE ENGINES
# ============================================================

model_engine = CandidateCInferenceEngine(
    MODEL_PATH,
    MANIFEST_PATH
)

stream_engine = CandidateCStreamingEngine(
    MODEL_PATH,
    MANIFEST_PATH
)

print(
    "PASS: Engines initialized"
)


# ============================================================
# BATCH FEATURE MATRIX
# ============================================================

print("\nBuilding complete batch feature matrix...")

batch_features = model_engine.build_features(
    data
)

feature_names = model_engine.feature_names

batch_matrix = batch_features[
    feature_names
].copy()

print(
    f"Batch feature matrix: "
    f"{batch_matrix.shape}"
)

assert batch_matrix.shape == (
    54000,
    118
)


# ============================================================
# STREAMING FEATURE MATRIX
# ============================================================

print("\nBuilding streaming feature matrix...")

CHUNK_SIZE = 1000

stream_feature_parts = []

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
    # Reproduce the streaming wrapper's exact state handling.
    # --------------------------------------------------------

    if stream_engine.history is None:

        combined = chunk.copy()

        history_length = 0

    else:

        combined = pd.concat(
            [
                stream_engine.history,
                chunk
            ],
            ignore_index=True
        )

        history_length = len(
            stream_engine.history
        )

    # --------------------------------------------------------
    # Build Candidate C features.
    # --------------------------------------------------------

    combined_features = (
        model_engine.build_features(
            combined
        )
    )

    new_features = combined_features.iloc[
        history_length:
    ].copy()

    stream_feature_parts.append(
        new_features[
            feature_names
        ].reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Update history exactly as streaming engine does.
    # --------------------------------------------------------

    stream_engine.history = (
        combined
        .tail(
            stream_engine.HISTORY_SIZE
        )
        .copy()
        .reset_index(drop=True)
    )


stream_matrix = pd.concat(
    stream_feature_parts,
    ignore_index=True
)

print(
    f"Streaming feature matrix: "
    f"{stream_matrix.shape}"
)

assert stream_matrix.shape == (
    54000,
    118
)


# ============================================================
# COMPARE FEATURE MATRICES
# ============================================================

print("\nComparing feature matrices...")

batch_values = batch_matrix.to_numpy(
    dtype=float
)

stream_values = stream_matrix.to_numpy(
    dtype=float
)

# ------------------------------------------------------------
# Handle NaNs consistently.
# ------------------------------------------------------------

both_nan = (
    np.isnan(batch_values)
    &
    np.isnan(stream_values)
)

numeric_difference = np.abs(
    batch_values - stream_values
)

numeric_difference[
    both_nan
] = 0.0

numeric_difference[
    np.isnan(numeric_difference)
] = np.inf


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE MATRIX COMPARISON")
print("=" * 70)

print(
    f"Maximum feature difference: "
    f"{np.max(numeric_difference):.15f}"
)

print(
    f"Rows with feature differences: "
    f"{np.any(numeric_difference > 1e-12, axis=1).sum()}"
)

print(
    f"Feature cells differing > 1e-12: "
    f"{(numeric_difference > 1e-12).sum()}"
)

print(
    f"Feature cells differing > 1e-6: "
    f"{(numeric_difference > 1e-6).sum()}"
)

print(
    f"Feature cells differing > 1e-4: "
    f"{(numeric_difference > 1e-4).sum()}"
)


# ============================================================
# TOP DIFFERING FEATURES
# ============================================================

feature_max_difference = (
    np.max(
        numeric_difference,
        axis=0
    )
)

feature_difference_count = (
    np.sum(
        numeric_difference > 1e-12,
        axis=0
    )
)


feature_summary = pd.DataFrame({
    "feature": feature_names,
    "max_difference": feature_max_difference,
    "difference_count": feature_difference_count
})

feature_summary = (
    feature_summary
    .sort_values(
        "max_difference",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("TOP DIFFERING FEATURES")
print("=" * 70)

print(
    feature_summary.head(20).to_string(
        index=False
    )
)


# ============================================================
# FIRST DIFFERING ROWS
# ============================================================

row_difference = np.any(
    numeric_difference > 1e-12,
    axis=1
)

different_rows = np.where(
    row_difference
)[0]


print("\n" + "=" * 70)
print("FIRST DIFFERING ROWS")
print("=" * 70)

for row in different_rows[:20]:

    max_feature_index = np.argmax(
        numeric_difference[row]
    )

    feature = feature_names[
        max_feature_index
    ]

    print(
        f"Row {row}: "
        f"timestamp={data.iloc[row]['timestamp']}, "
        f"feature={feature}, "
        f"batch={batch_values[row, max_feature_index]:.15f}, "
        f"stream={stream_values[row, max_feature_index]:.15f}, "
        f"diff={numeric_difference[row, max_feature_index]:.15f}"
    )


# ============================================================
# CHUNK BOUNDARY FOCUS
# ============================================================

print("\n" + "=" * 70)
print("CHUNK BOUNDARY FEATURE CHECK")
print("=" * 70)

boundary_positions = []

for boundary in range(
    1000,
    len(data),
    1000
):

    # Inspect 5 rows before and 10 rows after
    # each chunk boundary.

    for offset in range(
        -5,
        11
    ):

        position = boundary + offset

        if (
            position >= 0
            and position < len(data)
        ):

            boundary_positions.append(
                position
            )


boundary_positions = sorted(
    set(boundary_positions)
)

boundary_diff = numeric_difference[
    boundary_positions
]

print(
    f"Boundary-related rows examined: "
    f"{len(boundary_positions)}"
)

print(
    f"Boundary max feature difference: "
    f"{boundary_diff.max():.15f}"
)

print(
    f"Boundary differing cells: "
    f"{(boundary_diff > 1e-12).sum()}"
)


# ============================================================
# SAVE FEATURE DIAGNOSTICS
# ============================================================

OUTPUT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_streaming_feature_difference.csv"
)

feature_summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nFeature diagnostic saved:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25J COMPLETE")
print("=" * 70)

print(
    "No model changes."
)

print(
    "No threshold tuning."
)

print(
    "No feature selection."
)

print(
    "This stage only compares the actual "
    "118-feature matrices."
)