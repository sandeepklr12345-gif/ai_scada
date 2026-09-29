from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HAI_INFERENCE_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "inference"
    / "hai_candidate_C_inference.py"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
    / "final_candidate"
    / "hai_2305_candidate_C_isolation_forest.joblib"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
    / "final_candidate"
    / "hai_2305_candidate_C_feature_manifest.csv"
)

HAI_DATA = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "model_ready"
    / "hai-test1_model_ready.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_batch_stream_comparison.csv"
)


# ============================================================
# LOAD HAI INFERENCE MODULE
# ============================================================

spec = importlib.util.spec_from_file_location(
    "hai_candidate_C_inference",
    HAI_INFERENCE_PATH
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Could not load: {HAI_INFERENCE_PATH}"
    )

hai_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hai_module)

HAICandidateCInference = (
    hai_module.HAICandidateCInference
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("HAI BATCH VS STREAMING EXACT COMPARISON")
print("=" * 80)

print()
print("Loading HAI Test 1...")

df = pd.read_csv(HAI_DATA)

print(f"Rows loaded : {len(df)}")
print(f"Columns     : {len(df.columns)}")


# ============================================================
# ORIGINAL 58 FEATURES
# ============================================================

hai_features = [
    column
    for column in df.columns
    if column not in ["timestamp", "label"]
]

if len(hai_features) != 58:
    raise RuntimeError(
        f"Expected 58 original features, "
        f"found {len(hai_features)}"
    )

print(f"Original features : {len(hai_features)}")


# ============================================================
# LOAD TWO INDEPENDENT ENGINE INSTANCES
# ============================================================

print()
print("Loading batch engine...")

batch_engine = HAICandidateCInference(
    str(MODEL_PATH),
    str(MANIFEST_PATH)
)

print("Loading streaming engine...")

stream_engine = HAICandidateCInference(
    str(MODEL_PATH),
    str(MANIFEST_PATH)
)

print("Both engines loaded.")


# ============================================================
# BATCH INFERENCE
# ============================================================

print()
print("=" * 80)
print("RUNNING BATCH INFERENCE")
print("=" * 80)

batch_input = df[
    hai_features
].copy()

batch_output = batch_engine.predict_batch(
    batch_input
).reset_index(drop=True)

print(
    f"Batch output rows : {len(batch_output)}"
)


# ============================================================
# STREAMING INFERENCE
# ============================================================

print()
print("=" * 80)
print("RUNNING STATEFUL STREAMING INFERENCE")
print("=" * 80)

stream_results = []

for i in range(len(df)):

    row = df.iloc[i]

    row_input = pd.DataFrame(
        [row[hai_features].to_dict()]
    )

    output = stream_engine.predict_stream(
        row_input
    )

    stream_results.append(
        output.iloc[0].to_dict()
    )

    if (i + 1) % 5000 == 0:
        print(
            f"Processed "
            f"{i + 1}/{len(df)}"
        )


stream_output = pd.DataFrame(
    stream_results
).reset_index(drop=True)

print(
    f"Streaming output rows : "
    f"{len(stream_output)}"
)


# ============================================================
# ALIGN OUTPUT
# ============================================================

compare = pd.DataFrame({
    "row_index": np.arange(len(df)),
    "timestamp": df["timestamp"],
    "actual_label": df["label"],

    "batch_score":
        batch_output["anomaly_score"],

    "stream_score":
        stream_output["anomaly_score"],

    "batch_prediction":
        batch_output["prediction"],

    "stream_prediction":
        stream_output["prediction"],

    "batch_status":
        batch_output["status"],

    "stream_status":
        stream_output["status"],
})


compare["score_difference"] = (
    compare["batch_score"]
    - compare["stream_score"]
).abs()

compare["prediction_match"] = (
    compare["batch_prediction"]
    == compare["stream_prediction"]
)

compare["status_match"] = (
    compare["batch_status"]
    == compare["stream_status"]
)


# ============================================================
# FIND DIFFERENCES
# ============================================================

prediction_mismatches = compare[
    ~compare["prediction_match"]
].copy()

status_mismatches = compare[
    ~compare["status_match"]
].copy()

score_mismatches = compare[
    compare["score_difference"] > 1e-12
].copy()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("EXACT COMPARISON RESULTS")
print("=" * 80)

print(
    f"Rows compared              : "
    f"{len(compare)}"
)

print(
    f"Prediction mismatches      : "
    f"{len(prediction_mismatches)}"
)

print(
    f"Status mismatches          : "
    f"{len(status_mismatches)}"
)

print(
    f"Score differences > 1e-12  : "
    f"{len(score_mismatches)}"
)

if len(score_mismatches) > 0:

    print()
    print(
        "Maximum score difference : "
        f"{compare['score_difference'].max():.18f}"
    )

    print(
        "Mean score difference    : "
        f"{compare['score_difference'].mean():.18f}"
    )


# ============================================================
# SHOW EXACT PREDICTION MISMATCHES
# ============================================================

if len(prediction_mismatches) > 0:

    print()
    print("=" * 80)
    print("PREDICTION MISMATCH DETAILS")
    print("=" * 80)

    print(
        prediction_mismatches[
            [
                "row_index",
                "timestamp",
                "actual_label",
                "batch_score",
                "stream_score",
                "score_difference",
                "batch_prediction",
                "stream_prediction",
                "batch_status",
                "stream_status",
            ]
        ].to_string(index=False)
    )

else:

    print()
    print("No prediction mismatches found.")


# ============================================================
# HISTORY CHECK
# ============================================================

expected_history = stream_engine.HISTORY_SIZE

stream_history_rows = (
    compare["stream_status"]
    == "INSUFFICIENT_HISTORY"
).sum()

batch_history_rows = (
    compare["batch_status"]
    == "INSUFFICIENT_HISTORY"
).sum()

print()
print("=" * 80)
print("HISTORY CHECK")
print("=" * 80)

print(
    f"Expected history rows : "
    f"{expected_history}"
)

print(
    f"Batch history rows    : "
    f"{batch_history_rows}"
)

print(
    f"Stream history rows   : "
    f"{stream_history_rows}"
)


# ============================================================
# SAVE COMPARISON
# ============================================================

compare.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Comparison saved to:")
print(OUTPUT_FILE)


# ============================================================
# FINAL STATUS
# ============================================================

if len(status_mismatches) != 0:
    raise RuntimeError(
        "Batch and streaming status outputs differ."
    )

print()
print("=" * 80)

if len(prediction_mismatches) == 0:
    print("BATCH VS STREAMING: EXACT MATCH")
else:
    print(
        "BATCH VS STREAMING: "
        f"{len(prediction_mismatches)} "
        "PREDICTION DIFFERENCE(S) FOUND"
    )

print("=" * 80)