from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import importlib.util

HAI_INFERENCE_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "inference"
    / "hai_candidate_C_inference.py"
)

spec = importlib.util.spec_from_file_location(
    "hai_candidate_C_inference",
    HAI_INFERENCE_PATH
)

hai_module = importlib.util.module_from_spec(spec)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Could not load HAI inference module from: "
        f"{HAI_INFERENCE_PATH}"
    )

spec.loader.exec_module(hai_module)

HAICandidateCInference = (
    hai_module.HAICandidateCInference
)


# ============================================================
# PATHS
# ============================================================

HAI_DATA = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "model_ready"
    / "hai-test1_model_ready.csv"
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
    / "hai_test1_full_runtime_validation.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("HAI FULL RUNTIME VALIDATION")
print("=" * 80)

print()
print("Loading HAI Test 1...")

df = pd.read_csv(HAI_DATA)

print(f"Rows loaded : {len(df)}")
print(f"Columns     : {len(df.columns)}")


# ============================================================
# INPUT FEATURES
# ============================================================

hai_features = [
    column
    for column in df.columns
    if column not in ["timestamp", "label"]
]

if len(hai_features) != 58:
    raise RuntimeError(
        f"Expected 58 HAI features, "
        f"found {len(hai_features)}"
    )

print(f"Original HAI features : {len(hai_features)}")


# ============================================================
# LOAD DETECTOR
# ============================================================

print()
print("Loading Candidate C inference engine...")

detector = HAICandidateCInference(
    str(MODEL_PATH),
    str(MANIFEST_PATH)
)

print("Candidate C engine loaded.")
print(f"History size : {detector.HISTORY_SIZE}")


# ============================================================
# STATEFUL RUNTIME INFERENCE
# ============================================================

print()
print("=" * 80)
print("STARTING FULL STATEFUL RUNTIME")
print("=" * 80)

results = []

for i in range(len(df)):

    row = df.iloc[i]

    input_row = pd.DataFrame(
        [row[hai_features].to_dict()]
    )

    output = detector.predict_stream(
        input_row
    )

    result = output.iloc[0]

    results.append({
        "row_index": i,
        "timestamp": df.iloc[i]["timestamp"],
        "actual_label": df.iloc[i]["label"],
        "anomaly_score": result["anomaly_score"],
        "prediction": result["prediction"],
        "status": result["status"],
    })

    if (i + 1) % 5000 == 0:
        print(
            f"Processed "
            f"{i + 1}/{len(df)} rows"
        )


results_df = pd.DataFrame(results)


# ============================================================
# HISTORY VALIDATION
# ============================================================

print()
print("=" * 80)
print("HISTORY VALIDATION")
print("=" * 80)

insufficient_count = (
    results_df["status"]
    == "INSUFFICIENT_HISTORY"
).sum()

expected_history = detector.HISTORY_SIZE

print(
    f"Expected insufficient-history rows : "
    f"{expected_history}"
)

print(
    f"Actual insufficient-history rows   : "
    f"{insufficient_count}"
)

if insufficient_count != expected_history:
    raise RuntimeError(
        "Unexpected history initialization."
    )


# ============================================================
# VALID INFERENCE ROWS
# ============================================================

valid_mask = (
    results_df["status"]
    != "INSUFFICIENT_HISTORY"
)

valid_df = results_df.loc[
    valid_mask
].copy()

print()
print(
    f"Valid inference rows : {len(valid_df)}"
)


# ============================================================
# PREDICTION VALIDATION
# ============================================================

prediction_values = (
    valid_df["prediction"]
    .dropna()
    .unique()
)

invalid_predictions = [
    value
    for value in prediction_values
    if value not in [0, 1]
]

if invalid_predictions:
    raise RuntimeError(
        f"Invalid prediction values: "
        f"{invalid_predictions}"
    )


# ============================================================
# METRICS
# ============================================================

y_true = valid_df[
    "actual_label"
].astype(int)

y_pred = valid_df[
    "prediction"
].astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

accuracy = accuracy_score(
    y_true,
    y_pred
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("FULL RUNTIME RESULTS")
print("=" * 80)

print(f"Total rows             : {len(df)}")
print(f"History rows           : {insufficient_count}")
print(f"Valid inference rows   : {len(valid_df)}")

print()
print(f"Actual NORMAL          : {(y_true == 0).sum()}")
print(f"Actual ANOMALY         : {(y_true == 1).sum()}")

print()
print(f"Predicted NORMAL       : {(y_pred == 0).sum()}")
print(f"Predicted ANOMALY      : {(y_pred == 1).sum()}")

print()
print(f"True Negatives         : {tn}")
print(f"False Positives        : {fp}")
print(f"False Negatives        : {fn}")
print(f"True Positives         : {tp}")

print()
print(f"Precision              : {precision:.6f}")
print(f"Recall                 : {recall:.6f}")
print(f"F1 Score               : {f1:.6f}")
print(f"Accuracy               : {accuracy:.6f}")


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Output saved to:")
print(OUTPUT_FILE)


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(results_df) != len(df):
    raise RuntimeError(
        "Runtime output row count mismatch."
    )

if insufficient_count != expected_history:
    raise RuntimeError(
        "History validation failed."
    )

if not np.isfinite(
    valid_df["anomaly_score"].astype(float)
).all():
    raise RuntimeError(
        "Invalid anomaly scores detected."
    )

print()
print("=" * 80)
print("HAI FULL RUNTIME VALIDATION: PASS")
print("=" * 80)