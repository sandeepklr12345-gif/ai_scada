from pathlib import Path
import time

import cudf
import cupy as cp
import numpy as np
import joblib

from cuml.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# GPU EXPERIMENT
# CPU vs GPU Isolation Forest Detection Comparison
#
# IMPORTANT:
# Experimental only.
# Does NOT modify the official Stage 23C model.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEMPORAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
)

ISOLATION_DIR = TEMPORAL_DIR / "isolation_forest"

CPU_MODEL_FILE = (
    ISOLATION_DIR
    / "hai_2305_temporal_isolation_forest_baseline.joblib"
)

TRAIN_FILE = (
    TEMPORAL_DIR
    / "hai_2305_training_temporal_model_ready.csv"
)

TEST1_FILE = (
    TEMPORAL_DIR
    / "hai-test1_temporal_model_ready.csv"
)

TEST2_FILE = (
    TEMPORAL_DIR
    / "hai-test2_temporal_model_ready.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GPU EXPERIMENT: CPU vs GPU DETECTION COMPARISON")
print("=" * 70)


# ============================================================
# GPU
# ============================================================

device = cp.cuda.Device()
device.synchronize()

props = cp.cuda.runtime.getDeviceProperties(device.id)
gpu_name = props["name"]

if isinstance(gpu_name, bytes):
    gpu_name = gpu_name.decode()

print(f"GPU: {gpu_name}")


# ============================================================
# LOAD CPU MODEL
# ============================================================

print("\nLoading existing CPU Stage 23C model...")

cpu_model = joblib.load(CPU_MODEL_FILE)

print("PASS: CPU model loaded")
print(f"CPU model: {CPU_MODEL_FILE}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Test 1...")

test1_df = cudf.read_csv(TEST1_FILE)

print(f"Test 1 shape: {test1_df.shape}")

print("\nLoading Test 2...")

test2_df = cudf.read_csv(TEST2_FILE)

print(f"Test 2 shape: {test2_df.shape}")


# ============================================================
# FIND LABEL COLUMN
# ============================================================

def find_label_column(df):
    candidates = ["label", "Label", "LABEL"]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find label column. "
        f"Available columns include: {list(df.columns[-10:])}"
    )


label_col_test1 = find_label_column(test1_df)
label_col_test2 = find_label_column(test2_df)

print(f"\nTest 1 label column: {label_col_test1}")
print(f"Test 2 label column: {label_col_test2}")


# ============================================================
# IDENTIFY FEATURES
# ============================================================

exclude_columns = {
    "Time",
    "timestamp",
    "label",
    "Label",
    "LABEL",
}

feature_columns = [
    col
    for col in test1_df.columns
    if col not in exclude_columns
    and test1_df[col].dtype.kind in "if"
]

print(f"Numeric feature count: {len(feature_columns)}")

if len(feature_columns) != 232:
    raise ValueError(
        f"Expected 232 numeric features, found {len(feature_columns)}"
    )


# ============================================================
# PREPARE TEST DATA
# ============================================================

test1_valid = test1_df.dropna(subset=feature_columns)
test2_valid = test2_df.dropna(subset=feature_columns)

X_test1_gpu = test1_valid[feature_columns]
X_test2_gpu = test2_valid[feature_columns]

y_test1 = (
    test1_valid[label_col_test1]
    .to_numpy()
    .astype(np.int32)
)

y_test2 = (
    test2_valid[label_col_test2]
    .to_numpy()
    .astype(np.int32)
)

print("\nValid rows:")
print(f"Test 1: {len(X_test1_gpu)}")
print(f"Test 2: {len(X_test2_gpu)}")


# ============================================================
# CPU PREDICTIONS
# ============================================================

print("\nGenerating CPU predictions...")

X_test1_cpu = X_test1_gpu.to_pandas()
X_test2_cpu = X_test2_gpu.to_pandas()

cpu_pred_test1_raw = cpu_model.predict(X_test1_cpu)
cpu_pred_test2_raw = cpu_model.predict(X_test2_cpu)

# sklearn IsolationForest:
# normal = 1
# anomaly = -1
cpu_pred_test1 = (cpu_pred_test1_raw == -1).astype(np.int32)
cpu_pred_test2 = (cpu_pred_test2_raw == -1).astype(np.int32)

print(f"CPU Test 1 anomalies: {cpu_pred_test1.sum()}")
print(f"CPU Test 2 anomalies: {cpu_pred_test2.sum()}")


# ============================================================
# TRAIN GPU MODEL
# ============================================================

print("\nTraining GPU Isolation Forest...")

train_df = cudf.read_csv(TRAIN_FILE)

train_valid = train_df.dropna(subset=feature_columns)

X_train_gpu = train_valid[feature_columns]

print(f"GPU training rows: {len(X_train_gpu)}")

cp.cuda.Device().synchronize()
gpu_train_start = time.perf_counter()

gpu_model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
)

gpu_model.fit(X_train_gpu)

cp.cuda.Device().synchronize()
gpu_train_time = time.perf_counter() - gpu_train_start

print(f"GPU training time: {gpu_train_time:.4f} seconds")


# ============================================================
# GPU PREDICTIONS
# ============================================================

print("\nGenerating GPU predictions...")

cp.cuda.Device().synchronize()
gpu_test1_start = time.perf_counter()

gpu_pred_test1_raw = gpu_model.predict(X_test1_gpu)

cp.cuda.Device().synchronize()
gpu_test1_time = time.perf_counter() - gpu_test1_start

cp.cuda.Device().synchronize()
gpu_test2_start = time.perf_counter()

gpu_pred_test2_raw = gpu_model.predict(X_test2_gpu)

cp.cuda.Device().synchronize()
gpu_test2_time = time.perf_counter() - gpu_test2_start

gpu_pred_test1_raw = gpu_pred_test1_raw.to_numpy()
gpu_pred_test2_raw = gpu_pred_test2_raw.to_numpy()

gpu_pred_test1 = (gpu_pred_test1_raw == -1).astype(np.int32)
gpu_pred_test2 = (gpu_pred_test2_raw == -1).astype(np.int32)

print(f"GPU Test 1 anomalies: {gpu_pred_test1.sum()}")
print(f"GPU Test 2 anomalies: {gpu_pred_test2.sum()}")


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate(name, y_true, cpu_pred, gpu_pred):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    cpu_precision = precision_score(
        y_true,
        cpu_pred,
        zero_division=0,
    )

    cpu_recall = recall_score(
        y_true,
        cpu_pred,
        zero_division=0,
    )

    cpu_f1 = f1_score(
        y_true,
        cpu_pred,
        zero_division=0,
    )

    gpu_precision = precision_score(
        y_true,
        gpu_pred,
        zero_division=0,
    )

    gpu_recall = recall_score(
        y_true,
        gpu_pred,
        zero_division=0,
    )

    gpu_f1 = f1_score(
        y_true,
        gpu_pred,
        zero_division=0,
    )

    agreement = np.mean(cpu_pred == gpu_pred)

    print("\nCPU")
    print(f"Precision : {cpu_precision:.6f}")
    print(f"Recall    : {cpu_recall:.6f}")
    print(f"F1        : {cpu_f1:.6f}")

    print("\nCPU confusion matrix")
    print(confusion_matrix(y_true, cpu_pred))

    print("\nGPU")
    print(f"Precision : {gpu_precision:.6f}")
    print(f"Recall    : {gpu_recall:.6f}")
    print(f"F1        : {gpu_f1:.6f}")

    print("\nGPU confusion matrix")
    print(confusion_matrix(y_true, gpu_pred))

    print(f"\nCPU/GPU prediction agreement: {agreement:.6%}")

    print(f"CPU anomalies: {cpu_pred.sum()}")
    print(f"GPU anomalies: {gpu_pred.sum()}")


# ============================================================
# EVALUATE
# ============================================================

evaluate(
    "TEST 1",
    y_test1,
    cpu_pred_test1,
    gpu_pred_test1,
)

evaluate(
    "TEST 2",
    y_test2,
    cpu_pred_test2,
    gpu_pred_test2,
)


# ============================================================
# FINAL TIMING
# ============================================================

print("\n" + "=" * 70)
print("TIMING SUMMARY")
print("=" * 70)

print(f"GPU training       : {gpu_train_time:.4f} s")
print(f"GPU Test 1 predict : {gpu_test1_time:.4f} s")
print(f"GPU Test 2 predict : {gpu_test2_time:.4f} s")


# ============================================================
# GPU MEMORY
# ============================================================

free_mem, total_mem = cp.cuda.runtime.memGetInfo()
used_mem = total_mem - free_mem

print("\nGPU memory:")
print(f"Total : {total_mem / 1024**2:.2f} MiB")
print(f"Used  : {used_mem / 1024**2:.2f} MiB")
print(f"Free  : {free_mem / 1024**2:.2f} MiB")


print("\n" + "=" * 70)
print("CPU vs GPU COMPARISON COMPLETE")
print("=" * 70)
print("No official project model was modified.")
print("=" * 70)
