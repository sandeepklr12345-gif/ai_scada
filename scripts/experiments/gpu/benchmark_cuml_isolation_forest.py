from pathlib import Path
import time

import cudf
import cupy as cp
from cuml.ensemble import IsolationForest


# ============================================================
# GPU EXPERIMENT
# HAI 23.05 - cuML Isolation Forest Benchmark
#
# IMPORTANT:
# This is an experimental GPU benchmark.
# It does NOT modify or replace the official Stage 23C model.
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

TRAIN_FILE = TEMPORAL_DIR / "hai_2305_training_temporal_model_ready.csv"
TEST1_FILE = TEMPORAL_DIR / "hai-test1_temporal_model_ready.csv"
TEST2_FILE = TEMPORAL_DIR / "hai-test2_temporal_model_ready.csv"

OUTPUT_DIR = TEMPORAL_DIR / "isolation_forest" / "gpu_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = (
    OUTPUT_DIR
    / "hai_2305_cuml_isolation_forest_experimental.joblib"
)


# ============================================================
# GPU INFORMATION
# ============================================================

print("=" * 70)
print("GPU EXPERIMENT: cuML ISOLATION FOREST")
print("=" * 70)

device = cp.cuda.Device()
device.synchronize()

props = cp.cuda.runtime.getDeviceProperties(device.id)

gpu_name = props["name"]
if isinstance(gpu_name, bytes):
    gpu_name = gpu_name.decode()

print(f"GPU device       : {gpu_name}")
print(f"GPU device ID    : {device.id}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

start = time.perf_counter()

train_df = cudf.read_csv(TRAIN_FILE)

load_train_time = time.perf_counter() - start

print(f"Training shape   : {train_df.shape}")
print(f"Training load    : {load_train_time:.4f} seconds")


print("\nLoading Test 1...")

start = time.perf_counter()

test1_df = cudf.read_csv(TEST1_FILE)

load_test1_time = time.perf_counter() - start

print(f"Test 1 shape     : {test1_df.shape}")
print(f"Test 1 load      : {load_test1_time:.4f} seconds")


print("\nLoading Test 2...")

start = time.perf_counter()

test2_df = cudf.read_csv(TEST2_FILE)

load_test2_time = time.perf_counter() - start

print(f"Test 2 shape     : {test2_df.shape}")
print(f"Test 2 load      : {load_test2_time:.4f} seconds")


# ============================================================
# IDENTIFY NUMERIC FEATURES
# ============================================================

print("\nIdentifying numeric features...")

exclude_columns = {
    "Time",
    "timestamp",
    "label",
    "Label",
}

feature_columns = [
    col
    for col in train_df.columns
    if col not in exclude_columns
    and train_df[col].dtype.kind in "if"
]

print(f"Numeric features : {len(feature_columns)}")

if len(feature_columns) != 232:
    raise ValueError(
        f"Expected 232 numeric features, "
        f"found {len(feature_columns)}"
    )


# ============================================================
# REMOVE BOUNDARY ROWS
# ============================================================

print("\nPreparing training data...")

X_train = train_df[feature_columns].dropna()

print(f"Valid training rows: {len(X_train)}")


print("\nPreparing Test 1...")

X_test1 = test1_df[feature_columns].dropna()

print(f"Valid Test 1 rows: {len(X_test1)}")


print("\nPreparing Test 2...")

X_test2 = test2_df[feature_columns].dropna()

print(f"Valid Test 2 rows: {len(X_test2)}")


# ============================================================
# GPU ISOLATION FOREST
# ============================================================

print("\nTraining cuML Isolation Forest...")

cp.cuda.Device().synchronize()
train_start = time.perf_counter()

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
)

model.fit(X_train)

cp.cuda.Device().synchronize()
train_time = time.perf_counter() - train_start

print(f"GPU training time: {train_time:.4f} seconds")


# ============================================================
# TEST 1 PREDICTION
# ============================================================

print("\nPredicting Test 1...")

cp.cuda.Device().synchronize()
test1_start = time.perf_counter()

pred_test1 = model.predict(X_test1)

cp.cuda.Device().synchronize()
test1_predict_time = time.perf_counter() - test1_start

pred_test1_values = pred_test1.to_numpy()

test1_anomalies = int((pred_test1_values == -1).sum())

print(f"Test 1 prediction time : {test1_predict_time:.4f} seconds")
print(f"Test 1 predictions      : {len(pred_test1_values)}")
print(f"Test 1 anomalies        : {test1_anomalies}")


# ============================================================
# TEST 2 PREDICTION
# ============================================================

print("\nPredicting Test 2...")

cp.cuda.Device().synchronize()
test2_start = time.perf_counter()

pred_test2 = model.predict(X_test2)

cp.cuda.Device().synchronize()
test2_predict_time = time.perf_counter() - test2_start

pred_test2_values = pred_test2.to_numpy()

test2_anomalies = int((pred_test2_values == -1).sum())

print(f"Test 2 prediction time : {test2_predict_time:.4f} seconds")
print(f"Test 2 predictions      : {len(pred_test2_values)}")
print(f"Test 2 anomalies        : {test2_anomalies}")


# ============================================================
# GPU MEMORY
# ============================================================

free_mem, total_mem = cp.cuda.runtime.memGetInfo()

used_mem = total_mem - free_mem

print("\nGPU memory:")
print(f"Total GPU memory : {total_mem / 1024**2:.2f} MiB")
print(f"Used GPU memory  : {used_mem / 1024**2:.2f} MiB")
print(f"Free GPU memory  : {free_mem / 1024**2:.2f} MiB")


# ============================================================
# SAVE EXPERIMENTAL MODEL
# ============================================================

print("\nSaving experimental GPU model...")

# cuML models are intentionally saved separately from
# the official scikit-learn model.
import joblib

joblib.dump(model, MODEL_FILE)

print(f"GPU model saved: {MODEL_FILE}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GPU EXPERIMENT SUMMARY")
print("=" * 70)

print(f"GPU                  : {gpu_name}")
print(f"Features             : {len(feature_columns)}")
print(f"Training rows        : {len(X_train)}")
print(f"Test 1 rows          : {len(X_test1)}")
print(f"Test 2 rows          : {len(X_test2)}")
print(f"Training time        : {train_time:.4f} s")
print(f"Test 1 prediction    : {test1_predict_time:.4f} s")
print(f"Test 2 prediction    : {test2_predict_time:.4f} s")
print(f"Test 1 anomalies     : {test1_anomalies}")
print(f"Test 2 anomalies     : {test2_anomalies}")
print("=" * 70)
print("GPU EXPERIMENT COMPLETE")
print("=" * 70)