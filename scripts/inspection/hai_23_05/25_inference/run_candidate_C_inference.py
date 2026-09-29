import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# STAGE 25A: FINAL CANDIDATE C INFERENCE PIPELINE
# ============================================================

BASE_DIR = r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"

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

TRAIN_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_training.csv"
)

OUTPUT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_inference_test.csv"
)


print("=" * 70)
print("STAGE 25A: FINAL CANDIDATE C INFERENCE PIPELINE")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained Candidate C detector...")

model = joblib.load(MODEL_PATH)

print("PASS: Model loaded")


# ============================================================
# LOAD FEATURE SCHEMA
# ============================================================

print("\nLoading Candidate C feature schema...")

training_data = pd.read_csv(TRAIN_PATH)

feature_columns = [
    c for c in training_data.columns
    if c != "timestamp"
]

assert len(feature_columns) == 118

print(f"PASS: Candidate C feature count = {len(feature_columns)}")


# ============================================================
# CREATE INFERENCE SAMPLE
# ============================================================

print("\nCreating inference sample...")

# Use the first valid training row only as a schema/inference test.
sample = training_data[
    feature_columns
].copy()

valid_mask = sample.notna().all(axis=1)

sample = sample.loc[valid_mask].head(1).copy()

assert len(sample) == 1

print("PASS: Valid inference sample created")


# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\nValidating inference feature structure...")

assert list(sample.columns) == feature_columns

print("PASS: Feature ordering matches Candidate C")

assert sample.shape == (1, 118)

print("PASS: Inference matrix shape = 1 x 118")

assert sample.dtypes.apply(
    lambda dtype: np.issubdtype(dtype, np.number)
).all()

print("PASS: All inference features are numeric")

assert np.isfinite(
    sample.to_numpy()
).all()

print("PASS: No NaN or infinite values")


# ============================================================
# RUN INFERENCE
# ============================================================

print("\nRunning Candidate C inference...")

X = sample.to_numpy()

score = float(
    model.decision_function(X)[0]
)

prediction = int(
    score < 0
)

if prediction == 1:
    status = "ANOMALY"
else:
    status = "NORMAL"


print("PASS: Inference completed")


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("INFERENCE RESULT")
print("=" * 70)

print(f"Feature count : {X.shape[1]}")
print(f"Anomaly score : {score:.12f}")
print(f"Decision rule : score < 0")
print(f"Prediction    : {prediction}")
print(f"Status        : {status}")


# ============================================================
# SAVE RESULT
# ============================================================

result = pd.DataFrame([
    {
        "feature_count": X.shape[1],
        "anomaly_score": score,
        "prediction": prediction,
        "status": status
    }
])

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25A VALIDATION")
print("=" * 70)

assert X.shape == (1, 118)

print("PASS: 118-feature inference matrix")

assert np.isfinite(score)

print("PASS: Anomaly score is finite")

assert prediction in [0, 1]

print("PASS: Prediction is valid")

assert status in ["NORMAL", "ANOMALY"]

print("PASS: Status is valid")

assert os.path.exists(OUTPUT_PATH)

print("PASS: Inference result saved")

print("PASS: No model retraining")
print("PASS: No threshold tuning")
print("PASS: No feature modification")


print("\nOutput:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STAGE 25A: COMPLETE")
print("=" * 70)