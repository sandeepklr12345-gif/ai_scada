import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# STAGE 25B: FINAL CANDIDATE C BATCH INFERENCE
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

TEST1_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_test1.csv"
)

OUTPUT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_batch_inference_test.csv"
)


print("=" * 70)
print("STAGE 25B: FINAL CANDIDATE C BATCH INFERENCE")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Candidate C detector...")

model = joblib.load(MODEL_PATH)

print("PASS: Model loaded")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Candidate C Test 1 data...")

df = pd.read_csv(TEST1_PATH)

print(f"Rows loaded: {len(df)}")


# ============================================================
# IDENTIFY FEATURES
# ============================================================

feature_columns = [
    c for c in df.columns
    if c not in ["timestamp", "label"]
]

assert len(feature_columns) == 118

print(
    f"PASS: Candidate C feature count = "
    f"{len(feature_columns)}"
)


# ============================================================
# SELECT VALID ROWS
# ============================================================

X = df[feature_columns]

valid_mask = X.notna().all(axis=1)

X_valid = X.loc[valid_mask].copy()

timestamps = df.loc[
    valid_mask,
    "timestamp"
].reset_index(drop=True)

print(
    f"Valid inference rows: {len(X_valid)}"
)


# ============================================================
# VALIDATE INPUT
# ============================================================

assert X_valid.shape[1] == 118

print("PASS: Input matrix has 118 features")

assert X_valid.dtypes.apply(
    lambda dtype: np.issubdtype(
        dtype,
        np.number
    )
).all()

print("PASS: All features numeric")

assert np.isfinite(
    X_valid.to_numpy()
).all()

print("PASS: No NaN or infinite values")


# ============================================================
# RUN BATCH INFERENCE
# ============================================================

print("\nRunning batch inference...")

X_array = X_valid.to_numpy()

scores = model.decision_function(
    X_array
)

predictions = (
    scores < 0
).astype(int)

print("PASS: Batch inference completed")


# ============================================================
# CREATE OUTPUT
# ============================================================

output = pd.DataFrame({
    "timestamp": timestamps,
    "anomaly_score": scores,
    "prediction": predictions
})

output["status"] = np.where(
    output["prediction"] == 1,
    "ANOMALY",
    "NORMAL"
)


# ============================================================
# SAVE
# ============================================================

output.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"Output rows: {len(output)}"
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BATCH INFERENCE SUMMARY")
print("=" * 70)

print(
    f"Total valid rows : {len(output)}"
)

print(
    f"Normal predictions : "
    f"{(output['prediction'] == 0).sum()}"
)

print(
    f"Anomaly predictions : "
    f"{(output['prediction'] == 1).sum()}"
)

print(
    f"Anomaly rate : "
    f"{(output['prediction'] == 1).mean():.6f}"
)

print(
    f"Minimum score : "
    f"{output['anomaly_score'].min():.6f}"
)

print(
    f"Maximum score : "
    f"{output['anomaly_score'].max():.6f}"
)

print(
    f"Mean score : "
    f"{output['anomaly_score'].mean():.6f}"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25B VALIDATION")
print("=" * 70)

assert len(output) == len(X_valid)

print("PASS: Output row count matches valid input rows")

assert list(output.columns) == [
    "timestamp",
    "anomaly_score",
    "prediction",
    "status"
]

print("PASS: Output schema verified")

assert output["anomaly_score"].notna().all()

print("PASS: All anomaly scores present")

assert np.isfinite(
    output["anomaly_score"].to_numpy()
).all()

print("PASS: All anomaly scores finite")

assert output["prediction"].isin(
    [0, 1]
).all()

print("PASS: Predictions valid")

assert (
    output.loc[
        output["prediction"] == 1,
        "status"
    ] == "ANOMALY"
).all()

assert (
    output.loc[
        output["prediction"] == 0,
        "status"
    ] == "NORMAL"
).all()

print("PASS: Status values match predictions")

assert os.path.exists(OUTPUT_PATH)

print("PASS: Batch inference output saved")

print("PASS: No retraining")
print("PASS: No threshold tuning")
print("PASS: No feature modification")


# ============================================================
# OUTPUT
# ============================================================

print("\nOutput:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STAGE 25B: COMPLETE")
print("=" * 70)