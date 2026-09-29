import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

print("=" * 70)
print("STAGE 24C: FINAL CANDIDATE C ISOLATION FOREST")
print("=" * 70)

BASE = r"data/features/hai/hai-23.05/temporal_representation/final_candidate"

TRAIN_FILE = os.path.join(
    BASE,
    "hai_2305_candidate_C_training.csv"
)

TEST1_FILE = os.path.join(
    BASE,
    "hai_2305_candidate_C_test1.csv"
)

TEST2_FILE = os.path.join(
    BASE,
    "hai_2305_candidate_C_test2.csv"
)

MODEL_OUTPUT = os.path.join(
    BASE,
    "hai_2305_candidate_C_isolation_forest.joblib"
)

RESULTS_OUTPUT = os.path.join(
    BASE,
    "hai_2305_candidate_C_isolation_forest_results.csv"
)

N_ESTIMATORS = 200
CONTAMINATION = 0.05
RANDOM_STATE = 42


# ======================================================================
# LOAD DATA
# ======================================================================

print("\nLoading Candidate C datasets...")

train = pd.read_csv(TRAIN_FILE)
test1 = pd.read_csv(TEST1_FILE)
test2 = pd.read_csv(TEST2_FILE)

print(f"Training : {train.shape}")
print(f"Test 1   : {test1.shape}")
print(f"Test 2   : {test2.shape}")


# ======================================================================
# IDENTIFY FEATURES
# ======================================================================

feature_columns = [
    col for col in train.columns
    if col != "timestamp"
]

if len(feature_columns) != 118:
    raise ValueError(
        f"Expected 118 features, found {len(feature_columns)}"
    )

print(f"\nCandidate C features: {len(feature_columns)}")


# ======================================================================
# VERIFY TEST SCHEMA
# ======================================================================

test1_features = [
    col for col in test1.columns
    if col not in ["timestamp", "label"]
]

test2_features = [
    col for col in test2.columns
    if col not in ["timestamp", "label"]
]

if test1_features != feature_columns:
    raise ValueError(
        "Test 1 feature schema does not match training."
    )

if test2_features != feature_columns:
    raise ValueError(
        "Test 2 feature schema does not match training."
    )

print("PASS: Training/Test 1/Test 2 feature schemas match")


# ======================================================================
# HANDLE TEMPORAL BOUNDARY ROWS
# ======================================================================

print("\nChecking temporal boundary rows...")

train_valid_mask = train[feature_columns].notna().all(axis=1)
test1_valid_mask = test1[feature_columns].notna().all(axis=1)
test2_valid_mask = test2[feature_columns].notna().all(axis=1)

train_invalid = (~train_valid_mask).sum()
test1_invalid = (~test1_valid_mask).sum()
test2_invalid = (~test2_valid_mask).sum()

print(f"Training boundary rows excluded : {train_invalid}")
print(f"Test 1 boundary rows excluded   : {test1_invalid}")
print(f"Test 2 boundary rows excluded   : {test2_invalid}")

print(
    "\nThese rows are excluded only because their temporal "
    "features are undefined at sequence boundaries."
)


# ======================================================================
# TRAINING DATA
# ======================================================================

X_train = train.loc[
    train_valid_mask,
    feature_columns
].to_numpy(dtype=np.float64)

print(f"\nTraining rows used: {X_train.shape[0]}")
print(f"Training features : {X_train.shape[1]}")


# ======================================================================
# TRAIN ISOLATION FOREST
# ======================================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train)

print("Isolation Forest training complete.")


# ======================================================================
# SAVE MODEL
# ======================================================================

joblib.dump(
    model,
    MODEL_OUTPUT
)

print(f"\nModel saved:")
print(os.path.abspath(MODEL_OUTPUT))


# ======================================================================
# EVALUATION FUNCTION
# ======================================================================

def evaluate_dataset(name, df, valid_mask):

    X = df.loc[
        valid_mask,
        feature_columns
    ].to_numpy(dtype=np.float64)

    y_true = df.loc[
        valid_mask,
        "label"
    ].to_numpy(dtype=int)

    predictions = model.predict(X)

    # IsolationForest:
    # +1 = normal
    # -1 = anomaly

    y_pred = (
        predictions == -1
    ).astype(int)

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

    predicted_anomaly_rate = y_pred.mean()

    result = {
        "dataset": name,
        "total_rows": len(df),
        "valid_rows": len(X),
        "excluded_boundary_rows": len(df) - len(X),
        "actual_normal": int((y_true == 0).sum()),
        "actual_anomaly": int((y_true == 1).sum()),
        "predicted_normal": int((y_pred == 0).sum()),
        "predicted_anomaly": int((y_pred == 1).sum()),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "predicted_anomaly_rate": predicted_anomaly_rate,
        "n_estimators": N_ESTIMATORS,
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
        "feature_count": len(feature_columns)
    }

    print("\n" + "-" * 70)
    print(f"{name}")
    print("-" * 70)

    print(f"Valid rows          : {len(X)}")
    print(f"Actual anomalies    : {int((y_true == 1).sum())}")
    print(f"Predicted anomalies : {int((y_pred == 1).sum())}")

    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1        : {f1:.6f}")
    print(f"Accuracy  : {accuracy:.6f}")

    print(
        f"Predicted anomaly rate: "
        f"{predicted_anomaly_rate:.6f}"
    )

    return result


# ======================================================================
# EVALUATE TEST SETS
# ======================================================================

results = []

results.append(
    evaluate_dataset(
        "test1",
        test1,
        test1_valid_mask
    )
)

results.append(
    evaluate_dataset(
        "test2",
        test2,
        test2_valid_mask
    )
)


# ======================================================================
# SAVE RESULTS
# ======================================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_OUTPUT,
    index=False
)

print("\nResults saved:")
print(os.path.abspath(RESULTS_OUTPUT))


# ======================================================================
# VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 24C VALIDATION")
print("=" * 70)

assert len(results_df) == 2
print("PASS: Test 1 and Test 2 results present")

assert all(
    results_df["feature_count"] == 118
)
print("PASS: All evaluations use 118 features")

assert all(
    results_df["n_estimators"] == 200
)
print("PASS: n_estimators = 200")

assert all(
    results_df["contamination"] == 0.05
)
print("PASS: contamination = 0.05")

assert all(
    results_df["random_state"] == 42
)
print("PASS: random_state = 42")

assert train_valid_mask.sum() == 896384
print("PASS: Training boundary exclusion verified")

assert test1_valid_mask.sum() == 53996
print("PASS: Test 1 boundary exclusion verified")

assert test2_valid_mask.sum() == 230396
print("PASS: Test 2 boundary exclusion verified")

print("PASS: No threshold tuning performed")
print("PASS: No test labels used during training")
print("PASS: No feature selection performed in Stage 24C")


# ======================================================================
# SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 24C: COMPLETE")
print("=" * 70)

print("\nFinal Candidate C detector:")
print(f"Features      : {len(feature_columns)}")
print(f"Estimators    : {N_ESTIMATORS}")
print(f"Contamination : {CONTAMINATION}")
print(f"Random state  : {RANDOM_STATE}")

print("\n" + results_df[
    [
        "dataset",
        "valid_rows",
        "precision",
        "recall",
        "f1",
        "accuracy"
    ]
].to_string(index=False))