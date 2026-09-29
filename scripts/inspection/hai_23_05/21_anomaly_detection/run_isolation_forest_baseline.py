from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# STAGE 21B: RAW ISOLATION FOREST BASELINE
# ============================================================

print("=" * 70)
print("STAGE 21B: RAW ISOLATION FOREST BASELINE")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "model_ready"
    / "hai_2305_training_model_ready.csv"
)

TEST1_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "model_ready"
    / "hai-test1_model_ready.csv"
)

TEST2_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "model_ready"
    / "hai-test2_model_ready.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "anomaly_detection"
    / "isolation_forest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST1_OUTPUT = (
    OUTPUT_DIR
    / "hai_test1_isolation_forest_predictions.csv"
)

TEST2_OUTPUT = (
    OUTPUT_DIR
    / "hai_test2_isolation_forest_predictions.csv"
)

METRICS_OUTPUT = (
    OUTPUT_DIR
    / "hai_isolation_forest_baseline_metrics.csv"
)

CONFIG_OUTPUT = (
    OUTPUT_DIR
    / "hai_isolation_forest_baseline_config.csv"
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

N_ESTIMATORS = 200
CONTAMINATION = 0.05
RANDOM_STATE = 42
N_JOBS = -1


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading training data...")
training = pd.read_csv(TRAINING_FILE)

print("Loading Test 1...")
test1 = pd.read_csv(TEST1_FILE)

print("Loading Test 2...")
test2 = pd.read_csv(TEST2_FILE)

print("Files loaded successfully.")


# ------------------------------------------------------------
# FEATURE IDENTIFICATION
# ------------------------------------------------------------

features = [
    column
    for column in training.columns
    if column != "timestamp"
]

assert len(features) == 58

assert "label" in test1.columns
assert "label" in test2.columns

print("\nTraining rows :", f"{len(training):,}")
print("Test 1 rows   :", f"{len(test1):,}")
print("Test 2 rows   :", f"{len(test2):,}")
print("ML features   :", len(features))


# ------------------------------------------------------------
# DATA VALIDATION
# ------------------------------------------------------------

for name, dataset in [
    ("Training", training),
    ("Test 1", test1),
    ("Test 2", test2),
]:

    missing = (
        dataset[features]
        .isna()
        .sum()
        .sum()
    )

    infinite = (
        np.isinf(
            dataset[features].to_numpy()
        )
        .sum()
    )

    assert missing == 0
    assert infinite == 0

    print(
        f"PASS: {name} has no missing/infinite ML values"
    )


# ------------------------------------------------------------
# PREPARE MATRICES
# ------------------------------------------------------------

X_train = training[features]

X_test1 = test1[features]
y_test1 = test1["label"].astype(int)

X_test2 = test2[features]
y_test2 = test2["label"].astype(int)


# ------------------------------------------------------------
# LABEL VALIDATION
# ------------------------------------------------------------

assert set(y_test1.unique()).issubset({0, 1})
assert set(y_test2.unique()).issubset({0, 1})

print("\nTest 1 actual labels:")
print(
    y_test1.value_counts()
    .sort_index()
    .to_string()
)

print("\nTest 2 actual labels:")
print(
    y_test2.value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)

print("Model         : Isolation Forest")
print("Estimators    :", N_ESTIMATORS)
print("Contamination :", CONTAMINATION)
print("Random state  :", RANDOM_STATE)
print("Jobs          :", N_JOBS)
print("Scaling       : None")
print("Training      : HAI 23.05 training data only")


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING ISOLATION FOREST")
print("=" * 70)

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
)

model.fit(X_train)

print("Isolation Forest training complete.")


# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------

def generate_predictions(
    dataset,
    X,
    y,
    dataset_name,
):

    # sklearn:
    # +1 = normal
    # -1 = anomaly
    raw_prediction = model.predict(X)

    # Convert to project convention:
    # 0 = normal
    # 1 = anomaly
    predicted_label = (
        raw_prediction == -1
    ).astype(int)

    # decision_function:
    # larger = more normal
    # smaller = more anomalous
    decision_score = model.decision_function(X)

    # Convert so larger score means more anomalous
    anomaly_score = -decision_score

    result = pd.DataFrame({
        "timestamp": dataset["timestamp"].values,
        "actual_label": y.values,
        "predicted_label": predicted_label,
        "anomaly_score": anomaly_score,
    })

    return result


# ------------------------------------------------------------
# TEST 1
# ------------------------------------------------------------

print("\nGenerating Test 1 predictions...")

test1_result = generate_predictions(
    test1,
    X_test1,
    y_test1,
    "Test 1",
)

print("Test 1 predictions complete.")


# ------------------------------------------------------------
# TEST 2
# ------------------------------------------------------------

print("Generating Test 2 predictions...")

test2_result = generate_predictions(
    test2,
    X_test2,
    y_test2,
    "Test 2",
)

print("Test 2 predictions complete.")


# ------------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate(
    actual,
    predicted,
    dataset_name,
):

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    precision = precision_score(
        actual,
        predicted,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0,
    )

    tn, fp, fn, tp = cm.ravel()

    predicted_anomalies = int(
        predicted.sum()
    )

    actual_anomalies = int(
        actual.sum()
    )

    print("\n" + "=" * 70)
    print(dataset_name, "BASELINE RESULTS")
    print("=" * 70)

    print("\nActual anomalies     :", actual_anomalies)
    print("Predicted anomalies  :", predicted_anomalies)

    print("\nConfusion Matrix:")
    print(
        "                 Predicted"
    )
    print(
        "                 Normal  Anomaly"
    )
    print(
        f"Actual Normal    {tn:7d} {fp:8d}"
    )
    print(
        f"Actual Anomaly   {fn:7d} {tp:8d}"
    )

    print("\nPrecision :", precision)
    print("Recall    :", recall)
    print("F1-score  :", f1)

    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=[
                "Normal",
                "Anomaly",
            ],
            zero_division=0,
        )
    )

    return {
        "dataset": dataset_name,
        "actual_anomalies": actual_anomalies,
        "predicted_anomalies": predicted_anomalies,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------

test1_metrics = evaluate(
    y_test1,
    test1_result["predicted_label"],
    "TEST 1",
)

test2_metrics = evaluate(
    y_test2,
    test2_result["predicted_label"],
    "TEST 2",
)


# ------------------------------------------------------------
# SAVE PREDICTIONS
# ------------------------------------------------------------

test1_result.to_csv(
    TEST1_OUTPUT,
    index=False,
)

test2_result.to_csv(
    TEST2_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# SAVE METRICS
# ------------------------------------------------------------

metrics = pd.DataFrame([
    test1_metrics,
    test2_metrics,
])

metrics.to_csv(
    METRICS_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# SAVE CONFIGURATION
# ------------------------------------------------------------

config = pd.DataFrame([
    {
        "model": "IsolationForest",
        "n_estimators": N_ESTIMATORS,
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "scaling": "None",
        "training_rows": len(training),
        "features": len(features),
        "test1_rows": len(test1),
        "test2_rows": len(test2),
    }
])

config.to_csv(
    CONFIG_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(test1_result) == len(test1)
assert len(test2_result) == len(test2)

assert test1_result["predicted_label"].isin(
    [0, 1]
).all()

assert test2_result["predicted_label"].isin(
    [0, 1]
).all()

assert test1_result["anomaly_score"].notna().all()
assert test2_result["anomaly_score"].notna().all()

assert np.isfinite(
    test1_result["anomaly_score"]
).all()

assert np.isfinite(
    test2_result["anomaly_score"]
).all()

print("PASS: Test 1 prediction count preserved")
print("PASS: Test 2 prediction count preserved")
print("PASS: Predictions use 0/1 convention")
print("PASS: Anomaly scores are finite")
print("PASS: Official labels used only for evaluation")

print("\nOutputs saved to:")
print(TEST1_OUTPUT)
print(TEST2_OUTPUT)
print(METRICS_OUTPUT)
print(CONFIG_OUTPUT)

print("\n" + "=" * 70)
print("STAGE 21B: COMPLETE")
print("=" * 70)