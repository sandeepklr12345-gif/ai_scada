from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# ============================================================
# STAGE 21C: ISOLATION FOREST SCORE DISTRIBUTION ANALYSIS
# ============================================================

print("=" * 70)
print("STAGE 21C: ISOLATION FOREST SCORE DISTRIBUTION ANALYSIS")
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

DISTRIBUTION_OUTPUT = (
    OUTPUT_DIR
    / "hai_isolation_forest_score_distribution.csv"
)

PERCENTILE_OUTPUT = (
    OUTPUT_DIR
    / "hai_isolation_forest_score_percentiles.csv"
)

SAMPLE_OUTPUT = (
    OUTPUT_DIR
    / "hai_isolation_forest_score_samples.csv"
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
# FEATURES
# ------------------------------------------------------------

features = [
    column
    for column in training.columns
    if column != "timestamp"
]

assert len(features) == 58

print("\nTraining rows :", f"{len(training):,}")
print("Test 1 rows   :", f"{len(test1):,}")
print("Test 2 rows   :", f"{len(test2):,}")
print("ML features   :", len(features))


# ------------------------------------------------------------
# PREPARE MATRICES
# ------------------------------------------------------------

X_train = training[features]
X_test1 = test1[features]
X_test2 = test2[features]


# ------------------------------------------------------------
# TRAIN EXACT SAME BASELINE MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RETRAINING STAGE 21B MODEL")
print("=" * 70)

print("Model         : Isolation Forest")
print("Estimators    :", N_ESTIMATORS)
print("Contamination :", CONTAMINATION)
print("Random state  :", RANDOM_STATE)
print("Scaling       : None")

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
)

model.fit(X_train)

print("Model training complete.")


# ------------------------------------------------------------
# SCORE DATASETS
# ------------------------------------------------------------

def calculate_scores(model, X):

    # score_samples:
    # Higher = more normal
    # Lower = more anomalous
    raw_score = model.score_samples(X)

    # decision_function:
    # Positive = on normal side of threshold
    # Negative = on anomaly side of threshold
    decision_score = model.decision_function(X)

    # Convert to project-friendly anomaly score:
    # Higher = more anomalous
    anomaly_score = -decision_score

    return (
        raw_score,
        decision_score,
        anomaly_score,
    )


print("\nCalculating training scores...")

train_raw, train_decision, train_anomaly = (
    calculate_scores(
        model,
        X_train
    )
)

print("Training scores complete.")


print("Calculating Test 1 scores...")

test1_raw, test1_decision, test1_anomaly = (
    calculate_scores(
        model,
        X_test1
    )
)

print("Test 1 scores complete.")


print("Calculating Test 2 scores...")

test2_raw, test2_decision, test2_anomaly = (
    calculate_scores(
        model,
        X_test2
    )
)

print("Test 2 scores complete.")


# ------------------------------------------------------------
# MODEL THRESHOLD
# ------------------------------------------------------------

offset = model.offset_

print("\n" + "=" * 70)
print("ISOLATION FOREST THRESHOLD")
print("=" * 70)

print("Model offset_ :", offset)

print(
    "Decision rule : decision_function < 0 "
    "=> predicted anomaly"
)


# ------------------------------------------------------------
# VERIFY PREDICTIONS
# ------------------------------------------------------------

train_pred = (
    train_decision < 0
).astype(int)

test1_pred = (
    test1_decision < 0
).astype(int)

test2_pred = (
    test2_decision < 0
).astype(int)


print("\nPredicted anomalies using model threshold:")

print(
    "Training :",
    int(train_pred.sum()),
    f"({train_pred.mean() * 100:.4f}%)"
)

print(
    "Test 1   :",
    int(test1_pred.sum()),
    f"({test1_pred.mean() * 100:.4f}%)"
)

print(
    "Test 2   :",
    int(test2_pred.sum()),
    f"({test2_pred.mean() * 100:.4f}%)"
)


# ------------------------------------------------------------
# DISTRIBUTION FUNCTION
# ------------------------------------------------------------

def distribution_row(
    dataset_name,
    raw_score,
    decision_score,
    anomaly_score,
):

    return {
        "dataset": dataset_name,
        "rows": len(raw_score),

        "raw_score_min": np.min(raw_score),
        "raw_score_max": np.max(raw_score),
        "raw_score_mean": np.mean(raw_score),
        "raw_score_median": np.median(raw_score),
        "raw_score_std": np.std(raw_score),

        "decision_min": np.min(decision_score),
        "decision_max": np.max(decision_score),
        "decision_mean": np.mean(decision_score),
        "decision_median": np.median(decision_score),
        "decision_std": np.std(decision_score),

        "anomaly_score_min": np.min(anomaly_score),
        "anomaly_score_max": np.max(anomaly_score),
        "anomaly_score_mean": np.mean(anomaly_score),
        "anomaly_score_median": np.median(anomaly_score),
        "anomaly_score_std": np.std(anomaly_score),

        "predicted_anomalies": int(
            (decision_score < 0).sum()
        ),

        "predicted_anomaly_percent": (
            (decision_score < 0).mean() * 100
        ),
    }


distribution = pd.DataFrame([
    distribution_row(
        "Training",
        train_raw,
        train_decision,
        train_anomaly,
    ),

    distribution_row(
        "Test1",
        test1_raw,
        test1_decision,
        test1_anomaly,
    ),

    distribution_row(
        "Test2",
        test2_raw,
        test2_decision,
        test2_anomaly,
    ),
])


# ------------------------------------------------------------
# PERCENTILES
# ------------------------------------------------------------

percentile_values = [
    0,
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99,
    99.9,
    100,
]

percentile_rows = []

for dataset_name, anomaly_score in [
    ("Training", train_anomaly),
    ("Test1", test1_anomaly),
    ("Test2", test2_anomaly),
]:

    values = np.percentile(
        anomaly_score,
        percentile_values
    )

    for percentile, value in zip(
        percentile_values,
        values
    ):

        percentile_rows.append({
            "dataset": dataset_name,
            "percentile": percentile,
            "anomaly_score": value,
        })


percentiles = pd.DataFrame(
    percentile_rows
)


# ------------------------------------------------------------
# SCORE DISTRIBUTION PRINT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ANOMALY-SCORE DISTRIBUTION")
print("=" * 70)

print(
    distribution[
        [
            "dataset",
            "rows",
            "anomaly_score_min",
            "anomaly_score_max",
            "anomaly_score_mean",
            "anomaly_score_median",
            "anomaly_score_std",
            "predicted_anomalies",
            "predicted_anomaly_percent",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("ANOMALY-SCORE PERCENTILES")
print("=" * 70)

print(
    percentiles.to_string(index=False)
)


# ------------------------------------------------------------
# TRAINING-BASED THRESHOLD COMPARISON
# ------------------------------------------------------------

training_threshold = np.percentile(
    train_anomaly,
    95
)

print("\n" + "=" * 70)
print("TRAINING-DERIVED 95TH PERCENTILE")
print("=" * 70)

print(
    "Training anomaly-score 95th percentile:",
    training_threshold
)

for dataset_name, scores in [
    ("Training", train_anomaly),
    ("Test1", test1_anomaly),
    ("Test2", test2_anomaly),
]:

    percentage_above = (
        scores >= training_threshold
    ).mean() * 100

    print(
        f"{dataset_name:10s}: "
        f"{percentage_above:.4f}% above training 95th percentile"
    )


# ------------------------------------------------------------
# SCORE SAMPLES
# ------------------------------------------------------------

sample_rows = []

for dataset_name, scores in [
    ("Training", train_anomaly),
    ("Test1", test1_anomaly),
    ("Test2", test2_anomaly),
]:

    sample_count = min(
        1000,
        len(scores)
    )

    indices = np.linspace(
        0,
        len(scores) - 1,
        sample_count,
        dtype=int
    )

    for index in indices:

        sample_rows.append({
            "dataset": dataset_name,
            "index": int(index),
            "anomaly_score": float(
                scores[index]
            ),
        })


samples = pd.DataFrame(
    sample_rows
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

distribution.to_csv(
    DISTRIBUTION_OUTPUT,
    index=False
)

percentiles.to_csv(
    PERCENTILE_OUTPUT,
    index=False
)

samples.to_csv(
    SAMPLE_OUTPUT,
    index=False
)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(distribution) == 3
assert len(percentiles) == 36

assert np.isfinite(
    distribution.select_dtypes(
        include="number"
    ).to_numpy()
).all()

assert np.isfinite(
    percentiles["anomaly_score"]
).all()

assert len(train_anomaly) == 896400
assert len(test1_anomaly) == 54000
assert len(test2_anomaly) == 230400

print("PASS: Training score count preserved")
print("PASS: Test 1 score count preserved")
print("PASS: Test 2 score count preserved")
print("PASS: All score statistics are finite")
print("PASS: Three score distributions recorded")
print("PASS: Training-derived threshold calculated")
print("PASS: Test labels not used")


print("\nOutputs saved to:")
print(DISTRIBUTION_OUTPUT)
print(PERCENTILE_OUTPUT)
print(SAMPLE_OUTPUT)

print("\n" + "=" * 70)
print("STAGE 21C: COMPLETE")
print("=" * 70)