from pathlib import Path

import pandas as pd

import numpy as np

from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# ============================================================
# STAGE 23B
# HAI 23.05 - Temporal Isolation Forest Baseline
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TEMPORAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
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

OUTPUT_DIR = (
    TEMPORAL_DIR
    / "isolation_forest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_isolation_forest_results.csv"
)
# ============================================================
# MODEL CONFIGURATION
# ============================================================

N_ESTIMATORS = 200
CONTAMINATION = 0.05
RANDOM_STATE = 42
N_JOBS = -1


print("=" * 70)
print("STAGE 23B: TEMPORAL ISOLATION FOREST BASELINE")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("\nLoading temporal training data...")

train_df = pd.read_csv(
    TRAIN_FILE
)

print(
    f"Training rows    : {len(train_df):,}"
)

print(
    f"Training columns : {len(train_df.columns)}"
)

assert train_df.shape == (
    896400,
    233
)

assert "timestamp" in train_df.columns


# ============================================================
# 2. IDENTIFY TRAINING FEATURES
# ============================================================

training_features = [
    col
    for col in train_df.columns
    if col != "timestamp"
]

assert len(
    training_features
) == 232


# ============================================================
# 3. HANDLE TEMPORAL BOUNDARY NaNs
# ============================================================

print(
    "\nChecking temporal boundary NaNs..."
)

train_nan_count = (
    train_df[
        training_features
    ]
    .isna()
    .sum()
    .sum()
)

print(
    f"Training feature NaNs: "
    f"{train_nan_count:,}"
)

assert train_nan_count == 1392


# ------------------------------------------------------------
# Isolation Forest cannot accept NaNs.
#
# These NaNs occur only because temporal features cannot be
# calculated at sequence boundaries.
#
# We remove only rows containing temporal boundary NaNs.
# ------------------------------------------------------------

train_valid_mask = (
    train_df[
        training_features
    ]
    .notna()
    .all(axis=1)
)

train_model_df = train_df[
    train_valid_mask
].copy()

print(
    f"Training rows after boundary removal: "
    f"{len(train_model_df):,}"
)

assert len(
    train_model_df
) == 896384


# ============================================================
# 4. CREATE TRAINING MATRIX
# ============================================================

X_train = (
    train_model_df[
        training_features
    ]
    .to_numpy(
        dtype=np.float32
    )
)

assert X_train.shape == (
    896384,
    232
)

assert np.isfinite(
    X_train
).all()

print(
    "\nPASS: Training matrix contains "
    "896,384 × 232 finite values"
)


# ============================================================
# 5. TRAIN ISOLATION FOREST
# ============================================================

print(
    "\nTraining Isolation Forest..."
)

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

model.fit(
    X_train
)

print(
    "PASS: Isolation Forest trained"
)


# ============================================================
# 6. TRAINING PREDICTION SUMMARY
# ============================================================

print(
    "\nEvaluating training prediction distribution..."
)

train_predictions = model.predict(
    X_train
)

train_scores = model.decision_function(
    X_train
)

train_predicted_anomalies = (
    train_predictions == -1
).sum()

train_predicted_normal = (
    train_predictions == 1
).sum()

print(
    f"Training predicted anomalies : "
    f"{train_predicted_anomalies:,}"
)

print(
    f"Training predicted normal    : "
    f"{train_predicted_normal:,}"
)

print(
    f"Training anomaly rate        : "
    f"{train_predicted_anomalies / len(X_train):.6f}"
)


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_test_set(
    test_name,
    test_file
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"EVALUATING {test_name.upper()}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    test_df = pd.read_csv(
        test_file
    )

    print(
        f"Rows    : {len(test_df):,}"
    )

    print(
        f"Columns : {len(test_df.columns)}"
    )

    assert (
        len(test_df.columns)
        == 234
    )

    assert "timestamp" in test_df.columns
    assert "label" in test_df.columns


    # --------------------------------------------------------
    # Feature schema
    # --------------------------------------------------------

    test_features = [
        col
        for col in test_df.columns
        if col not in [
            "timestamp",
            "label"
        ]
    ]

    assert len(
        test_features
    ) == 232

    assert (
        test_features
        == training_features
    )

    print(
        "PASS: Test feature schema matches training"
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_true = test_df[
        "label"
    ].to_numpy(
        dtype=np.int8
    )

    assert set(
        np.unique(y_true)
    ).issubset({
        0,
        1
    })


    # --------------------------------------------------------
    # Boundary rows
    # --------------------------------------------------------

    test_nan_mask = (
        test_df[
            test_features
        ]
        .isna()
        .any(axis=1)
    )

    test_boundary_rows = (
        test_nan_mask.sum()
    )

    print(
        f"Rows with temporal boundary NaNs: "
        f"{test_boundary_rows}"
    )

    assert test_boundary_rows == 4


    # --------------------------------------------------------
    # Remove boundary rows only
    # --------------------------------------------------------

    eval_df = test_df[
        ~test_nan_mask
    ].copy()

    y_eval = eval_df[
        "label"
    ].to_numpy(
        dtype=np.int8
    )

    X_test = (
        eval_df[
            test_features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    assert np.isfinite(
        X_test
    ).all()


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions_raw = model.predict(
        X_test
    )

    scores = model.decision_function(
        X_test
    )


    # Convert Isolation Forest:
    #
    # +1 = normal
    # -1 = anomaly
    #
    y_pred = (
        predictions_raw == -1
    ).astype(
        np.int8
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_eval,
        y_pred,
        labels=[0, 1]
    ).ravel()


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_eval,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_eval,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_eval,
        y_pred,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_eval,
        y_pred
    )

    predicted_anomalies = (
        y_pred == 1
    ).sum()

    predicted_anomaly_rate = (
        predicted_anomalies
        / len(y_pred)
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\nActual anomaly labels:"
    )

    print(
        f"Normal  : {(y_eval == 0).sum():,}"
    )

    print(
        f"Anomaly : {(y_eval == 1).sum():,}"
    )


    print(
        "\nPredicted:"
    )

    print(
        f"Normal  : {(y_pred == 0).sum():,}"
    )

    print(
        f"Anomaly : {(y_pred == 1).sum():,}"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        f"TN = {tn:,}"
    )

    print(
        f"FP = {fp:,}"
    )

    print(
        f"FN = {fn:,}"
    )

    print(
        f"TP = {tp:,}"
    )


    print(
        "\nMetrics:"
    )

    print(
        f"Precision          : {precision:.6f}"
    )

    print(
        f"Recall             : {recall:.6f}"
    )

    print(
        f"F1 Score           : {f1:.6f}"
    )

    print(
        f"Accuracy           : {accuracy:.6f}"
    )

    print(
        f"Predicted anomaly rate: "
        f"{predicted_anomaly_rate:.6f}"
    )


    print(
        "\nScore statistics:"
    )

    print(
        f"Minimum : {scores.min():.6f}"
    )

    print(
        f"Maximum : {scores.max():.6f}"
    )

    print(
        f"Mean    : {scores.mean():.6f}"
    )

    print(
        f"Median  : {np.median(scores):.6f}"
    )

    print(
        f"Std     : {scores.std():.6f}"
    )


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "dataset": test_name,
        "rows_total": len(test_df),
        "rows_evaluated": len(eval_df),
        "boundary_rows_removed": int(
            test_boundary_rows
        ),
        "actual_normal": int(
            (y_eval == 0).sum()
        ),
        "actual_anomaly": int(
            (y_eval == 1).sum()
        ),
        "predicted_normal": int(
            (y_pred == 0).sum()
        ),
        "predicted_anomaly": int(
            (y_pred == 1).sum()
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "predicted_anomaly_rate": float(
            predicted_anomaly_rate
        ),
        "score_min": float(
            scores.min()
        ),
        "score_max": float(
            scores.max()
        ),
        "score_mean": float(
            scores.mean()
        ),
        "score_median": float(
            np.median(scores)
        ),
        "score_std": float(
            scores.std()
        )
    }


# ============================================================
# 8. EVALUATE TEST 1
# ============================================================

results = []

results.append(
    evaluate_test_set(
        "test1",
        TEST1_FILE
    )
)


# ============================================================
# 9. EVALUATE TEST 2
# ============================================================

results.append(
    evaluate_test_set(
        "test2",
        TEST2_FILE
    )
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23B FINAL SUMMARY"
)

print(
    "=" * 70
)

print(
    results_df[
        [
            "dataset",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "predicted_anomaly_rate"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nTraining configuration:"
)

print(
    f"n_estimators : {N_ESTIMATORS}"
)

print(
    f"contamination : {CONTAMINATION}"
)

print(
    f"random_state : {RANDOM_STATE}"
)

print(
    f"n_jobs       : {N_JOBS}"
)


# ============================================================
# 12. VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL VALIDATION"
)

print(
    "=" * 70
)

assert len(results_df) == 2

assert set(
    results_df["dataset"]
) == {
    "test1",
    "test2"
}

assert np.isfinite(
    results_df[
        [
            "precision",
            "recall",
            "f1",
            "accuracy"
        ]
    ].to_numpy()
).all()

assert (
    results_df["rows_total"]
    .tolist()
    == [54000, 230400]
)

assert (
    results_df["boundary_rows_removed"]
    .tolist()
    == [4, 4]
)

print(
    "PASS: Temporal Isolation Forest baseline trained"
)

print(
    "PASS: Test 1 evaluated"
)

print(
    "PASS: Test 2 evaluated"
)

print(
    "PASS: Boundary rows handled consistently"
)

print(
    "PASS: Evaluation metrics finite"
)

print(
    "PASS: Same model used for both test sets"
)

print(
    "PASS: Test labels used only after prediction"
)

print(
    "\nResults file:"
)

print(
    RESULTS_FILE
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23B: COMPLETE"
)

print(
    "=" * 70
)