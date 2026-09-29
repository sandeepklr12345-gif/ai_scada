from pathlib import Path

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


# ============================================================
# STAGE 23E
# HAI 23.05 - Temporal Feature Family Ablation
# ============================================================

ROOT = Path(
    r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"
)

TEMPORAL_DIR = (
    ROOT
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

RESULT_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_family_ablation_results.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

N_ESTIMATORS = 200
CONTAMINATION = 0.05
RANDOM_STATE = 42
N_JOBS = -1


print("=" * 70)
print("STAGE 23E: TEMPORAL FEATURE FAMILY ABLATION")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(
    TRAIN_FILE
)

assert train_df.shape == (
    896400,
    233
)

all_features = [
    col
    for col in train_df.columns
    if col != "timestamp"
]

assert len(all_features) == 232


# ============================================================
# 2. IDENTIFY FEATURE FAMILIES
# ============================================================

original_features = [
    f for f in all_features
    if "__" not in f
]

diff_features = [
    f for f in all_features
    if f.endswith("__diff_1s")
]

abs_diff_features = [
    f for f in all_features
    if f.endswith("__abs_diff_1s")
]

rolling_features = [
    f for f in all_features
    if f.endswith("__rolling_std_5s")
]


assert len(original_features) == 58
assert len(diff_features) == 58
assert len(abs_diff_features) == 58
assert len(rolling_features) == 58


# ============================================================
# 3. DEFINE EXPERIMENTS
# ============================================================

experiments = {
    "original_only": original_features,

    "original_plus_diff_1s":
        original_features
        + diff_features,

    "original_plus_abs_diff_1s":
        original_features
        + abs_diff_features,

    "original_plus_rolling_std_5s":
        original_features
        + rolling_features,

    "full_temporal":
        original_features
        + diff_features
        + abs_diff_features
        + rolling_features
}


print(
    "\nExperiments:"
)

for name, features in experiments.items():

    print(
        f"{name:35s}: "
        f"{len(features)} features"
    )


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

print(
    "\nLoading Test 1..."
)

test1_df = pd.read_csv(
    TEST1_FILE
)

print(
    "Loading Test 2..."
)

test2_df = pd.read_csv(
    TEST2_FILE
)

assert test1_df.shape == (
    54000,
    234
)

assert test2_df.shape == (
    230400,
    234
)


# ============================================================
# 5. VALIDATION
# ============================================================

for df in [
    test1_df,
    test2_df
]:

    assert "timestamp" in df.columns
    assert "label" in df.columns

    test_features = [
        col
        for col in df.columns
        if col not in [
            "timestamp",
            "label"
        ]
    ]

    assert test_features == all_features


# ============================================================
# 6. REMOVE TEMPORAL BOUNDARY ROWS
# ============================================================

train_valid_mask = (
    train_df[
        all_features
    ]
    .notna()
    .all(axis=1)
)

train_valid_df = train_df[
    train_valid_mask
].copy()

assert len(
    train_valid_df
) == 896384


test1_valid_mask = (
    test1_df[
        all_features
    ]
    .notna()
    .all(axis=1)
)

test1_valid_df = test1_df[
    test1_valid_mask
].copy()

assert len(
    test1_valid_df
) == 53996


test2_valid_mask = (
    test2_df[
        all_features
    ]
    .notna()
    .all(axis=1)
)

test2_valid_df = test2_df[
    test2_valid_mask
].copy()

assert len(
    test2_valid_df
) == 230396


y_test1 = test1_valid_df[
    "label"
].to_numpy(
    dtype=np.int8
)

y_test2 = test2_valid_df[
    "label"
].to_numpy(
    dtype=np.int8
)


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    df,
    features,
    y_true
):

    X = df[
        features
    ].to_numpy(
        dtype=np.float32
    )

    assert np.isfinite(
        X
    ).all()

    scores = model.decision_function(
        X
    )

    predictions = (
        scores < 0
    ).astype(
        np.int8
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "predicted_anomaly_rate": float(
            predictions.mean()
        )
    }


# ============================================================
# 8. RUN ABLATION EXPERIMENTS
# ============================================================

results = []

for experiment_name, features in experiments.items():

    print(
        "\n" + "=" * 70
    )

    print(
        f"RUNNING: {experiment_name}"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Training matrix
    # --------------------------------------------------------

    X_train = (
        train_valid_df[
            features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    assert np.isfinite(
        X_train
    ).all()


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

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
        "Model trained"
    )


    # --------------------------------------------------------
    # Test 1
    # --------------------------------------------------------

    test1_result = evaluate(
        model,
        test1_valid_df,
        features,
        y_test1
    )


    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    test2_result = evaluate(
        model,
        test2_valid_df,
        features,
        y_test2
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append(
        {
            "experiment": experiment_name,
            "feature_count": len(features),

            "test1_precision":
                test1_result["precision"],

            "test1_recall":
                test1_result["recall"],

            "test1_f1":
                test1_result["f1"],

            "test1_accuracy":
                test1_result["accuracy"],

            "test1_predicted_anomaly_rate":
                test1_result[
                    "predicted_anomaly_rate"
                ],

            "test1_TN":
                test1_result["TN"],

            "test1_FP":
                test1_result["FP"],

            "test1_FN":
                test1_result["FN"],

            "test1_TP":
                test1_result["TP"],

            "test2_precision":
                test2_result["precision"],

            "test2_recall":
                test2_result["recall"],

            "test2_f1":
                test2_result["f1"],

            "test2_accuracy":
                test2_result["accuracy"],

            "test2_predicted_anomaly_rate":
                test2_result[
                    "predicted_anomaly_rate"
                ],

            "test2_TN":
                test2_result["TN"],

            "test2_FP":
                test2_result["FP"],

            "test2_FN":
                test2_result["FN"],

            "test2_TP":
                test2_result["TP"]
        }
    )


    print(
        "\nTest 1:"
    )

    print(
        f"Precision : "
        f"{test1_result['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{test1_result['recall']:.6f}"
    )

    print(
        f"F1        : "
        f"{test1_result['f1']:.6f}"
    )

    print(
        f"Accuracy  : "
        f"{test1_result['accuracy']:.6f}"
    )


    print(
        "\nTest 2:"
    )

    print(
        f"Precision : "
        f"{test2_result['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{test2_result['recall']:.6f}"
    )

    print(
        f"F1        : "
        f"{test2_result['f1']:.6f}"
    )

    print(
        f"Accuracy  : "
        f"{test2_result['accuracy']:.6f}"
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 10. PRINT COMPARISON
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23E ABLATION RESULTS"
)

print(
    "=" * 70
)

print(
    results_df[
        [
            "experiment",
            "feature_count",
            "test1_precision",
            "test1_recall",
            "test1_f1",
            "test2_precision",
            "test2_recall",
            "test2_f1"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 11. VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23E FINAL VALIDATION"
)

print(
    "=" * 70
)

assert len(
    results_df
) == 5

assert set(
    results_df["experiment"]
) == set(
    experiments.keys()
)

assert (
    results_df["feature_count"]
    .tolist()
    == [58, 116, 116, 116, 232]
)

numeric_results = (
    results_df
    .select_dtypes(
        include=np.number
    )
    .to_numpy()
)

assert np.isfinite(
    numeric_results
).all()

assert Path(
    RESULT_FILE
).exists()

print(
    "PASS: Original-only baseline evaluated"
)

print(
    "PASS: Difference family evaluated"
)

print(
    "PASS: Absolute-difference family evaluated"
)

print(
    "PASS: Rolling-standard-deviation family evaluated"
)

print(
    "PASS: Full temporal representation evaluated"
)

print(
    "PASS: Same Isolation Forest configuration used"
)

print(
    "PASS: Test labels used only for evaluation"
)

print(
    "PASS: No threshold tuning"
)

print(
    "PASS: No feature selection based on test results"
)

print(
    "\nResults file:"
)

print(
    RESULT_FILE
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23E: COMPLETE"
)

print(
    "=" * 70
)