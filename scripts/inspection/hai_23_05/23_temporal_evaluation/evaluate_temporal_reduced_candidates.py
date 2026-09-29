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
# STAGE 23G
# HAI 23.05 - Evaluate Reduced Temporal Candidates
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

MANIFEST_FILE = (
    TEMPORAL_DIR
    / "reduced_candidates"
    / "hai_2305_temporal_reduced_candidate_manifest.csv"
)

FEATURE_LIST_FILE = (
    TEMPORAL_DIR
    / "reduced_candidates"
    / "hai_2305_temporal_reduced_candidate_features.txt"
)

OUTPUT_DIR = (
    TEMPORAL_DIR
    / "reduced_candidates"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_reduced_candidate_results.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

N_ESTIMATORS = 200
CONTAMINATION = 0.05
RANDOM_STATE = 42
N_JOBS = -1


print("=" * 70)
print("STAGE 23G: EVALUATE REDUCED TEMPORAL CANDIDATES")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(
    TRAIN_FILE
)

all_features = [
    col
    for col in train_df.columns
    if col != "timestamp"
]

assert len(all_features) == 232

print(
    f"Training rows    : {len(train_df):,}"
)

print(
    f"Available features: {len(all_features)}"
)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nLoading Test 1...")

test1_df = pd.read_csv(
    TEST1_FILE
)

print("Loading Test 2...")

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
# 3. VALIDATE TEST SCHEMA
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


print(
    "PASS: Test schemas match training schema"
)


# ============================================================
# 4. REMOVE TEMPORAL BOUNDARY ROWS
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


print(
    "\nBoundary handling:"
)

print(
    "Training valid rows : "
    f"{len(train_valid_df):,}"
)

print(
    "Test 1 valid rows   : "
    f"{len(test1_valid_df):,}"
)

print(
    "Test 2 valid rows   : "
    f"{len(test2_valid_df):,}"
)


# ============================================================
# 5. LOAD FEATURE LISTS
# ============================================================

print(
    "\nLoading candidate feature lists..."
)

with open(
    FEATURE_LIST_FILE,
    "r",
    encoding="utf-8"
) as f:

    lines = [
        line.strip()
        for line in f
        if line.strip()
    ]


candidate_features = {}

current_candidate = None

for line in lines:

    if (
        line.startswith("=")
        or line.startswith(
            "Feature count:"
        )
    ):
        continue

    if (
        line.startswith("candidate_")
        or line == "full_temporal_reference"
    ):

        current_candidate = line

        candidate_features[
            current_candidate
        ] = []

        continue

    if current_candidate is not None:

        if line in all_features:

            candidate_features[
                current_candidate
            ].append(line)


expected_candidates = {
    "candidate_A": 78,
    "candidate_B": 98,
    "candidate_C": 118,
    "candidate_D": 88,
    "candidate_E": 118,
    "full_temporal_reference": 232
}

assert set(
    candidate_features.keys()
) == set(
    expected_candidates.keys()
)

for name, expected_count in (
    expected_candidates.items()
):

    assert len(
        candidate_features[name]
    ) == expected_count


print(
    "PASS: Candidate feature lists loaded"
)


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    df,
    features,
    y_true
):

    X = (
        df[
            features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    assert np.isfinite(
        X
    ).all()

    predictions_raw = model.predict(
        X
    )

    predictions = (
        predictions_raw == -1
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
# 7. RUN CANDIDATES
# ============================================================

results = []

for candidate_name, features in (
    candidate_features.items()
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"RUNNING: {candidate_name}"
    )

    print(
        f"Feature count: {len(features)}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Training
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

    test1_result = evaluate_model(
        model,
        test1_valid_df,
        features,
        y_test1
    )


    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    test2_result = evaluate_model(
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
            "candidate":
                candidate_name,

            "feature_count":
                len(features),

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
# 8. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 9. PRINT SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23G RESULTS"
)

print(
    "=" * 70
)

print(
    results_df[
        [
            "candidate",
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
# 10. FINAL VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23G FINAL VALIDATION"
)

print(
    "=" * 70
)

assert len(
    results_df
) == 6

assert set(
    results_df[
        "candidate"
    ]
) == set(
    expected_candidates.keys()
)

assert (
    results_df[
        "feature_count"
    ]
    .tolist()
    == [
        78,
        98,
        118,
        88,
        118,
        232
    ]
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
    "PASS: All six candidates evaluated"
)

print(
    "PASS: Same Isolation Forest configuration used"
)

print(
    "PASS: Training-only feature selection preserved"
)

print(
    "PASS: Test 1 used only for evaluation"
)

print(
    "PASS: Test 2 used only for evaluation"
)

print(
    "PASS: No threshold tuning"
)

print(
    "PASS: Full 232-feature reference included"
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
    "STAGE 23G: COMPLETE"
)

print(
    "=" * 70
)