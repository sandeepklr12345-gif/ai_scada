from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ============================================================
# STAGE 23C
# HAI 23.05 - Temporal Isolation Forest Score Analysis
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

MODEL_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_isolation_forest_baseline.joblib"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_isolation_forest_score_summary.csv"
)

LABEL_SCORE_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_isolation_forest_label_score_summary.csv"
)

# ============================================================
# CONFIGURATION
# ============================================================

N_ESTIMATORS = 200
CONTAMINATION = 0.05
RANDOM_STATE = 42
N_JOBS = -1


print("=" * 70)
print("STAGE 23C: TEMPORAL ISOLATION FOREST SCORE ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(
    TRAIN_FILE
)

training_features = [
    col
    for col in train_df.columns
    if col != "timestamp"
]

assert len(training_features) == 232

print(
    f"Training rows    : {len(train_df):,}"
)

print(
    f"Training features : {len(training_features)}"
)


# ============================================================
# 2. REMOVE TEMPORAL BOUNDARY ROWS
# ============================================================

train_valid_mask = (
    train_df[
        training_features
    ]
    .notna()
    .all(axis=1)
)

train_model_df = train_df[
    train_valid_mask
]

print(
    f"Valid training rows: "
    f"{len(train_model_df):,}"
)

assert len(train_model_df) == 896384


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


# ============================================================
# 3. TRAIN SAME BASELINE MODEL
# ============================================================

print(
    "\nTraining same Stage 23B model..."
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
    "PASS: Model trained"
)


# ============================================================
# 4. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"Model saved:\n{MODEL_FILE}"
)


# ============================================================
# 5. MODEL THRESHOLD
# ============================================================

print(
    "\nModel threshold information:"
)

print(
    f"offset_: {model.offset_:.12f}"
)

print(
    "decision_function < 0 => anomaly"
)


# ============================================================
# SCORE SUMMARY FUNCTION
# ============================================================

def score_summary(
    dataset_name,
    scores
):

    return {
        "dataset": dataset_name,
        "count": len(scores),
        "min": float(
            np.min(scores)
        ),
        "max": float(
            np.max(scores)
        ),
        "mean": float(
            np.mean(scores)
        ),
        "median": float(
            np.median(scores)
        ),
        "std": float(
            np.std(scores)
        ),
        "p01": float(
            np.percentile(scores, 1)
        ),
        "p05": float(
            np.percentile(scores, 5)
        ),
        "p10": float(
            np.percentile(scores, 10)
        ),
        "p25": float(
            np.percentile(scores, 25)
        ),
        "p50": float(
            np.percentile(scores, 50)
        ),
        "p75": float(
            np.percentile(scores, 75)
        ),
        "p90": float(
            np.percentile(scores, 90)
        ),
        "p95": float(
            np.percentile(scores, 95)
        ),
        "p99": float(
            np.percentile(scores, 99)
        ),
        "below_zero": int(
            (scores < 0).sum()
        ),
        "above_or_equal_zero": int(
            (scores >= 0).sum()
        )
    }


# ============================================================
# 6. TRAINING SCORES
# ============================================================

print(
    "\nCalculating training scores..."
)

train_scores = model.decision_function(
    X_train
)

assert np.isfinite(
    train_scores
).all()

train_summary = score_summary(
    "training",
    train_scores
)

print(
    "\nTraining score distribution:"
)

for key, value in train_summary.items():

    if key != "dataset":
        print(
            f"{key:22s}: {value}"
        )


# ============================================================
# 7. TEST EVALUATION FUNCTION
# ============================================================

def analyze_test_set(
    dataset_name,
    test_file
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"ANALYZING {dataset_name.upper()}"
    )

    print(
        "=" * 70
    )

    df = pd.read_csv(
        test_file
    )

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

    assert len(test_features) == 232

    assert (
        test_features
        == training_features
    )

    boundary_mask = (
        df[
            test_features
        ]
        .isna()
        .any(axis=1)
    )

    print(
        f"Total rows       : {len(df):,}"
    )

    print(
        f"Boundary rows    : "
        f"{boundary_mask.sum()}"
    )

    valid_df = df[
        ~boundary_mask
    ].copy()

    X_test = (
        valid_df[
            test_features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    y_test = valid_df[
        "label"
    ].to_numpy(
        dtype=np.int8
    )

    assert np.isfinite(
        X_test
    ).all()

    scores = model.decision_function(
        X_test
    )

    predictions = (
        scores < 0
    ).astype(
        np.int8
    )

    assert np.isfinite(
        scores
    ).all()

    print(
        f"Valid rows       : {len(valid_df):,}"
    )

    print(
        f"Actual normal    : "
        f"{(y_test == 0).sum():,}"
    )

    print(
        f"Actual anomaly   : "
        f"{(y_test == 1).sum():,}"
    )

    print(
        f"Predicted anomaly: "
        f"{predictions.sum():,}"
    )

    summary = score_summary(
        dataset_name,
        scores
    )

    # --------------------------------------------------------
    # Label-conditioned score distributions
    # --------------------------------------------------------

    normal_scores = scores[
        y_test == 0
    ]

    anomaly_scores = scores[
        y_test == 1
    ]

    label_rows = [
        {
            "dataset": dataset_name,
            "actual_label": 0,
            "count": len(normal_scores),
            "min": float(
                np.min(normal_scores)
            ),
            "max": float(
                np.max(normal_scores)
            ),
            "mean": float(
                np.mean(normal_scores)
            ),
            "median": float(
                np.median(normal_scores)
            ),
            "std": float(
                np.std(normal_scores)
            ),
            "p05": float(
                np.percentile(
                    normal_scores,
                    5
                )
            ),
            "p25": float(
                np.percentile(
                    normal_scores,
                    25
                )
            ),
            "p50": float(
                np.percentile(
                    normal_scores,
                    50
                )
            ),
            "p75": float(
                np.percentile(
                    normal_scores,
                    75
                )
            ),
            "p95": float(
                np.percentile(
                    normal_scores,
                    95
                )
            ),
            "below_zero": int(
                (normal_scores < 0).sum()
            )
        },
        {
            "dataset": dataset_name,
            "actual_label": 1,
            "count": len(anomaly_scores),
            "min": float(
                np.min(anomaly_scores)
            ),
            "max": float(
                np.max(anomaly_scores)
            ),
            "mean": float(
                np.mean(anomaly_scores)
            ),
            "median": float(
                np.median(anomaly_scores)
            ),
            "std": float(
                np.std(anomaly_scores)
            ),
            "p05": float(
                np.percentile(
                    anomaly_scores,
                    5
                )
            ),
            "p25": float(
                np.percentile(
                    anomaly_scores,
                    25
                )
            ),
            "p50": float(
                np.percentile(
                    anomaly_scores,
                    50
                )
            ),
            "p75": float(
                np.percentile(
                    anomaly_scores,
                    75
                )
            ),
            "p95": float(
                np.percentile(
                    anomaly_scores,
                    95
                )
            ),
            "below_zero": int(
                (anomaly_scores < 0).sum()
            )
        }
    ]

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\nOverall score distribution:"
    )

    for key, value in summary.items():

        if key != "dataset":
            print(
                f"{key:22s}: {value}"
            )

    print(
        "\nScore distribution by actual label:"
    )

    for row in label_rows:

        print(
            f"\nActual label = "
            f"{row['actual_label']}"
        )

        print(
            f"Count    : {row['count']:,}"
        )

        print(
            f"Mean     : {row['mean']:.6f}"
        )

        print(
            f"Median   : {row['median']:.6f}"
        )

        print(
            f"P05      : {row['p05']:.6f}"
        )

        print(
            f"P25      : {row['p25']:.6f}"
        )

        print(
            f"P75      : {row['p75']:.6f}"
        )

        print(
            f"P95      : {row['p95']:.6f}"
        )

        print(
            f"Below 0  : {row['below_zero']:,}"
        )

    return summary, label_rows


# ============================================================
# 8. TEST 1
# ============================================================

test1_summary, test1_label_rows = (
    analyze_test_set(
        "test1",
        TEST1_FILE
    )
)


# ============================================================
# 9. TEST 2
# ============================================================

test2_summary, test2_label_rows = (
    analyze_test_set(
        "test2",
        TEST2_FILE
    )
)


# ============================================================
# 10. SAVE SCORE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        train_summary,
        test1_summary,
        test2_summary
    ]
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 11. SAVE LABEL SCORE SUMMARY
# ============================================================

label_summary_df = pd.DataFrame(
    test1_label_rows
    + test2_label_rows
)

label_summary_df.to_csv(
    LABEL_SCORE_FILE,
    index=False
)


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23C FINAL VALIDATION"
)

print(
    "=" * 70
)

assert len(summary_df) == 3

assert len(label_summary_df) == 4

assert np.isfinite(
    summary_df.select_dtypes(
        include=np.number
    ).to_numpy()
).all()

assert np.isfinite(
    label_summary_df.select_dtypes(
        include=np.number
    ).to_numpy()
).all()

assert Path(
    MODEL_FILE
).exists()

assert Path(
    SUMMARY_FILE
).exists()

assert Path(
    LABEL_SCORE_FILE
).exists()

print(
    "PASS: Training score distribution analyzed"
)

print(
    "PASS: Test 1 score distribution analyzed"
)

print(
    "PASS: Test 2 score distribution analyzed"
)

print(
    "PASS: Actual-label score distributions analyzed"
)

print(
    "PASS: Model threshold preserved"
)

print(
    "PASS: No threshold tuning performed"
)

print(
    "PASS: No test labels used for model training"
)

print(
    "\nScore summary:"
)

print(
    summary_df[
        [
            "dataset",
            "mean",
            "median",
            "p05",
            "p95",
            "below_zero"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nOutput files:"
)

print(
    MODEL_FILE
)

print(
    SUMMARY_FILE
)

print(
    LABEL_SCORE_FILE
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23C: COMPLETE"
)

print(
    "=" * 70
)