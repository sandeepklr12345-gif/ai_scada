import os
import joblib
import numpy as np
import pandas as pd

print("=" * 70)
print("STAGE 24D: FINAL CANDIDATE C SCORE ANALYSIS")
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

MODEL_FILE = os.path.join(
    BASE,
    "hai_2305_candidate_C_isolation_forest.joblib"
)

SCORE_OUTPUT = os.path.join(
    BASE,
    "hai_2305_candidate_C_score_summary.csv"
)

LABEL_SCORE_OUTPUT = os.path.join(
    BASE,
    "hai_2305_candidate_C_label_score_summary.csv"
)

# ======================================================================
# LOAD
# ======================================================================

print("\nLoading final Candidate C detector...")

model = joblib.load(MODEL_FILE)

train = pd.read_csv(TRAIN_FILE)
test1 = pd.read_csv(TEST1_FILE)
test2 = pd.read_csv(TEST2_FILE)

feature_columns = [
    col for col in train.columns
    if col != "timestamp"
]

print(f"Feature count: {len(feature_columns)}")

assert len(feature_columns) == 118


# ======================================================================
# VALID ROW MASKS
# ======================================================================

train_valid = train[feature_columns].notna().all(axis=1)
test1_valid = test1[feature_columns].notna().all(axis=1)
test2_valid = test2[feature_columns].notna().all(axis=1)


# ======================================================================
# SCORE FUNCTION
# ======================================================================

def get_scores(df, valid_mask):

    X = df.loc[
        valid_mask,
        feature_columns
    ].to_numpy(dtype=np.float64)

    return model.decision_function(X)


# ======================================================================
# SCORES
# ======================================================================

print("\nCalculating anomaly scores...")

train_scores = get_scores(train, train_valid)
test1_scores = get_scores(test1, test1_valid)
test2_scores = get_scores(test2, test2_valid)

print("PASS: Scores calculated")


# ======================================================================
# MODEL OFFSET
# ======================================================================

offset = model.offset_

print("\n" + "=" * 70)
print("MODEL SCORE INFORMATION")
print("=" * 70)

print(f"Model offset: {offset:.12f}")

print(
    "Isolation Forest decision_function < 0 "
    "indicates model-level anomaly."
)


# ======================================================================
# SUMMARY FUNCTION
# ======================================================================

def summarize_scores(name, scores):

    values = {
        "dataset": name,
        "count": len(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "p01": np.percentile(scores, 1),
        "p05": np.percentile(scores, 5),
        "p10": np.percentile(scores, 10),
        "p25": np.percentile(scores, 25),
        "p50": np.percentile(scores, 50),
        "p75": np.percentile(scores, 75),
        "p90": np.percentile(scores, 90),
        "p95": np.percentile(scores, 95),
        "p99": np.percentile(scores, 99),
        "below_zero": int((scores < 0).sum()),
        "below_zero_percent": float(
            (scores < 0).mean() * 100
        )
    }

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    for key, value in values.items():
        if key not in ["dataset", "count"]:
            if isinstance(value, float):
                print(f"{key:24s}: {value:.6f}")
            else:
                print(f"{key:24s}: {value}")

    return values


# ======================================================================
# OVERALL SCORE SUMMARIES
# ======================================================================

summary_rows = []

summary_rows.append(
    summarize_scores(
        "training",
        train_scores
    )
)

summary_rows.append(
    summarize_scores(
        "test1",
        test1_scores
    )
)

summary_rows.append(
    summarize_scores(
        "test2",
        test2_scores
    )
)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    SCORE_OUTPUT,
    index=False
)


# ======================================================================
# LABEL-CONDITIONAL SCORE ANALYSIS
# ======================================================================

def label_score_analysis(name, df, valid_mask, scores):

    labels = df.loc[
        valid_mask,
        "label"
    ].to_numpy(dtype=int)

    rows = []

    for label_value in [0, 1]:

        selected = scores[
            labels == label_value
        ]

        rows.append({
            "dataset": name,
            "label": label_value,
            "count": len(selected),
            "min": np.min(selected),
            "max": np.max(selected),
            "mean": np.mean(selected),
            "median": np.median(selected),
            "std": np.std(selected),
            "p05": np.percentile(selected, 5),
            "p25": np.percentile(selected, 25),
            "p50": np.percentile(selected, 50),
            "p75": np.percentile(selected, 75),
            "p95": np.percentile(selected, 95),
            "below_zero": int(
                (selected < 0).sum()
            ),
            "below_zero_percent": float(
                (selected < 0).mean() * 100
            )
        })

    return rows


label_rows = []

label_rows.extend(
    label_score_analysis(
        "test1",
        test1,
        test1_valid,
        test1_scores
    )
)

label_rows.extend(
    label_score_analysis(
        "test2",
        test2,
        test2_valid,
        test2_scores
    )
)

label_df = pd.DataFrame(label_rows)

print("\n" + "=" * 70)
print("LABEL-CONDITIONAL SCORE SUMMARY")
print("=" * 70)

print(
    label_df.to_string(index=False)
)

label_df.to_csv(
    LABEL_SCORE_OUTPUT,
    index=False
)


# ======================================================================
# SCORE SEPARATION
# ======================================================================

print("\n" + "=" * 70)
print("SCORE SEPARATION")
print("=" * 70)

for dataset in ["test1", "test2"]:

    subset = label_df[
        label_df["dataset"] == dataset
    ]

    normal_mean = subset[
        subset["label"] == 0
    ]["mean"].iloc[0]

    anomaly_mean = subset[
        subset["label"] == 1
    ]["mean"].iloc[0]

    difference = normal_mean - anomaly_mean

    print(
        f"{dataset}: normal_mean - anomaly_mean = "
        f"{difference:.6f}"
    )


# ======================================================================
# VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 24D VALIDATION")
print("=" * 70)

assert len(summary_df) == 3
print("PASS: Training/Test 1/Test 2 score summaries created")

assert len(label_df) == 4
print("PASS: Test 1/Test 2 label-conditioned summaries created")

assert len(train_scores) == 896384
assert len(test1_scores) == 53996
assert len(test2_scores) == 230396

print("PASS: Score counts match valid rows")

assert np.isfinite(train_scores).all()
assert np.isfinite(test1_scores).all()
assert np.isfinite(test2_scores).all()

print("PASS: All anomaly scores finite")

assert len(feature_columns) == 118

print("PASS: Final Candidate C uses 118 features")

print("PASS: No threshold tuning performed")
print("PASS: No model retraining performed")
print("PASS: No feature selection performed")


# ======================================================================
# OUTPUTS
# ======================================================================

print("\nOutputs:")

print(os.path.abspath(SCORE_OUTPUT))
print(os.path.abspath(LABEL_SCORE_OUTPUT))

print("\n" + "=" * 70)
print("STAGE 24D: COMPLETE")
print("=" * 70)