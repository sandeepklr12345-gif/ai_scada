import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# STAGE 24E: FINAL CANDIDATE C ERROR GROUP ANALYSIS
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

TEST1_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_test1.csv"
)

TEST2_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_test2.csv"
)

RESULT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_error_group_summary.csv"
)

FEATURE_RESULT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_error_group_feature_summary.csv"
)


print("=" * 70)
print("STAGE 24E: FINAL CANDIDATE C ERROR GROUP ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading final Candidate C detector...")

model = joblib.load(MODEL_PATH)

print("PASS: Model loaded")


# ============================================================
# IDENTIFY FEATURES
# ============================================================

train = pd.read_csv(TRAIN_PATH)
test1 = pd.read_csv(TEST1_PATH)
test2 = pd.read_csv(TEST2_PATH)

feature_columns = [
    c for c in train.columns
    if c != "timestamp"
]

assert len(feature_columns) == 118

print(f"Feature count: {len(feature_columns)}")


# ============================================================
# ANALYZE DATASET
# ============================================================

def analyze_dataset(df, dataset_name):

    print("\n" + "-" * 70)
    print(dataset_name)
    print("-" * 70)

    X = df[feature_columns]

    # Valid rows only
    valid_mask = X.notna().all(axis=1)

    X_valid = X.loc[valid_mask].copy()

    # Labels
    labels = df.loc[valid_mask, "label"].astype(int).to_numpy()

    # Isolation Forest score
    scores = model.decision_function(X_valid)

    predictions = (scores < 0).astype(int)

    # Confusion groups
    tn_mask = (labels == 0) & (predictions == 0)
    fp_mask = (labels == 0) & (predictions == 1)
    fn_mask = (labels == 1) & (predictions == 0)
    tp_mask = (labels == 1) & (predictions == 1)

    groups = {
        "TN": tn_mask,
        "FP": fp_mask,
        "FN": fn_mask,
        "TP": tp_mask
    }

    rows = []

    for group_name, mask in groups.items():

        group_scores = scores[mask]

        if len(group_scores) == 0:
            rows.append({
                "dataset": dataset_name,
                "group": group_name,
                "count": 0,
                "mean_score": np.nan,
                "median_score": np.nan,
                "std_score": np.nan,
                "min_score": np.nan,
                "max_score": np.nan
            })

        else:
            rows.append({
                "dataset": dataset_name,
                "group": group_name,
                "count": int(mask.sum()),
                "mean_score": float(np.mean(group_scores)),
                "median_score": float(np.median(group_scores)),
                "std_score": float(np.std(group_scores)),
                "min_score": float(np.min(group_scores)),
                "max_score": float(np.max(group_scores))
            })

    # --------------------------------------------------------
    # FEATURE MEANS BY ERROR GROUP
    # --------------------------------------------------------

    feature_rows = []

    for group_name, mask in groups.items():

        group_X = X_valid.loc[mask]

        if len(group_X) == 0:
            continue

        means = group_X.mean()

        for feature, value in means.items():

            feature_rows.append({
                "dataset": dataset_name,
                "group": group_name,
                "feature": feature,
                "mean_value": float(value)
            })

    return rows, feature_rows, groups, scores, predictions, labels, X_valid


# ============================================================
# RUN TEST 1
# ============================================================

test1_rows, test1_feature_rows, *_ = analyze_dataset(
    test1,
    "test1"
)


# ============================================================
# RUN TEST 2
# ============================================================

test2_rows, test2_feature_rows, *_ = analyze_dataset(
    test2,
    "test2"
)


# ============================================================
# SAVE GROUP SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    test1_rows + test2_rows
)

summary_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# SAVE FEATURE GROUP SUMMARY
# ============================================================

feature_summary_df = pd.DataFrame(
    test1_feature_rows + test2_feature_rows
)

feature_summary_df.to_csv(
    FEATURE_RESULT_PATH,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR GROUP SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(index=False)
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 24E VALIDATION")
print("=" * 70)

expected_test1 = {
    "TN": 49685,
    "FP": 1330,
    "FN": 2406,
    "TP": 575
}

expected_test2 = {
    "TN": 204348,
    "FP": 17645,
    "FN": 5557,
    "TP": 2846
}


for dataset_name, expected in [
    ("test1", expected_test1),
    ("test2", expected_test2)
]:

    subset = summary_df[
        summary_df["dataset"] == dataset_name
    ]

    actual = dict(
        zip(
            subset["group"],
            subset["count"]
        )
    )

    assert actual == expected

print("PASS: Error-group counts exactly match Stage 24C")

assert len(feature_summary_df) == 2 * 4 * 118

print("PASS: Feature-group summary contains all groups and features")

assert np.isfinite(
    summary_df[
        [
            "mean_score",
            "median_score",
            "std_score",
            "min_score",
            "max_score"
        ]
    ].to_numpy()
).all()

print("PASS: Error-group score statistics are finite")

print("PASS: No retraining performed")
print("PASS: No threshold tuning performed")
print("PASS: No feature selection performed")
print("PASS: Test labels used only for post-hoc evaluation")


# ============================================================
# OUTPUTS
# ============================================================

print("\nOutputs:")
print(RESULT_PATH)
print(FEATURE_RESULT_PATH)

print("\n" + "=" * 70)
print("STAGE 24E: COMPLETE")
print("=" * 70)