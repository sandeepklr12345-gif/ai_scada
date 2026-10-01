from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd

# ============================================================
# STAGE 24F: FINAL CANDIDATE C ERROR-GROUP FEATURE ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE_DIR = str(PROJECT_ROOT)

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

TEST2_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_test2.csv"
)

OUTPUT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_error_feature_analysis.csv"
)

TOP_OUTPUT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_top_error_features.csv"
)

print("=" * 70)
print("STAGE 24F: FINAL CANDIDATE C ERROR-GROUP FEATURE ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading final Candidate C detector...")

model = joblib.load(MODEL_PATH)

print("PASS: Model loaded")


# ============================================================
# LOAD TEST DATA
# ============================================================

test1 = pd.read_csv(TEST1_PATH)
test2 = pd.read_csv(TEST2_PATH)

feature_columns = [
    c for c in test1.columns
    if c not in ["timestamp", "label"]
]

assert len(feature_columns) == 118

print(f"Feature count: {len(feature_columns)}")


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_dataset(df, dataset_name):

    print("\n" + "-" * 70)
    print(dataset_name)
    print("-" * 70)

    X = df[feature_columns]

    valid_mask = X.notna().all(axis=1)

    X_valid = X.loc[valid_mask].copy()

    labels = df.loc[valid_mask, "label"].astype(int).to_numpy()

    scores = model.decision_function(X_valid)

    predictions = (scores < 0).astype(int)

    # --------------------------------------------------------
    # ERROR GROUPS
    # --------------------------------------------------------

    masks = {
        "TN": (labels == 0) & (predictions == 0),
        "FP": (labels == 0) & (predictions == 1),
        "FN": (labels == 1) & (predictions == 0),
        "TP": (labels == 1) & (predictions == 1)
    }

    # --------------------------------------------------------
    # GLOBAL FEATURE MEANS
    # --------------------------------------------------------

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    global_normal_mean = X_valid.loc[
        normal_mask
    ].mean()

    global_anomaly_mean = X_valid.loc[
        anomaly_mask
    ].mean()

    # --------------------------------------------------------
    # ERROR-GROUP MEANS
    # --------------------------------------------------------

    group_means = {}

    for group_name, mask in masks.items():

        if mask.sum() > 0:
            group_means[group_name] = X_valid.loc[mask].mean()
        else:
            group_means[group_name] = pd.Series(
                np.nan,
                index=feature_columns
            )

    # --------------------------------------------------------
    # FEATURE ANALYSIS
    # --------------------------------------------------------

    rows = []

    for feature in feature_columns:

        normal_mean = global_normal_mean[feature]
        anomaly_mean = global_anomaly_mean[feature]

        tn_mean = group_means["TN"][feature]
        fp_mean = group_means["FP"][feature]
        fn_mean = group_means["FN"][feature]
        tp_mean = group_means["TP"][feature]

        # Difference between actual anomaly and normal groups
        normal_anomaly_diff = anomaly_mean - normal_mean

        # Difference between TP and FN
        tp_fn_diff = tp_mean - fn_mean

        # Difference between FP and TN
        fp_tn_diff = fp_mean - tn_mean

        # Absolute values for ranking
        rows.append({
            "dataset": dataset_name,
            "feature": feature,

            "normal_mean": float(normal_mean),
            "anomaly_mean": float(anomaly_mean),

            "TN_mean": float(tn_mean),
            "FP_mean": float(fp_mean),
            "FN_mean": float(fn_mean),
            "TP_mean": float(tp_mean),

            "anomaly_normal_difference": float(
                normal_anomaly_diff
            ),

            "abs_anomaly_normal_difference": float(
                abs(normal_anomaly_diff)
            ),

            "TP_FN_difference": float(
                tp_fn_diff
            ),

            "abs_TP_FN_difference": float(
                abs(tp_fn_diff)
            ),

            "FP_TN_difference": float(
                fp_tn_diff
            ),

            "abs_FP_TN_difference": float(
                abs(fp_tn_diff)
            )
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN ANALYSIS
# ============================================================

test1_results = analyze_dataset(
    test1,
    "test1"
)

test2_results = analyze_dataset(
    test2,
    "test2"
)

results = pd.concat(
    [test1_results, test2_results],
    ignore_index=True
)


# ============================================================
# SAVE FULL RESULTS
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# TOP FEATURES
# ============================================================

top_rows = []

for dataset_name in ["test1", "test2"]:

    subset = results[
        results["dataset"] == dataset_name
    ].copy()

    # Top 15 by actual anomaly-vs-normal separation
    top_anomaly_normal = (
        subset
        .sort_values(
            "abs_anomaly_normal_difference",
            ascending=False
        )
        .head(15)
    )

    for _, row in top_anomaly_normal.iterrows():

        top_rows.append({
            "dataset": dataset_name,
            "analysis_type": "anomaly_vs_normal",
            "feature": row["feature"],
            "difference": row[
                "anomaly_normal_difference"
            ],
            "absolute_difference": row[
                "abs_anomaly_normal_difference"
            ]
        })

    # Top 15 by TP-vs-FN difference
    top_tp_fn = (
        subset
        .sort_values(
            "abs_TP_FN_difference",
            ascending=False
        )
        .head(15)
    )

    for _, row in top_tp_fn.iterrows():

        top_rows.append({
            "dataset": dataset_name,
            "analysis_type": "TP_vs_FN",
            "feature": row["feature"],
            "difference": row["TP_FN_difference"],
            "absolute_difference": row[
                "abs_TP_FN_difference"
            ]
        })

    # Top 15 by FP-vs-TN difference
    top_fp_tn = (
        subset
        .sort_values(
            "abs_FP_TN_difference",
            ascending=False
        )
        .head(15)
    )

    for _, row in top_fp_tn.iterrows():

        top_rows.append({
            "dataset": dataset_name,
            "analysis_type": "FP_vs_TN",
            "feature": row["feature"],
            "difference": row["FP_TN_difference"],
            "absolute_difference": row[
                "abs_FP_TN_difference"
            ]
        })


top_df = pd.DataFrame(top_rows)

top_df.to_csv(
    TOP_OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("TOP FEATURE DIFFERENCES")
print("=" * 70)

for dataset_name in ["test1", "test2"]:

    print("\n" + "-" * 70)
    print(f"{dataset_name.upper()} - ANOMALY VS NORMAL")
    print("-" * 70)

    subset = (
        results[
            results["dataset"] == dataset_name
        ]
        .sort_values(
            "abs_anomaly_normal_difference",
            ascending=False
        )
        .head(15)
    )

    print(
        subset[
            [
                "feature",
                "normal_mean",
                "anomaly_mean",
                "anomaly_normal_difference"
            ]
        ].to_string(index=False)
    )

    print("\n" + "-" * 70)
    print(f"{dataset_name.upper()} - TP VS FN")
    print("-" * 70)

    subset = (
        results[
            results["dataset"] == dataset_name
        ]
        .sort_values(
            "abs_TP_FN_difference",
            ascending=False
        )
        .head(15)
    )

    print(
        subset[
            [
                "feature",
                "TP_mean",
                "FN_mean",
                "TP_FN_difference"
            ]
        ].to_string(index=False)
    )

    print("\n" + "-" * 70)
    print(f"{dataset_name.upper()} - FP VS TN")
    print("-" * 70)

    subset = (
        results[
            results["dataset"] == dataset_name
        ]
        .sort_values(
            "abs_FP_TN_difference",
            ascending=False
        )
        .head(15)
    )

    print(
        subset[
            [
                "feature",
                "FP_mean",
                "TN_mean",
                "FP_TN_difference"
            ]
        ].to_string(index=False)
    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 24F VALIDATION")
print("=" * 70)

assert len(results) == 2 * 118

print("PASS: 236 feature-analysis rows created")

assert set(results["dataset"]) == {
    "test1",
    "test2"
}

print("PASS: Test 1 and Test 2 analyses present")

numeric_columns = [
    "normal_mean",
    "anomaly_mean",
    "TN_mean",
    "FP_mean",
    "FN_mean",
    "TP_mean",
    "anomaly_normal_difference",
    "abs_anomaly_normal_difference",
    "TP_FN_difference",
    "abs_TP_FN_difference",
    "FP_TN_difference",
    "abs_FP_TN_difference"
]

assert np.isfinite(
    results[numeric_columns].to_numpy()
).all()

print("PASS: Feature statistics are finite")

assert len(top_df) == 90

print("PASS: Top-feature summary contains expected 90 rows")

print("PASS: No retraining performed")
print("PASS: No threshold tuning performed")
print("PASS: No feature selection performed")
print("PASS: Test labels used only for post-hoc analysis")


# ============================================================
# OUTPUTS
# ============================================================

print("\nOutputs:")
print(OUTPUT_PATH)
print(TOP_OUTPUT_PATH)

print("\n" + "=" * 70)
print("STAGE 24F: COMPLETE")
print("=" * 70)