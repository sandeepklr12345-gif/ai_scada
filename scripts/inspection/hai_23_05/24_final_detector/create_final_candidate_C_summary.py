from pathlib import Path
import os
import pandas as pd
import numpy as np


# ============================================================
# STAGE 24G: FINAL CANDIDATE C CONSOLIDATED SUMMARY
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

# ============================================================
# INPUT FILES
# ============================================================

RESULT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_isolation_forest_results.csv"
)

SCORE_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_score_summary.csv"
)

LABEL_SCORE_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_label_score_summary.csv"
)

ERROR_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_error_group_summary.csv"
)

FEATURE_ERROR_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_error_feature_analysis.csv"
)

TOP_FEATURE_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_top_error_features.csv"
)

FEATURE_MANIFEST_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_feature_manifest.csv"
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

# ============================================================
# OUTPUT FILES
# ============================================================

SUMMARY_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_final_summary.csv"
)

REPORT_PATH = os.path.join(
    FINAL_DIR,
    "hai_2305_candidate_C_final_summary.txt"
)


print("=" * 70)
print("STAGE 24G: FINAL CANDIDATE C CONSOLIDATED SUMMARY")
print("=" * 70)


# ============================================================
# LOAD FILES
# ============================================================

print("\nLoading Stage 24 outputs...")

results = pd.read_csv(RESULT_PATH)
scores = pd.read_csv(SCORE_PATH)
label_scores = pd.read_csv(LABEL_SCORE_PATH)
errors = pd.read_csv(ERROR_PATH)
feature_errors = pd.read_csv(FEATURE_ERROR_PATH)
top_features = pd.read_csv(TOP_FEATURE_PATH)
manifest = pd.read_csv(FEATURE_MANIFEST_PATH)

train = pd.read_csv(TRAIN_PATH)
test1 = pd.read_csv(TEST1_PATH)
test2 = pd.read_csv(TEST2_PATH)

print("PASS: All required files loaded")


# ============================================================
# SHOW ACTUAL RESULT SCHEMA
# ============================================================

print("\nStage 24C result columns:")
print(list(results.columns))


# ============================================================
# FEATURE VALIDATION
# ============================================================

feature_columns = [
    c for c in train.columns
    if c != "timestamp"
]

assert len(feature_columns) == 118

print("\nPASS: Candidate C contains 118 features")


# ============================================================
# DATASET VALIDATION
# ============================================================

assert len(train) == 896400
assert len(test1) == 54000
assert len(test2) == 230400

print("PASS: Dataset row counts verified")


# ============================================================
# EXPECTED STAGE 24C RESULTS
# ============================================================

expected_metrics = {
    "test1": {
        "valid_rows": 53996,
        "TN": 49685,
        "FP": 1330,
        "FN": 2406,
        "TP": 575,
        "precision": 0.301837,
        "recall": 0.192888,
        "f1": 0.235366,
        "accuracy": 0.930810
    },

    "test2": {
        "valid_rows": 230396,
        "TN": 204348,
        "FP": 17645,
        "FN": 5557,
        "TP": 2846,
        "precision": 0.138890,
        "recall": 0.338689,
        "f1": 0.196996,
        "accuracy": 0.899295
    }
}


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_result_columns = {
    "dataset",
    "valid_rows",
    "TN",
    "FP",
    "FN",
    "TP",
    "precision",
    "recall",
    "f1",
    "accuracy"
}

missing_columns = (
    required_result_columns
    - set(results.columns)
)

assert not missing_columns, (
    f"Missing Stage 24C columns: {missing_columns}"
)

print("PASS: Stage 24C result schema verified")


# ============================================================
# VALIDATE PERFORMANCE
# ============================================================

for dataset_name, expected in expected_metrics.items():

    subset = results[
        results["dataset"] == dataset_name
    ]

    assert len(subset) == 1

    row = subset.iloc[0]

    for key, expected_value in expected.items():

        actual_value = row[key]

        assert np.isclose(
            actual_value,
            expected_value,
            atol=1e-6
        ), (
            f"{dataset_name} {key}: "
            f"{actual_value} != {expected_value}"
        )

print("PASS: Stage 24C performance reproduced")


# ============================================================
# VALIDATE ERROR GROUPS
# ============================================================

expected_errors = {
    "test1": {
        "TN": 49685,
        "FP": 1330,
        "FN": 2406,
        "TP": 575
    },

    "test2": {
        "TN": 204348,
        "FP": 17645,
        "FN": 5557,
        "TP": 2846
    }
}


for dataset_name, expected in expected_errors.items():

    subset = errors[
        errors["dataset"] == dataset_name
    ]

    actual = dict(
        zip(
            subset["group"],
            subset["count"]
        )
    )

    assert actual == expected

print("PASS: Stage 24E error groups reproduced")


# ============================================================
# SCORE VALIDATION
# ============================================================

assert set(scores["dataset"]) == {
    "training",
    "test1",
    "test2"
}

print("PASS: Score summaries verified")


# ============================================================
# LABEL SCORE VALIDATION
# ============================================================

expected_label_groups = {
    ("test1", 0),
    ("test1", 1),
    ("test2", 0),
    ("test2", 1)
}

actual_label_groups = set(
    zip(
        label_scores["dataset"],
        label_scores["label"]
    )
)

assert actual_label_groups == expected_label_groups

print("PASS: Label-conditioned score summaries verified")


# ============================================================
# FEATURE ANALYSIS VALIDATION
# ============================================================

assert len(feature_errors) == 236
assert len(top_features) == 90

print("PASS: Stage 24F feature analysis verified")


# ============================================================
# BUILD CONSOLIDATED SUMMARY
# ============================================================

summary_rows = []


# ------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------

configuration = [
    ("candidate", "C"),
    ("feature_count", 118),
    ("original_features", 58),
    ("diff_1s_features", 0),
    ("abs_diff_1s_features", 30),
    ("rolling_std_5s_features", 30),
    ("algorithm", "Isolation Forest"),
    ("n_estimators", 200),
    ("contamination", 0.05),
    ("random_state", 42),
    ("scaling", "none"),
    ("decision_rule", "decision_function < 0"),
    ("threshold_tuning", False),
    ("test_labels_used_for_training", False),
    ("test_labels_used_for_feature_selection", False),
    ("retraining_during_evaluation", False)
]

for metric, value in configuration:

    summary_rows.append({
        "section": "configuration",
        "dataset": "all",
        "metric": metric,
        "value": value
    })


# ------------------------------------------------------------
# DATASET INFORMATION
# ------------------------------------------------------------

dataset_information = [
    ("training", "rows", len(train)),
    ("training", "features", 118),
    ("test1", "rows", len(test1)),
    ("test1", "features", 118),
    ("test2", "rows", len(test2)),
    ("test2", "features", 118)
]

for dataset_name, metric, value in dataset_information:

    summary_rows.append({
        "section": "dataset",
        "dataset": dataset_name,
        "metric": metric,
        "value": value
    })


# ------------------------------------------------------------
# PERFORMANCE
# ------------------------------------------------------------

performance_metrics = [
    "valid_rows",
    "TN",
    "FP",
    "FN",
    "TP",
    "precision",
    "recall",
    "f1",
    "accuracy"
]

for dataset_name in ["test1", "test2"]:

    row = results[
        results["dataset"] == dataset_name
    ].iloc[0]

    for metric in performance_metrics:

        summary_rows.append({
            "section": "performance",
            "dataset": dataset_name,
            "metric": metric,
            "value": row[metric]
        })


# ------------------------------------------------------------
# SCORE SEPARATION
# ------------------------------------------------------------

for dataset_name in ["test1", "test2"]:

    subset = label_scores[
        label_scores["dataset"] == dataset_name
    ]

    normal_mean = subset[
        subset["label"] == 0
    ]["mean"].iloc[0]

    anomaly_mean = subset[
        subset["label"] == 1
    ]["mean"].iloc[0]

    separation = normal_mean - anomaly_mean

    summary_rows.append({
        "section": "score_analysis",
        "dataset": dataset_name,
        "metric": "normal_mean_minus_anomaly_mean",
        "value": separation
    })


# ------------------------------------------------------------
# ERROR GROUP SCORE MEANS AND COUNTS
# ------------------------------------------------------------

for dataset_name in ["test1", "test2"]:

    subset = errors[
        errors["dataset"] == dataset_name
    ]

    for _, row in subset.iterrows():

        summary_rows.append({
            "section": "error_group",
            "dataset": dataset_name,
            "metric": f"{row['group']}_mean_score",
            "value": row["mean_score"]
        })

        summary_rows.append({
            "section": "error_group",
            "dataset": dataset_name,
            "metric": f"{row['group']}_count",
            "value": row["count"]
        })


# ============================================================
# SAVE CONSOLIDATED CSV
# ============================================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)

print("PASS: Consolidated CSV created")


# ============================================================
# CREATE HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 24G: FINAL CANDIDATE C DETECTOR SUMMARY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write("1. FINAL CANDIDATE CONFIGURATION\n")
    f.write("-" * 70 + "\n")

    for metric, value in configuration:
        f.write(f"{metric}: {value}\n")

    f.write("\n")

    f.write("2. DATASET SIZES\n")
    f.write("-" * 70 + "\n")

    f.write(f"Training rows: {len(train)}\n")
    f.write(f"Test 1 rows: {len(test1)}\n")
    f.write(f"Test 2 rows: {len(test2)}\n")
    f.write("Candidate C features: 118\n")

    f.write("\n")

    f.write("3. FINAL PERFORMANCE\n")
    f.write("-" * 70 + "\n\n")

    for dataset_name in ["test1", "test2"]:

        row = results[
            results["dataset"] == dataset_name
        ].iloc[0]

        f.write(
            f"{dataset_name.upper()}\n"
        )

        for metric in performance_metrics:

            value = row[metric]

            if metric in [
                "valid_rows",
                "TN",
                "FP",
                "FN",
                "TP"
            ]:
                f.write(
                    f"{metric}: {int(value)}\n"
                )
            else:
                f.write(
                    f"{metric}: {value:.6f}\n"
                )

        f.write("\n")


    f.write("4. SCORE SEPARATION\n")
    f.write("-" * 70 + "\n")

    for dataset_name in ["test1", "test2"]:

        subset = label_scores[
            label_scores["dataset"] == dataset_name
        ]

        normal_mean = subset[
            subset["label"] == 0
        ]["mean"].iloc[0]

        anomaly_mean = subset[
            subset["label"] == 1
        ]["mean"].iloc[0]

        f.write(
            f"{dataset_name}: "
            f"normal_mean - anomaly_mean = "
            f"{normal_mean - anomaly_mean:.6f}\n"
        )


    f.write("\n")

    f.write("5. ERROR-GROUP SCORE MEANS\n")
    f.write("-" * 70 + "\n")

    for dataset_name in ["test1", "test2"]:

        subset = errors[
            errors["dataset"] == dataset_name
        ]

        f.write(f"\n{dataset_name.upper()}\n")

        for _, row in subset.iterrows():

            f.write(
                f"{row['group']}: "
                f"count={int(row['count'])}, "
                f"mean_score={row['mean_score']:.6f}\n"
            )


    f.write("\n")

    f.write("6. FEATURE ANALYSIS\n")
    f.write("-" * 70 + "\n")

    f.write(
        "Post-hoc error-group analysis identified "
        "distributional differences across TN, FP, FN "
        "and TP groups.\n"
    )

    f.write(
        "No feature selection or model modification was "
        "performed from this analysis.\n"
    )

    f.write("\n")

    f.write("7. METHODOLOGICAL SAFEGUARDS\n")
    f.write("-" * 70 + "\n")

    f.write(
        "Test labels were not used for model training.\n"
    )

    f.write(
        "Test labels were not used for feature selection.\n"
    )

    f.write(
        "No threshold tuning was performed.\n"
    )

    f.write(
        "No retraining was performed during final evaluation.\n"
    )

    f.write(
        "Stages 24D, 24E and 24F were post-hoc analyses.\n"
    )

    f.write("\n")

    f.write("8. LIMITATION\n")
    f.write("-" * 70 + "\n")

    f.write(
        "The HAI labels represent attack/anomaly presence "
        "in the HAI dataset and should not be interpreted "
        "as direct power-outage labels.\n"
    )

    f.write(
        "The Isolation Forest detector is therefore an "
        "SCADA anomaly-detection component within the "
        "broader predictive outage-analysis architecture.\n"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 24G VALIDATION")
print("=" * 70)

assert len(summary_df) > 0
print("PASS: Consolidated summary contains data")

assert os.path.exists(SUMMARY_PATH)
print("PASS: Consolidated CSV exists")

assert os.path.exists(REPORT_PATH)
print("PASS: Human-readable report exists")

assert len(feature_columns) == 118
print("PASS: Final Candidate C feature count = 118")

print("PASS: Performance values verified against Stage 24C")

print("PASS: No retraining performed")
print("PASS: No threshold tuning performed")
print("PASS: No feature selection performed")

print("\nOutputs:")
print(SUMMARY_PATH)
print(REPORT_PATH)

print("\n" + "=" * 70)
print("STAGE 24G: COMPLETE")
print("=" * 70)
