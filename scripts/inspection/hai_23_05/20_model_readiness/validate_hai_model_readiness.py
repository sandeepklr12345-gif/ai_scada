from pathlib import Path
import pandas as pd


# ============================================================
# STAGE 20: HAI MODEL-READINESS VALIDATION
# ============================================================

print("=" * 70)
print("STAGE 20: HAI MODEL-READINESS VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]


RESOLVED_DECISIONS = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_feature_selection"
    / "hai_2305_training_feature_selection_resolved.csv"
)

TRAINING_DATA = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_ml_candidates"
    / "hai_2305_training_ml_candidates.csv"
)

TEST1_DATA = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_ml_candidates"
    / "hai-test1_ml_candidates.csv"
)

TEST2_DATA = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_ml_candidates"
    / "hai-test2_ml_candidates.csv"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "model_ready"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAINING_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_training_model_ready.csv"
)

TEST1_OUTPUT = (
    OUTPUT_DIR
    / "hai-test1_model_ready.csv"
)

TEST2_OUTPUT = (
    OUTPUT_DIR
    / "hai-test2_model_ready.csv"
)

FEATURE_LIST_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_model_ready_features.csv"
)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("\nLoading feature-selection decisions...")
decisions = pd.read_csv(RESOLVED_DECISIONS)

print("Loading training dataset...")
training = pd.read_csv(TRAINING_DATA)

print("Loading Test 1...")
test1 = pd.read_csv(TEST1_DATA)

print("Loading Test 2...")
test2 = pd.read_csv(TEST2_DATA)

print("Files loaded successfully.")


# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

assert len(decisions) == 66
assert decisions["feature"].is_unique

assert "resolved_decision" in decisions.columns

assert "timestamp" in training.columns
assert "timestamp" in test1.columns
assert "timestamp" in test2.columns

assert "label" in test1.columns
assert "label" in test2.columns


# ------------------------------------------------------------
# DERIVE KEEP FEATURES
# ------------------------------------------------------------

keep_features = decisions.loc[
    decisions["resolved_decision"] == "KEEP",
    "feature"
].tolist()

review_features = decisions.loc[
    decisions["resolved_decision"] == "REVIEW",
    "feature"
].tolist()

exclude_features = decisions.loc[
    decisions["resolved_decision"] == "EXCLUDE",
    "feature"
].tolist()


print("\n" + "=" * 70)
print("FEATURE SELECTION SUMMARY")
print("=" * 70)

print("KEEP    :", len(keep_features))
print("REVIEW  :", len(review_features))
print("EXCLUDE :", len(exclude_features))
print("TOTAL   :", len(decisions))


# ------------------------------------------------------------
# EXPECTED CURRENT STATE
# ------------------------------------------------------------

assert len(keep_features) == 58
assert len(review_features) == 8
assert len(exclude_features) == 0

assert len(set(keep_features)) == 58


# ------------------------------------------------------------
# VERIFY FEATURES EXIST
# ------------------------------------------------------------

for dataset_name, dataset in [
    ("Training", training),
    ("Test 1", test1),
    ("Test 2", test2),
]:

    missing = [
        feature
        for feature in keep_features
        if feature not in dataset.columns
    ]

    assert not missing, (
        f"{dataset_name} missing KEEP features: {missing}"
    )

    print(
        f"PASS: {dataset_name} contains all 58 KEEP features"
    )


# ------------------------------------------------------------
# CREATE MODEL-READY DATASETS
# ------------------------------------------------------------

training_columns = [
    "timestamp"
] + keep_features

test_columns = [
    "timestamp"
] + keep_features + ["label"]


training_ready = training[
    training_columns
].copy()

test1_ready = test1[
    test_columns
].copy()

test2_ready = test2[
    test_columns
].copy()


# ------------------------------------------------------------
# SCHEMA VALIDATION
# ------------------------------------------------------------

expected_training_columns = [
    "timestamp"
] + keep_features

expected_test_columns = [
    "timestamp"
] + keep_features + ["label"]


assert list(
    training_ready.columns
) == expected_training_columns

assert list(
    test1_ready.columns
) == expected_test_columns

assert list(
    test2_ready.columns
) == expected_test_columns

print("PASS: Training schema is correct")
print("PASS: Test 1 schema is correct")
print("PASS: Test 2 schema is correct")


# ------------------------------------------------------------
# ROW COUNT VALIDATION
# ------------------------------------------------------------

assert len(training_ready) == len(training)
assert len(test1_ready) == len(test1)
assert len(test2_ready) == len(test2)

print(
    f"PASS: Training row count preserved: {len(training_ready):,}"
)

print(
    f"PASS: Test 1 row count preserved: {len(test1_ready):,}"
)

print(
    f"PASS: Test 2 row count preserved: {len(test2_ready):,}"
)


# ------------------------------------------------------------
# MISSING VALUE VALIDATION
# ------------------------------------------------------------

training_missing = (
    training_ready[keep_features]
    .isna()
    .sum()
    .sum()
)

test1_missing = (
    test1_ready[keep_features]
    .isna()
    .sum()
    .sum()
)

test2_missing = (
    test2_ready[keep_features]
    .isna()
    .sum()
    .sum()
)


assert training_missing == 0
assert test1_missing == 0
assert test2_missing == 0

print("PASS: No missing values in training features")
print("PASS: No missing values in Test 1 features")
print("PASS: No missing values in Test 2 features")


# ------------------------------------------------------------
# INFINITE VALUE VALIDATION
# ------------------------------------------------------------

training_inf = (
    training_ready[keep_features]
    .select_dtypes(include="number")
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

test1_inf = (
    test1_ready[keep_features]
    .select_dtypes(include="number")
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

test2_inf = (
    test2_ready[keep_features]
    .select_dtypes(include="number")
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)


assert training_inf == 0
assert test1_inf == 0
assert test2_inf == 0

print("PASS: No infinite values in training features")
print("PASS: No infinite values in Test 1 features")
print("PASS: No infinite values in Test 2 features")


# ------------------------------------------------------------
# NUMERIC FEATURE VALIDATION
# ------------------------------------------------------------

for dataset_name, dataset in [
    ("Training", training_ready),
    ("Test 1", test1_ready),
    ("Test 2", test2_ready),
]:

    non_numeric = [
        feature
        for feature in keep_features
        if not pd.api.types.is_numeric_dtype(
            dataset[feature]
        )
    ]

    assert not non_numeric, (
        f"{dataset_name} has non-numeric features: "
        f"{non_numeric}"
    )

    print(
        f"PASS: {dataset_name} has 58 numeric ML features"
    )


# ------------------------------------------------------------
# TIMESTAMP VALIDATION
# ------------------------------------------------------------

for dataset_name, dataset in [
    ("Training", training_ready),
    ("Test 1", test1_ready),
    ("Test 2", test2_ready),
]:

    timestamps = pd.to_datetime(
        dataset["timestamp"],
        errors="coerce"
    )

    assert timestamps.notna().all()

    print(
        f"PASS: {dataset_name} timestamps are valid"
    )


# ------------------------------------------------------------
# LABEL PROTECTION
# ------------------------------------------------------------

assert "label" not in training_ready.columns
assert "label" in test1_ready.columns
assert "label" in test2_ready.columns

print("PASS: Training dataset contains no labels")
print("PASS: Test labels preserved only for evaluation")


# ------------------------------------------------------------
# REVIEW FEATURES ARE NOT INCLUDED
# ------------------------------------------------------------

for feature in review_features:

    assert feature not in training_ready.columns
    assert feature not in test1_ready.columns
    assert feature not in test2_ready.columns


print(
    "PASS: All 8 REVIEW features excluded from "
    "model-ready matrix without deleting source data"
)


# ------------------------------------------------------------
# SAVE FEATURE LIST
# ------------------------------------------------------------

feature_metadata = decisions[
    decisions["resolved_decision"] == "KEEP"
].copy()

feature_metadata = feature_metadata[
    [
        "feature",
        "role",
        "high_correlation_count",
        "max_absolute_correlation",
        "most_correlated_feature",
        "most_correlated_correlation",
        "review_category",
        "resolved_decision",
        "resolved_reason",
    ]
]

feature_metadata.to_csv(
    FEATURE_LIST_OUTPUT,
    index=False
)


# ------------------------------------------------------------
# SAVE DATASETS
# ------------------------------------------------------------

training_ready.to_csv(
    TRAINING_OUTPUT,
    index=False
)

test1_ready.to_csv(
    TEST1_OUTPUT,
    index=False
)

test2_ready.to_csv(
    TEST2_OUTPUT,
    index=False
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 20 RESULTS")
print("=" * 70)

print(
    "Model-ready features :",
    len(keep_features)
)

print(
    "Training rows         :",
    len(training_ready)
)

print(
    "Test 1 rows           :",
    len(test1_ready)
)

print(
    "Test 2 rows           :",
    len(test2_ready)
)

print(
    "Review features       :",
    len(review_features)
)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(training_ready.columns) == 59
assert len(test1_ready.columns) == 60
assert len(test2_ready.columns) == 60

assert len(training_ready) == 896400
assert len(test1_ready) == 54000
assert len(test2_ready) == 230400

assert len(feature_metadata) == 58

print("PASS: Training = timestamp + 58 features")
print("PASS: Test 1 = timestamp + 58 features + label")
print("PASS: Test 2 = timestamp + 58 features + label")
print("PASS: Training row count preserved")
print("PASS: Test 1 row count preserved")
print("PASS: Test 2 row count preserved")
print("PASS: 58-feature schema consistent across datasets")
print("PASS: No missing values")
print("PASS: No infinite values")
print("PASS: All ML features numeric")
print("PASS: Test labels protected from feature selection")
print("PASS: 8 REVIEW features kept outside model matrix")

print("\nOutputs saved to:")
print(TRAINING_OUTPUT)
print(TEST1_OUTPUT)
print(TEST2_OUTPUT)
print(FEATURE_LIST_OUTPUT)

print("\n" + "=" * 70)
print("STAGE 20: COMPLETE")
print("=" * 70)