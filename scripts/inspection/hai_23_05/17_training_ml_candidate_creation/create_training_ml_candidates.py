from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
)

ROLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "hai_2305_feature_roles.csv"
)

VARIABILITY_FILE = (
    TRAINING_DIR
    / "hai_2305_training_feature_variability.csv"
)

TRAINING_FILE = (
    TRAINING_DIR
    / "hai_2305_training_all.csv"
)

TEST1_FILE = (
    PROCESSED_DIR
    / "hai-test1_aligned.csv"
)

TEST2_FILE = (
    PROCESSED_DIR
    / "hai-test2_aligned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_ml_candidates"
)

TRAINING_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_training_ml_candidates.csv"
)

TEST1_OUTPUT = (
    OUTPUT_DIR
    / "hai-test1_ml_candidates.csv"
)

TEST2_OUTPUT = (
    OUTPUT_DIR
    / "hai-test2_ml_candidates.csv"
)

FEATURE_LIST_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_training_ml_candidate_features.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 17: TRAINING-BASED ML CANDIDATE CREATION")
print("=" * 70)


# ============================================================
# LOAD METADATA
# ============================================================

variability = pd.read_csv(
    VARIABILITY_FILE
)

roles = pd.read_csv(
    ROLE_FILE
)

print("\nLoading training data...")
training = pd.read_csv(TRAINING_FILE)

print("Loading Test 1...")
test1 = pd.read_csv(TEST1_FILE)

print("Loading Test 2...")
test2 = pd.read_csv(TEST2_FILE)

print("\nFiles loaded successfully.")


# ============================================================
# IDENTIFY TRAINING-VARIABLE FEATURES
# ============================================================

variable_features = variability[
    variability["constant_training"] == False
]["feature"].tolist()

constant_features = variability[
    variability["constant_training"] == True
]["feature"].tolist()


print("\nTraining feature statistics:")
print(f"Total SCADA features      : {len(variability)}")
print(f"Training-variable features: {len(variable_features)}")
print(f"Training-constant features: {len(constant_features)}")


# ============================================================
# VALIDATE EXPECTED COUNTS
# ============================================================

assert len(variability) == 86
assert len(variable_features) == 66
assert len(constant_features) == 20

assert len(set(variable_features)) == 66
assert len(set(constant_features)) == 20

assert set(variable_features).isdisjoint(
    set(constant_features)
)

assert (
    set(variable_features)
    | set(constant_features)
) == set(variability["feature"])


print("PASS: 86 features partitioned into 66 variable + 20 constant")


# ============================================================
# VERIFY FEATURES EXIST IN ALL DATASETS
# ============================================================

for feature in variable_features:

    assert feature in training.columns
    assert feature in test1.columns
    assert feature in test2.columns

print(
    "PASS: All 66 candidate features exist in "
    "training, Test 1 and Test 2"
)


# ============================================================
# IDENTIFY COMMON NON-FEATURE COLUMNS
# ============================================================

# ============================================================
# IDENTIFY COMMON NON-FEATURE COLUMNS
# ============================================================

assert "timestamp" in training.columns
assert "timestamp" in test1.columns
assert "timestamp" in test2.columns

assert "label" in test1.columns
assert "label" in test2.columns

TRAINING_COLUMNS = ["timestamp"] + variable_features

TEST_COLUMNS = (
    ["timestamp"]
    + variable_features
    + ["label"]
)


# ============================================================
# CREATE TRAINING CANDIDATE DATASET
# ============================================================

training_candidates = training[
    TRAINING_COLUMNS
].copy()


# ============================================================
# CREATE TEST DATASETS USING TRAINING FEATURE SCHEMA
# ============================================================

test1_candidates = test1[
    TEST_COLUMNS
].copy()

test2_candidates = test2[
    TEST_COLUMNS
].copy()


# ============================================================
# CREATE FEATURE METADATA
# ============================================================

role_lookup = dict(
    zip(
        roles["feature"],
        roles["role"]
    )
)

feature_metadata = []

for feature in variable_features:

    row = variability[
        variability["feature"] == feature
    ].iloc[0]

    feature_metadata.append(
        {
            "feature": feature,
            "role": role_lookup.get(
                feature,
                "Unknown"
            ),
            "training_unique_values":
                int(row["unique_values"]),
            "training_minimum":
                row["minimum"],
            "training_maximum":
                row["maximum"],
            "constant_training":
                bool(row["constant_training"]),
        }
    )

feature_metadata_df = pd.DataFrame(
    feature_metadata
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

training_candidates.to_csv(
    TRAINING_OUTPUT,
    index=False
)

test1_candidates.to_csv(
    TEST1_OUTPUT,
    index=False
)

test2_candidates.to_csv(
    TEST2_OUTPUT,
    index=False
)

feature_metadata_df.to_csv(
    FEATURE_LIST_OUTPUT,
    index=False
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 17 RESULTS")
print("=" * 70)

print(
    f"\nTraining candidate dataset : "
    f"{training_candidates.shape[0]} rows × "
    f"{training_candidates.shape[1]} columns"
)

print(
    f"Test 1 candidate dataset   : "
    f"{test1_candidates.shape[0]} rows × "
    f"{test1_candidates.shape[1]} columns"
)

print(
    f"Test 2 candidate dataset   : "
    f"{test2_candidates.shape[0]} rows × "
    f"{test2_candidates.shape[1]} columns"
)

print(
    f"\nML candidate SCADA features: "
    f"{len(variable_features)}"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)


# Feature counts

assert len(variable_features) == 66

assert training_candidates.shape[1] == 67
assert test1_candidates.shape[1] == 68
assert test2_candidates.shape[1] == 68


# Expected columns

assert (
    training_candidates.columns.tolist()
    == TRAINING_COLUMNS
)

assert (
    test1_candidates.columns.tolist()
    == TEST_COLUMNS
)

assert (
    test2_candidates.columns.tolist()
    == TEST_COLUMNS
)


# Row counts

assert len(training_candidates) == len(training)
assert len(test1_candidates) == len(test1)
assert len(test2_candidates) == len(test2)


# No missing values

assert training_candidates.isna().sum().sum() == 0
assert test1_candidates.isna().sum().sum() == 0
assert test2_candidates.isna().sum().sum() == 0


# Feature metadata

assert len(feature_metadata_df) == 66
assert feature_metadata_df["feature"].is_unique

assert not feature_metadata_df[
    "constant_training"
].any()


# Ensure excluded features are absent

for feature in constant_features:

    assert feature not in training_candidates.columns
    assert feature not in test1_candidates.columns
    assert feature not in test2_candidates.columns


print("PASS: 66 training-variable features selected")
print("PASS: Training schema validated")
print("PASS: Test 1 uses identical feature schema")
print("PASS: Test 2 uses identical feature schema")
print("PASS: Row counts preserved")
print("PASS: No missing values")
print("PASS: Feature metadata validated")
print("PASS: All 20 training-constant features excluded")
print("PASS: No test labels used for feature selection")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nCreated:")
print(TRAINING_OUTPUT)
print(TEST1_OUTPUT)
print(TEST2_OUTPUT)
print(FEATURE_LIST_OUTPUT)

print("\n" + "=" * 70)
print("STAGE 17: COMPLETE")
print("=" * 70)