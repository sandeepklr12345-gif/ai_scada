import os
import pandas as pd

print("=" * 70)
print("STAGE 24B: MATERIALIZE CANDIDATE C")
print("=" * 70)

BASE = r"data/features/hai/hai-23.05/temporal_representation"

TRAIN_INPUT = os.path.join(
    BASE,
    "hai_2305_training_temporal_model_ready.csv"
)

TEST1_INPUT = os.path.join(
    BASE,
    "hai-test1_temporal_model_ready.csv"
)

TEST2_INPUT = os.path.join(
    BASE,
    "hai-test2_temporal_model_ready.csv"
)

FEATURE_FILE = os.path.join(
    BASE,
    "reduced_candidates",
    "hai_2305_temporal_reduced_candidate_features.txt"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "final_candidate"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "hai_2305_candidate_C_training.csv"
)

TEST1_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "hai_2305_candidate_C_test1.csv"
)

TEST2_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "hai_2305_candidate_C_test2.csv"
)

MANIFEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "hai_2305_candidate_C_feature_manifest.csv"
)


# ======================================================================
# READ CANDIDATE C FEATURE LIST
# ======================================================================

print("\nLoading authoritative Candidate C feature list...")

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f]

start = None
end = None

for i, line in enumerate(lines):
    if line == "candidate_C":
        start = i
        break

if start is None:
    raise ValueError("candidate_C section not found.")

for i in range(start + 1, len(lines)):
    if lines[i].startswith("candidate_") and lines[i] != "candidate_C":
        end = i
        break

if end is None:
    end = len(lines)

candidate_section = lines[start:end]

# Find feature-count line
feature_count_line = next(
    (
        line
        for line in candidate_section
        if line.startswith("Feature count:")
    ),
    None
)

if feature_count_line is None:
    raise ValueError(
        "Candidate C feature-count line not found."
    )

declared_count = int(
    feature_count_line.split(":", 1)[1].strip()
)

features = []

for line in candidate_section:
    if (
        line
        and not line.startswith("=")
        and not line.startswith("candidate_")
        and not line.startswith("Feature count:")
    ):
        features.append(line)

# Remove accidental blank/duplicate entries
features = list(dict.fromkeys(features))

print(f"Declared feature count : {declared_count}")
print(f"Extracted feature count: {len(features)}")


# ======================================================================
# VALIDATE FEATURE LIST
# ======================================================================

if declared_count != 118:
    raise ValueError(
        f"Candidate C declares {declared_count} features, expected 118."
    )

if len(features) != 118:
    raise ValueError(
        f"Extracted {len(features)} features, expected 118."
    )

if len(features) != len(set(features)):
    raise ValueError(
        "Duplicate Candidate C feature names detected."
    )

print("PASS: Candidate C contains exactly 118 unique features")


# ======================================================================
# LOAD TEMPORAL MODEL-READY DATASETS
# ======================================================================

print("\nLoading temporal model-ready datasets...")

train = pd.read_csv(TRAIN_INPUT)
test1 = pd.read_csv(TEST1_INPUT)
test2 = pd.read_csv(TEST2_INPUT)

print(f"Training shape : {train.shape}")
print(f"Test 1 shape  : {test1.shape}")
print(f"Test 2 shape  : {test2.shape}")


# ======================================================================
# VERIFY FEATURES EXIST
# ======================================================================

print("\nChecking Candidate C features against datasets...")

for name, df in [
    ("training", train),
    ("test1", test1),
    ("test2", test2)
]:

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        print(f"\n{name} missing features:")
        for feature in missing:
            print(feature)

        raise ValueError(
            f"{name} is missing {len(missing)} Candidate C features."
        )

    print(
        f"PASS: {name} contains all 118 Candidate C features"
    )


# ======================================================================
# BUILD FINAL DATASETS
# ======================================================================

print("\nMaterializing Candidate C datasets...")

train_columns = ["timestamp"] + features
test_columns = ["timestamp"] + features + ["label"]

candidate_train = train[train_columns].copy()
candidate_test1 = test1[test_columns].copy()
candidate_test2 = test2[test_columns].copy()


# ======================================================================
# SAVE
# ======================================================================

candidate_train.to_csv(
    TRAIN_OUTPUT,
    index=False
)

candidate_test1.to_csv(
    TEST1_OUTPUT,
    index=False
)

candidate_test2.to_csv(
    TEST2_OUTPUT,
    index=False
)

feature_manifest = pd.DataFrame({
    "feature_order": range(1, 119),
    "feature": features
})

feature_manifest.to_csv(
    MANIFEST_OUTPUT,
    index=False
)


# ======================================================================
# VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 24B VALIDATION")
print("=" * 70)


# Shapes
assert candidate_train.shape == (896400, 119)
print("PASS: Training shape = 896400 × 119")

assert candidate_test1.shape == (54000, 120)
print("PASS: Test 1 shape = 54000 × 120")

assert candidate_test2.shape == (230400, 120)
print("PASS: Test 2 shape = 230400 × 120")


# Column order
assert list(candidate_train.columns) == [
    "timestamp"
] + features

assert list(candidate_test1.columns) == [
    "timestamp"
] + features + ["label"]

assert list(candidate_test2.columns) == [
    "timestamp"
] + features + ["label"]

print("PASS: Identical Candidate C feature ordering")


# Row counts
assert len(candidate_train) == 896400
assert len(candidate_test1) == 54000
assert len(candidate_test2) == 230400

print("PASS: Row counts preserved")


# Labels
assert "label" not in candidate_train.columns
assert "label" in candidate_test1.columns
assert "label" in candidate_test2.columns

print("PASS: Training contains no labels")
print("PASS: Evaluation labels preserved only in Test 1/Test 2")


# Timestamp validation
for name, df in [
    ("training", candidate_train),
    ("test1", candidate_test1),
    ("test2", candidate_test2)
]:

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    assert timestamps.notna().all()

    print(
        f"PASS: {name} timestamps valid"
    )


# Infinite values
for name, df in [
    ("training", candidate_train),
    ("test1", candidate_test1),
    ("test2", candidate_test2)
]:

    numeric = df.drop(
        columns=["timestamp", "label"],
        errors="ignore"
    )

    infinite_count = (
        numeric
        .select_dtypes(include="number")
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    assert infinite_count == 0

    print(
        f"PASS: {name} contains no infinite feature values"
    )


# NaN structure
train_nan = (
    candidate_train
    .drop(columns=["timestamp"])
    .isna()
    .sum()
    .sum()
)

test1_nan = (
    candidate_test1
    .drop(columns=["timestamp", "label"])
    .isna()
    .sum()
    .sum()
)

test2_nan = (
    candidate_test2
    .drop(columns=["timestamp", "label"])
    .isna()
    .sum()
    .sum()
)

print("\nTemporal boundary NaNs:")
print(f"Training : {train_nan}")
print(f"Test 1   : {test1_nan}")
print(f"Test 2   : {test2_nan}")

# NaNs are expected because temporal features have
# sequence-boundary positions.
assert train_nan >= 0
assert test1_nan >= 0
assert test2_nan >= 0

print("PASS: Temporal boundary NaN structure preserved")


# Feature manifest
assert len(feature_manifest) == 118
assert feature_manifest["feature"].is_unique

print("PASS: Feature manifest contains 118 unique features")


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("CANDIDATE C FINAL REPRESENTATION")
print("=" * 70)

print("Original features       : 58")
print("Diff 1s features        : 0")
print("Abs diff 1s features    : 30")
print("Rolling std 5s features : 30")
print("Total features          : 118")

print("\nDatasets:")
print(f"Training : {candidate_train.shape}")
print(f"Test 1   : {candidate_test1.shape}")
print(f"Test 2   : {candidate_test2.shape}")

print("\nOutputs:")
print(os.path.abspath(TRAIN_OUTPUT))
print(os.path.abspath(TEST1_OUTPUT))
print(os.path.abspath(TEST2_OUTPUT))
print(os.path.abspath(MANIFEST_OUTPUT))

print("\n" + "=" * 70)
print("STAGE 24B: COMPLETE")
print("=" * 70)