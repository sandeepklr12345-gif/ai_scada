from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
)

ROLE_FILE = FEATURE_DIR / "hai_2305_feature_roles.csv"

OUTPUT_DIR = FEATURE_DIR / "ml_candidates"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

TEST1_INPUT = PROCESSED_DIR / "hai-test1_aligned.csv"
TEST2_INPUT = PROCESSED_DIR / "hai-test2_aligned.csv"

TEST1_OUTPUT = OUTPUT_DIR / "hai-test1_ml_candidates.csv"
TEST2_OUTPUT = OUTPUT_DIR / "hai-test2_ml_candidates.csv"

FEATURE_LIST_OUTPUT = OUTPUT_DIR / "hai_2305_ml_candidate_features.csv"


# ============================================================
# LOAD ROLE / STABILITY INFORMATION
# ============================================================

roles = pd.read_csv(ROLE_FILE)


required_role_columns = {
    "feature",
    "constant_both"
}

missing_role_columns = (
    required_role_columns
    - set(roles.columns)
)

if missing_role_columns:
    raise ValueError(
        "Missing columns in feature-role file: "
        + ", ".join(sorted(missing_role_columns))
    )


# ============================================================
# IDENTIFY ML CANDIDATE FEATURES
# ============================================================

candidate_features = roles.loc[
    roles["constant_both"] == False,
    "feature"
].tolist()


print("=" * 70)
print("HAI 23.05 ML CANDIDATE DATASET CREATION")
print("=" * 70)

print()
print(f"Total documented features : {len(roles)}")
print(f"Constant in both          : {sum(roles['constant_both'])}")
print(f"ML candidate features     : {len(candidate_features)}")


# ============================================================
# EXPECTED COUNT
# ============================================================

EXPECTED_CANDIDATE_COUNT = 68

if len(candidate_features) != EXPECTED_CANDIDATE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_CANDIDATE_COUNT} ML candidate features, "
        f"but found {len(candidate_features)}."
    )


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

test1 = pd.read_csv(TEST1_INPUT)
test2 = pd.read_csv(TEST2_INPUT)


# ============================================================
# VALIDATE COMMON STRUCTURE
# ============================================================

for name, df in [
    ("Test1", test1),
    ("Test2", test2)
]:

    required_columns = {"timestamp", "label"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing required columns: "
            + ", ".join(sorted(missing))
        )

    missing_features = (
        set(candidate_features)
        - set(df.columns)
    )

    if missing_features:
        raise ValueError(
            f"{name}: candidate features missing from dataset:\n"
            + "\n".join(sorted(missing_features))
        )


# ============================================================
# CREATE ML CANDIDATE DATASETS
# ============================================================

output_columns = [
    "timestamp"
] + candidate_features + [
    "label"
]

test1_ml = test1[output_columns].copy()
test2_ml = test2[output_columns].copy()


# ============================================================
# BASIC VALIDATION FUNCTION
# ============================================================

def validate_ml_dataset(df, name):

    print()
    print(f"{name} VALIDATION")
    print("-" * 70)

    expected_columns = (
        1
        + EXPECTED_CANDIDATE_COUNT
        + 1
    )

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print(
        f"Expected columns : {expected_columns}"
    )

    if len(df.columns) != expected_columns:
        raise ValueError(
            f"{name}: incorrect column count."
        )

    # Timestamp
    if "timestamp" not in df.columns:
        raise ValueError(
            f"{name}: timestamp missing."
        )

    # Label
    if "label" not in df.columns:
        raise ValueError(
            f"{name}: label missing."
        )

    # Feature count
    actual_features = [
        c for c in df.columns
        if c not in ["timestamp", "label"]
    ]

    if len(actual_features) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"{name}: expected "
            f"{EXPECTED_CANDIDATE_COUNT} features, "
            f"found {len(actual_features)}."
        )

    # No duplicate columns
    if df.columns.duplicated().any():
        raise ValueError(
            f"{name}: duplicate columns detected."
        )

    # Missing values
    missing_values = df.isna().sum().sum()

    print(
        f"Missing values : {missing_values:,}"
    )

    if missing_values != 0:
        raise ValueError(
            f"{name}: missing values detected."
        )

    # Duplicate rows
    duplicate_rows = df.duplicated().sum()

    print(
        f"Duplicate rows : {duplicate_rows:,}"
    )

    if duplicate_rows != 0:
        raise ValueError(
            f"{name}: duplicate rows detected."
        )

    # Label validation
    labels = sorted(
        df["label"].dropna().unique().tolist()
    )

    print(
        f"Labels : {labels}"
    )

    if labels != [0, 1]:
        raise ValueError(
            f"{name}: unexpected label values."
        )

    # Numeric feature validation
    non_numeric = [
        c
        for c in actual_features
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    if non_numeric:
        raise ValueError(
            f"{name}: non-numeric features found:\n"
            + "\n".join(non_numeric)
        )

    # Infinite values
    infinite_values = (
        df[actual_features]
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(
        f"Infinite values : {infinite_values:,}"
    )

    if infinite_values != 0:
        raise ValueError(
            f"{name}: infinite values detected."
        )

    # Timestamp validation
    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_timestamps = timestamps.isna().sum()

    print(
        f"Invalid timestamps : {invalid_timestamps:,}"
    )

    if invalid_timestamps != 0:
        raise ValueError(
            f"{name}: invalid timestamps detected."
        )

    # Label distribution
    print()
    print("Label distribution:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(f"{name}: PASS")


# ============================================================
# VALIDATE
# ============================================================

validate_ml_dataset(
    test1_ml,
    "TEST1"
)

validate_ml_dataset(
    test2_ml,
    "TEST2"
)


# ============================================================
# SAVE DATASETS
# ============================================================

test1_ml.to_csv(
    TEST1_OUTPUT,
    index=False
)

test2_ml.to_csv(
    TEST2_OUTPUT,
    index=False
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_list = roles[
    roles["feature"].isin(candidate_features)
][
    [
        "feature",
        "role",
        "variability",
        "constant_test1",
        "constant_test2",
        "constant_both"
    ]
].copy()

feature_list = feature_list.sort_values(
    "feature"
).reset_index(drop=True)

feature_list.to_csv(
    FEATURE_LIST_OUTPUT,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(TEST1_OUTPUT)
print(TEST2_OUTPUT)
print(FEATURE_LIST_OUTPUT)

print()
print("=" * 70)
print("HAI 23.05 ML CANDIDATE CREATION COMPLETE")
print("=" * 70)

print()
print(
    f"ML candidate features: {len(candidate_features)}"
)

print(
    "Original processed datasets were not modified."
)