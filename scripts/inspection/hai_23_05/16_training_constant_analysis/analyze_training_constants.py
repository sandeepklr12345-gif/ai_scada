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

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
)

ROLE_FILE = FEATURE_DIR / "hai_2305_feature_roles.csv"

TRAINING_VARIABILITY_FILE = (
    TRAINING_DIR
    / "hai_2305_training_feature_variability.csv"
)

TRAINING_DATA_FILE = (
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

OUTPUT_FILE = (
    TRAINING_DIR
    / "hai_2305_training_constant_analysis.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 16: TRAINING-CONSTANT FEATURE ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD METADATA
# ============================================================

roles = pd.read_csv(ROLE_FILE)

training_variability = pd.read_csv(
    TRAINING_VARIABILITY_FILE
)

print("\nLoading training data...")
training_data = pd.read_csv(
    TRAINING_DATA_FILE
)

print("Loading Test 1...")
test1 = pd.read_csv(TEST1_FILE)

print("Loading Test 2...")
test2 = pd.read_csv(TEST2_FILE)


print("\nFiles loaded successfully.")


# ============================================================
# BASIC VALIDATION
# ============================================================

assert "feature" in training_variability.columns
assert "unique_values" in training_variability.columns
assert "constant_training" in training_variability.columns
assert "minimum" in training_variability.columns
assert "maximum" in training_variability.columns

assert "feature" in roles.columns
assert "role" in roles.columns

print("PASS: Training variability schema")
print("PASS: Feature-role schema")


# ============================================================
# IDENTIFY TRAINING-CONSTANT FEATURES
# ============================================================

constant_training_df = training_variability[
    training_variability["constant_training"] == True
].copy()

constant_training_features = (
    constant_training_df["feature"].tolist()
)

print(
    f"\nTraining-constant features: "
    f"{len(constant_training_features)}"
)


# ============================================================
# VERIFY ALL FEATURES EXIST
# ============================================================

for feature in constant_training_features:

    assert feature in training_data.columns, (
        f"{feature} missing from training data"
    )

    assert feature in test1.columns, (
        f"{feature} missing from Test 1"
    )

    assert feature in test2.columns, (
        f"{feature} missing from Test 2"
    )


print(
    "PASS: All training-constant features exist "
    "in training, Test 1 and Test 2"
)


# ============================================================
# ROLE LOOKUP
# ============================================================

role_lookup = dict(
    zip(
        roles["feature"],
        roles["role"]
    )
)


# ============================================================
# ANALYSIS
# ============================================================

results = []

for feature in constant_training_features:

    # --------------------------------------------------------
    # Training statistics
    # --------------------------------------------------------

    training_row = constant_training_df[
        constant_training_df["feature"] == feature
    ].iloc[0]

    training_unique = int(
        training_row["unique_values"]
    )

    training_min = training_row["minimum"]
    training_max = training_row["maximum"]

    # Since the feature is constant in training,
    # minimum and maximum should be identical.
    assert training_min == training_max, (
        f"{feature} is marked constant but "
        f"minimum != maximum"
    )

    # --------------------------------------------------------
    # Test statistics
    # --------------------------------------------------------

    test1_unique = int(
        test1[feature].nunique(
            dropna=False
        )
    )

    test2_unique = int(
        test2[feature].nunique(
            dropna=False
        )
    )

    # --------------------------------------------------------
    # Variability flags
    # --------------------------------------------------------

    variable_test1 = test1_unique > 1
    variable_test2 = test2_unique > 1

    constant_test1 = test1_unique == 1
    constant_test2 = test2_unique == 1

    constant_everywhere = (
        training_unique == 1
        and constant_test1
        and constant_test2
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if constant_everywhere:

        classification = (
            "Constant in training, Test1 and Test2"
        )

    elif variable_test1 and variable_test2:

        classification = (
            "Constant in training, variable in Test1 and Test2"
        )

    elif variable_test1:

        classification = (
            "Constant in training, variable in Test1"
        )

    elif variable_test2:

        classification = (
            "Constant in training, variable in Test2"
        )

    else:

        classification = "Requires review"

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        {
            "feature": feature,
            "role": role_lookup.get(
                feature,
                "Unknown"
            ),
            "training_unique_values": training_unique,
            "training_constant_value": training_min,
            "test1_unique_values": test1_unique,
            "test2_unique_values": test2_unique,
            "variable_test1": variable_test1,
            "variable_test2": variable_test2,
            "constant_in_all_available_datasets":
                constant_everywhere,
            "classification":
                classification,
        }
    )


# ============================================================
# RESULT DATAFRAME
# ============================================================

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    by=[
        "constant_in_all_available_datasets",
        "feature"
    ],
    ascending=[
        False,
        True
    ]
).reset_index(drop=True)


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

constant_everywhere = result_df[
    result_df[
        "constant_in_all_available_datasets"
    ] == True
]

variable_in_tests = result_df[
    result_df[
        "constant_in_all_available_datasets"
    ] == False
]


print("\n" + "=" * 70)
print("STAGE 16 RESULTS")
print("=" * 70)

print(
    f"\nTraining-constant features : "
    f"{len(result_df)}"
)

print(
    f"Constant everywhere        : "
    f"{len(constant_everywhere)}"
)

print(
    f"Training-constant but "
    f"test-variable             : "
    f"{len(variable_in_tests)}"
)


# ============================================================
# PRINT DETAILED TABLE
# ============================================================

print("\n" + "-" * 70)
print("FEATURE ANALYSIS")
print("-" * 70)

print(
    result_df[
        [
            "feature",
            "role",
            "training_unique_values",
            "training_constant_value",
            "test1_unique_values",
            "test2_unique_values",
            "variable_test1",
            "variable_test2",
            "classification",
        ]
    ].to_string(index=False)
)


# ============================================================
# IMPORTANT FEATURES
# ============================================================

print("\n" + "-" * 70)
print("TRAINING-CONSTANT BUT TEST-VARIABLE FEATURES")
print("-" * 70)

if len(variable_in_tests) > 0:

    print(
        variable_in_tests[
            [
                "feature",
                "role",
                "training_constant_value",
                "test1_unique_values",
                "test2_unique_values",
                "classification",
            ]
        ].to_string(index=False)
    )

else:

    print("None")


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

assert len(result_df) == 20

assert result_df["feature"].is_unique

assert result_df[
    "training_unique_values"
].eq(1).all()

assert result_df[
    "constant_in_all_available_datasets"
].sum() == 18

assert (
    (
        result_df["feature"] == "P2_Emerg"
    )
    &
    (~result_df["variable_test1"])
    &
    result_df["variable_test2"]
).any()

assert (
    (
        result_df["feature"] == "P2_OnOff"
    )
    &
    (~result_df["variable_test1"])
    &
    result_df["variable_test2"]
).any()

print("PASS: Exactly 20 training-constant features")
print("PASS: No duplicate features")
print("PASS: All 20 have one training value")
print("PASS: 18 are constant everywhere")
print("PASS: P2_Emerg is variable in both tests")
print("PASS: P2_OnOff is variable in both tests")

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STAGE 16: COMPLETE")
print("=" * 70)