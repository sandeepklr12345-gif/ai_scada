from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Input paths
# ---------------------------------------------------------

TEST2_SCADA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
    / "hai-test2.csv"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test2_classifier_targets.csv"
)


# ---------------------------------------------------------
# Output path
# ---------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "hai_test2_attack_classifier_dataset.csv"
)


# ---------------------------------------------------------
# Constant features identified during Step 6.5A audit
# ---------------------------------------------------------

CONSTANT_FEATURES = [
    "P1_PIT01_HH",
    "P1_PP01AD",
    "P1_PP01AR",
    "P1_PP01BD",
    "P1_PP01BR",
    "P1_PP02D",
    "P1_PP02R",
    "P1_SOL01D",
    "P1_SOL03D",
    "P1_STSP",
    "P2_RTR",
    "P2_TripEx",
    "P2_VTR01",
    "P2_VTR02",
    "P2_VTR03",
    "P2_VTR04",
    "P3_LH01",
    "P3_LL01",
]


TARGET_COLUMNS = [
    "attack_label",
    "scenario_id",
    "attack_codes",
]


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 90)
    print("HAI 23.05 TEST 2 ATTACK CLASSIFIER DATASET PREPARATION")
    print("=" * 90)

    # -----------------------------------------------------
    # Load SCADA
    # -----------------------------------------------------

    print("\n[1] Loading Test 2 SCADA...")

    scada = pd.read_csv(TEST2_SCADA_PATH)

    if "timestamp" not in scada.columns:
        fail(
            "SCADA dataset does not contain timestamp column."
        )

    scada["timestamp"] = pd.to_datetime(
        scada["timestamp"]
    )

    print(
        "SCADA shape:",
        scada.shape
    )

    # -----------------------------------------------------
    # Load validated targets
    # -----------------------------------------------------

    print("\n[2] Loading validated classifier targets...")

    target = pd.read_csv(TARGET_PATH)

    required_target_columns = {
        "timestamp",
        "attack_label",
        "scenario_id",
        "attack_codes",
    }

    missing_target_columns = (
        required_target_columns
        - set(target.columns)
    )

    if missing_target_columns:
        fail(
            "Target dataset missing columns: "
            + ", ".join(
                sorted(missing_target_columns)
            )
        )

    target["timestamp"] = pd.to_datetime(
        target["timestamp"]
    )

    print(
        "Target shape:",
        target.shape
    )

    # -----------------------------------------------------
    # Validate row count
    # -----------------------------------------------------

    print("\n[3] Row-count validation")

    if len(scada) != len(target):
        fail(
            f"SCADA rows ({len(scada)}) != "
            f"target rows ({len(target)})."
        )

    print(
        f"PASS: {len(scada):,} rows aligned."
    )

    # -----------------------------------------------------
    # Validate timestamp alignment
    # -----------------------------------------------------

    print("\n[4] Timestamp alignment")

    timestamp_mismatch = (
        scada["timestamp"]
        != target["timestamp"]
    )

    mismatch_count = int(
        timestamp_mismatch.sum()
    )

    print(
        "Timestamp mismatches:",
        mismatch_count
    )

    if mismatch_count != 0:
        fail(
            "SCADA and target timestamps are not aligned."
        )

    print(
        "PASS: Timestamp alignment verified."
    )

    # -----------------------------------------------------
    # Validate uniqueness
    # -----------------------------------------------------

    print("\n[5] Timestamp uniqueness")

    if scada["timestamp"].duplicated().any():
        fail(
            "SCADA contains duplicate timestamps."
        )

    if target["timestamp"].duplicated().any():
        fail(
            "Target contains duplicate timestamps."
        )

    print(
        "PASS: Timestamps are unique."
    )

    # -----------------------------------------------------
    # Identify original SCADA features
    # -----------------------------------------------------

    print("\n[6] Identifying SCADA features")

    scada_feature_columns = [
        column
        for column in scada.columns
        if column != "timestamp"
    ]

    print(
        "Original SCADA features:",
        len(scada_feature_columns)
    )

    if len(scada_feature_columns) != 86:
        fail(
            f"Expected 86 original SCADA features, "
            f"found {len(scada_feature_columns)}."
        )

    # -----------------------------------------------------
    # Verify all features are numeric
    # -----------------------------------------------------

    non_numeric_features = [
        column
        for column in scada_feature_columns
        if not pd.api.types.is_numeric_dtype(
            scada[column]
        )
    ]

    if non_numeric_features:
        fail(
            "Non-numeric SCADA features found: "
            + ", ".join(non_numeric_features)
        )

    print(
        "PASS: All 86 SCADA features are numeric."
    )

    # -----------------------------------------------------
    # Verify constant features exist
    # -----------------------------------------------------

    print("\n[7] Validating constant features")

    missing_constant_features = (
        set(CONSTANT_FEATURES)
        - set(scada_feature_columns)
    )

    if missing_constant_features:
        fail(
            "Constant features missing from SCADA: "
            + ", ".join(
                sorted(missing_constant_features)
            )
        )

    print(
        "PASS: All 18 audited constant features exist."
    )

    # -----------------------------------------------------
    # Verify they are actually constant
    # -----------------------------------------------------

    print(
        "\n[8] Re-validating constant feature values"
    )

    non_constant_features = []

    for feature in CONSTANT_FEATURES:

        unique_count = (
            scada[feature]
            .nunique(dropna=False)
        )

        if unique_count > 1:
            non_constant_features.append(
                (feature, unique_count)
            )

    if non_constant_features:

        print(
            "\nUnexpected non-constant features:"
        )

        for feature, count in non_constant_features:
            print(
                f"  {feature}: {count} unique values"
            )

        fail(
            "One or more audited constant features "
            "are no longer constant."
        )

    print(
        "PASS: All 18 features are constant."
    )

    # -----------------------------------------------------
    # Select variable features
    # -----------------------------------------------------

    variable_features = [
        column
        for column in scada_feature_columns
        if column not in CONSTANT_FEATURES
    ]

    print(
        "\nVariable features:",
        len(variable_features)
    )

    if len(variable_features) != 68:
        fail(
            f"Expected 68 variable features, "
            f"found {len(variable_features)}."
        )

    print(
        "PASS: 68 variable classifier features selected."
    )

    # -----------------------------------------------------
    # Check missing values
    # -----------------------------------------------------

    print("\n[9] Missing-value validation")

    missing_counts = (
        scada[variable_features]
        .isna()
        .sum()
    )

    missing_features = (
        missing_counts[
            missing_counts > 0
        ]
    )

    if not missing_features.empty:

        print(
            missing_features
            .sort_values(ascending=False)
            .to_string()
        )

        fail(
            "Missing values detected in classifier features."
        )

    print(
        "PASS: No missing values in 68 features."
    )

    # -----------------------------------------------------
    # Construct feature dataframe
    # -----------------------------------------------------

    print("\n[10] Constructing classifier dataset")

    feature_data = scada[
        ["timestamp"] + variable_features
    ].copy()

    target_data = target[
        [
            "timestamp",
            "attack_label",
            "scenario_id",
            "attack_codes",
        ]
    ].copy()

    # -----------------------------------------------------
    # Merge using timestamp
    # -----------------------------------------------------

    classifier_dataset = feature_data.merge(
        target_data,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    print(
        "Merged dataset shape:",
        classifier_dataset.shape
    )

    expected_columns = (
        1
        + 68
        + 3
    )

    if classifier_dataset.shape[1] != expected_columns:
        fail(
            f"Expected {expected_columns} columns, "
            f"found {classifier_dataset.shape[1]}."
        )

    print(
        f"PASS: Dataset contains "
        f"{expected_columns} columns."
    )

    # -----------------------------------------------------
    # Verify no rows were lost
    # -----------------------------------------------------

    if len(classifier_dataset) != len(scada):
        fail(
            "Rows were lost during feature-target merge."
        )

    print(
        "PASS: No rows lost during merge."
    )

    # -----------------------------------------------------
    # Verify target distribution
    # -----------------------------------------------------

    print("\n[11] Target validation")

    attack_rows = int(
        classifier_dataset["attack_label"].sum()
    )

    normal_rows = int(
        (
            classifier_dataset["attack_label"]
            == 0
        ).sum()
    )

    print(
        "Attack rows:",
        attack_rows
    )

    print(
        "Normal rows:",
        normal_rows
    )

    if attack_rows != 9900:
        fail(
            f"Expected 9,900 attack rows, "
            f"found {attack_rows}."
        )

    if normal_rows != 220500:
        fail(
            f"Expected 220,500 normal rows, "
            f"found {normal_rows}."
        )

    print(
        "PASS: Target distribution preserved."
    )

    # -----------------------------------------------------
    # Verify scenario count
    # -----------------------------------------------------

    scenario_count = (
        classifier_dataset.loc[
            classifier_dataset["attack_label"] == 1,
            "scenario_id"
        ]
        .nunique()
    )

    print(
        "Attack scenarios:",
        scenario_count
    )

    if scenario_count != 38:
        fail(
            f"Expected 38 attack scenarios, "
            f"found {scenario_count}."
        )

    print(
        "PASS: All 38 scenarios preserved."
    )

    # -----------------------------------------------------
    # Verify combinations
    # -----------------------------------------------------

    combination_count = (
        classifier_dataset[
            classifier_dataset["attack_codes"]
            .str.contains("+", regex=False)
        ]["scenario_id"]
        .nunique()
    )

    print(
        "Combination scenarios:",
        combination_count
    )

    if combination_count != 10:
        fail(
            f"Expected 10 combination scenarios, "
            f"found {combination_count}."
        )

    print(
        "PASS: All 10 combination scenarios preserved."
    )

    # -----------------------------------------------------
    # Target leakage check
    # -----------------------------------------------------

    print("\n[12] Target leakage validation")

    feature_set = set(variable_features)

    leakage = (
        feature_set
        & set(TARGET_COLUMNS)
    )

    if leakage:
        fail(
            "Target columns found in feature set: "
            + ", ".join(sorted(leakage))
        )

    print(
        "PASS: No target columns in feature matrix."
    )

    # -----------------------------------------------------
    # Verify constant features are excluded
    # -----------------------------------------------------

    excluded_constants = (
        set(CONSTANT_FEATURES)
        & set(classifier_dataset.columns)
    )

    if excluded_constants:
        fail(
            "Constant features were not excluded: "
            + ", ".join(
                sorted(excluded_constants)
            )
        )

    print(
        "PASS: All 18 constant features excluded."
    )

    # -----------------------------------------------------
    # Final numeric feature check
    # -----------------------------------------------------

    final_feature_columns = [
        column
        for column in classifier_dataset.columns
        if column not in {
            "timestamp",
            "attack_label",
            "scenario_id",
            "attack_codes",
        }
    ]

    non_numeric_final = [
        column
        for column in final_feature_columns
        if not pd.api.types.is_numeric_dtype(
            classifier_dataset[column]
        )
    ]

    if non_numeric_final:
        fail(
            "Non-numeric classifier features found: "
            + ", ".join(non_numeric_final)
        )

    print(
        "PASS: All 68 classifier features are numeric."
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    classifier_dataset.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL CLASSIFIER DATASET SUMMARY")
    print("=" * 90)

    print(
        "\nRows:",
        len(classifier_dataset)
    )

    print(
        "Columns:",
        len(classifier_dataset.columns)
    )

    print(
        "Model features:",
        len(final_feature_columns)
    )

    print(
        "Attack rows:",
        attack_rows
    )

    print(
        "Normal rows:",
        normal_rows
    )

    print(
        "Attack scenarios:",
        scenario_count
    )

    print(
        "Combination scenarios:",
        combination_count
    )

    print(
        "\nExcluded constant features:",
        len(CONSTANT_FEATURES)
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )

    print("\n" + "=" * 90)
    print("ALL CLASSIFIER DATASET VALIDATIONS PASSED")
    print("STEP 6.5B COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()