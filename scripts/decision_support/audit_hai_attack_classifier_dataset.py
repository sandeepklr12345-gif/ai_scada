from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Paths
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

TEMPORAL_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
    / "hai-test2_temporal_model_ready.csv"
)


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def fail(message):
    raise ValueError(message)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 90)
    print("HAI 23.05 ATTACK CLASSIFIER FEATURE-TARGET ALIGNMENT AUDIT")
    print("=" * 90)

    # -----------------------------------------------------
    # Load SCADA
    # -----------------------------------------------------

    print("\n[1] Loading Test 2 SCADA...")

    scada = pd.read_csv(TEST2_SCADA_PATH)

    if "timestamp" not in scada.columns:
        fail("SCADA dataset does not contain timestamp column.")

    scada["timestamp"] = pd.to_datetime(
        scada["timestamp"]
    )

    print("SCADA shape:", scada.shape)

    # -----------------------------------------------------
    # Load classifier targets
    # -----------------------------------------------------

    print("\n[2] Loading classifier targets...")

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
            + ", ".join(sorted(missing_target_columns))
        )

    target["timestamp"] = pd.to_datetime(
        target["timestamp"]
    )

    print("Target shape:", target.shape)

    # -----------------------------------------------------
    # Basic row-count validation
    # -----------------------------------------------------

    print("\n[3] Row-count validation")

    if len(scada) != len(target):
        fail(
            f"SCADA rows ({len(scada)}) != "
            f"target rows ({len(target)})."
        )

    print(
        f"PASS: {len(scada):,} SCADA rows = "
        f"{len(target):,} target rows"
    )

    # -----------------------------------------------------
    # Timestamp alignment
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

        print(
            "\nFirst timestamp mismatches:"
        )

        print(
            pd.DataFrame({
                "scada_timestamp":
                    scada.loc[
                        timestamp_mismatch,
                        "timestamp"
                    ].head(10),

                "target_timestamp":
                    target.loc[
                        timestamp_mismatch,
                        "timestamp"
                    ].head(10),
            }).to_string(index=False)
        )

        fail(
            "SCADA and target timestamps are not aligned."
        )

    print(
        "PASS: SCADA and target timestamps "
        "are exactly aligned."
    )

    # -----------------------------------------------------
    # SCADA timestamp uniqueness
    # -----------------------------------------------------

    print("\n[5] Timestamp uniqueness")

    scada_duplicates = int(
        scada["timestamp"].duplicated().sum()
    )

    target_duplicates = int(
        target["timestamp"].duplicated().sum()
    )

    print(
        "SCADA duplicate timestamps :",
        scada_duplicates
    )

    print(
        "Target duplicate timestamps:",
        target_duplicates
    )

    if scada_duplicates != 0:
        fail(
            "SCADA contains duplicate timestamps."
        )

    if target_duplicates != 0:
        fail(
            "Target contains duplicate timestamps."
        )

    print(
        "PASS: Both datasets have unique timestamps."
    )

    # -----------------------------------------------------
    # Target distribution
    # -----------------------------------------------------

    print("\n[6] Target distribution")

    print(
        target["attack_label"]
        .value_counts()
        .sort_index()
    )

    attack_rows = int(
        target["attack_label"].sum()
    )

    normal_rows = int(
        (target["attack_label"] == 0).sum()
    )

    print(
        "\nAttack rows:",
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
        "PASS: Target distribution is correct."
    )

    # -----------------------------------------------------
    # Scenario distribution
    # -----------------------------------------------------

    print("\n[7] Scenario distribution")

    scenario_distribution = (
        target[
            target["attack_label"] == 1
        ]["scenario_id"]
        .value_counts()
        .sort_index()
    )

    print(
        scenario_distribution.to_string()
    )

    scenario_count = (
        target.loc[
            target["attack_label"] == 1,
            "scenario_id"
        ]
        .nunique()
    )

    print(
        "\nUnique attack scenarios:",
        scenario_count
    )

    if scenario_count != 38:
        fail(
            f"Expected 38 attack scenarios, "
            f"found {scenario_count}."
        )

    print(
        "PASS: All 38 scenarios are represented."
    )

    # -----------------------------------------------------
    # Combination scenarios
    # -----------------------------------------------------

    print("\n[8] Combination attack audit")

    combination_rows = target[
        target["attack_codes"].str.contains(
            "+",
            regex=False
        )
    ]

    combination_scenarios = (
        combination_rows["scenario_id"]
        .unique()
        .tolist()
    )

    print(
        "Combination scenarios:",
        len(combination_scenarios)
    )

    print(
        sorted(combination_scenarios)
    )

    if len(combination_scenarios) != 10:
        fail(
            f"Expected 10 combination scenarios, "
            f"found {len(combination_scenarios)}."
        )

    print(
        "PASS: All 10 combination scenarios preserved."
    )

    # -----------------------------------------------------
    # Check for target leakage inside SCADA columns
    # -----------------------------------------------------

    print("\n[9] Target leakage audit")

    target_columns = {
        "attack_label",
        "scenario_id",
        "attack_codes",
    }

    leaked_columns = (
        set(scada.columns)
        & target_columns
    )

    print(
        "Target columns present in SCADA:",
        leaked_columns
    )

    if leaked_columns:
        fail(
            "Target information is present inside "
            "the SCADA feature dataset."
        )

    print(
        "PASS: No classifier target columns "
        "exist in the SCADA dataset."
    )

    # -----------------------------------------------------
    # Numeric feature audit
    # -----------------------------------------------------

    print("\n[10] Numeric feature audit")

    feature_columns = [
        column
        for column in scada.columns
        if column != "timestamp"
    ]

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            scada[column]
        )
    ]

    non_numeric_features = [
        column
        for column in feature_columns
        if column not in numeric_features
    ]

    print(
        "Total SCADA columns:",
        len(scada.columns)
    )

    print(
        "Candidate numeric features:",
        len(numeric_features)
    )

    print(
        "Non-numeric columns:",
        non_numeric_features
    )

    if non_numeric_features:
        print(
            "\nNOTE: Non-numeric SCADA columns "
            "will require explicit handling."
        )

    # -----------------------------------------------------
    # Missing-value audit
    # -----------------------------------------------------

    print("\n[11] Missing-value audit")

    missing_counts = (
        scada[numeric_features]
        .isna()
        .sum()
    )

    missing_features = (
        missing_counts[
            missing_counts > 0
        ]
    )

    print(
        "Features with missing values:",
        len(missing_features)
    )

    if not missing_features.empty:

        print(
            missing_features
            .sort_values(ascending=False)
            .head(20)
            .to_string()
        )

    else:
        print(
            "PASS: No missing values in numeric features."
        )

    # -----------------------------------------------------
    # Constant feature audit
    # -----------------------------------------------------

    print("\n[12] Constant-feature audit")

    unique_counts = (
        scada[numeric_features]
        .nunique()
    )

    constant_features = (
        unique_counts[
            unique_counts <= 1
        ]
        .index
        .tolist()
    )

    print(
        "Constant numeric features:",
        len(constant_features)
    )

    if constant_features:
        print(
            constant_features
        )

    # -----------------------------------------------------
    # Temporal feature file
    # -----------------------------------------------------

    print("\n[13] Temporal feature dataset audit")

    if not TEMPORAL_FEATURE_PATH.exists():

        print(
            "Temporal feature dataset not found:"
        )

        print(
            TEMPORAL_FEATURE_PATH
        )

        print(
            "\nNo changes made."
        )

    else:

        temporal = pd.read_csv(
            TEMPORAL_FEATURE_PATH
        )

        print(
            "Temporal dataset shape:",
            temporal.shape
        )

        if "timestamp" not in temporal.columns:
            fail(
                "Temporal dataset has no timestamp column."
            )

        temporal["timestamp"] = pd.to_datetime(
            temporal["timestamp"]
        )

        # ---------------------------------------------
        # Row count
        # ---------------------------------------------

        if len(temporal) != len(target):
            fail(
                "Temporal dataset and target have "
                "different row counts."
            )

        print(
            "PASS: Temporal row count matches target."
        )

        # ---------------------------------------------
        # Timestamp alignment
        # ---------------------------------------------

        temporal_mismatch = (
            temporal["timestamp"]
            != target["timestamp"]
        )

        temporal_mismatch_count = int(
            temporal_mismatch.sum()
        )

        print(
            "Temporal timestamp mismatches:",
            temporal_mismatch_count
        )

        if temporal_mismatch_count != 0:
            fail(
                "Temporal features are not timestamp-aligned "
                "with classifier targets."
            )

        print(
            "PASS: Temporal features and targets "
            "are timestamp aligned."
        )

        # ---------------------------------------------
        # Target leakage
        # ---------------------------------------------

        temporal_target_columns = (
            set(temporal.columns)
            & target_columns
        )

        print(
            "Target columns in temporal dataset:",
            temporal_target_columns
        )

        if temporal_target_columns:
            fail(
                "Temporal dataset contains classifier "
                "target columns."
            )

        print(
            "PASS: No classifier target leakage "
            "in temporal dataset."
        )

        # ---------------------------------------------
        # Numeric features
        # ---------------------------------------------

        temporal_feature_columns = [
            column
            for column in temporal.columns
            if column != "timestamp"
        ]

        temporal_numeric_features = [
            column
            for column in temporal_feature_columns
            if pd.api.types.is_numeric_dtype(
                temporal[column]
            )
        ]

        print(
            "Temporal numeric features:",
            len(temporal_numeric_features)
        )

        # ---------------------------------------------
        # Missing values
        # ---------------------------------------------

        temporal_missing = (
            temporal[
                temporal_numeric_features
            ]
            .isna()
            .sum()
        )

        temporal_missing = temporal_missing[
            temporal_missing > 0
        ]

        print(
            "Temporal features with missing values:",
            len(temporal_missing)
        )

        if temporal_missing.empty:
            print(
                "PASS: No missing temporal feature values."
            )
        else:
            print(
                temporal_missing
                .sort_values(
                    ascending=False
                )
                .head(20)
                .to_string()
            )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 90)
    print("AUDIT SUMMARY")
    print("=" * 90)

    print(
        "\nSCADA rows              :",
        len(scada)
    )

    print(
        "Target rows             :",
        len(target)
    )

    print(
        "Attack rows             :",
        attack_rows
    )

    print(
        "Normal rows             :",
        normal_rows
    )

    print(
        "Attack scenarios        :",
        scenario_count
    )

    print(
        "Combination scenarios   :",
        len(combination_scenarios)
    )

    print(
        "Numeric SCADA features  :",
        len(numeric_features)
    )

    print("\n" + "=" * 90)
    print("FEATURE-TARGET ALIGNMENT AUDIT COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()