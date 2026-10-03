from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test2_attack_classifier_dataset.csv"
)


# ---------------------------------------------------------
# Output
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
    / "hai_test2_multilabel_classifier_dataset.csv"
)


# ---------------------------------------------------------
# Official attack-code vocabulary
#
# This is deliberately defined explicitly from the
# documented Test 2 scenario catalog.
# ---------------------------------------------------------

ATTACK_CODES = [
    "AP01",
    "AP02",
    "AP03",
    "AP04",
    "AP05",
    "AP07",
    "AP08",
    "AP09",
    "AP11",
    "AP13",
    "AP14",
    "AP15",
    "AP16",
    "AP17",
    "AP18",
    "AP19",
    "AP20",
    "AP21",
    "AP22",
    "AP23",
    "AP24",
    "AP25",
    "AP26",
    "AP27",
    "AP30",
    "AP32",
    "AP35",
    "AP42",
    "AP43",
    "AP44",
    "AP45",
    "AP46",
    "AP47",
    "AE01",
    "AE03",
    "AE05",
    "AE06",
    "AE07",
    "AE08",
]


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 90)
    print("HAI 23.05 TEST 2 MULTI-LABEL TARGET CONSTRUCTION")
    print("=" * 90)

    # -----------------------------------------------------
    # Load classifier dataset
    # -----------------------------------------------------

    print("\n[1] Loading classifier dataset...")

    data = pd.read_csv(INPUT_PATH)

    print(
        "Input shape:",
        data.shape
    )

    required_columns = {
        "timestamp",
        "attack_label",
        "scenario_id",
        "attack_codes",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        fail(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    # -----------------------------------------------------
    # Identify model features
    # -----------------------------------------------------

    metadata_columns = {
        "timestamp",
        "attack_label",
        "scenario_id",
        "attack_codes",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in metadata_columns
    ]

    print(
        "Model features:",
        len(feature_columns)
    )

    if len(feature_columns) != 68:
        fail(
            f"Expected 68 model features, "
            f"found {len(feature_columns)}."
        )

    # -----------------------------------------------------
    # Validate attack-code vocabulary
    # -----------------------------------------------------

    print("\n[2] Validating attack-code vocabulary")

    observed_codes = set()

    for value in data["attack_codes"]:

        if value == "NORMAL":
            continue

        for code in value.split("+"):
            observed_codes.add(code)

    observed_codes = sorted(observed_codes)

    print(
        "Observed attack codes:",
        len(observed_codes)
    )

    print(
        observed_codes
    )

    undocumented_codes = (
        set(observed_codes)
        - set(ATTACK_CODES)
    )

    if undocumented_codes:
        fail(
            "Undocumented attack codes found: "
            + ", ".join(sorted(undocumented_codes))
        )

    missing_codes = (
        set(ATTACK_CODES)
        - set(observed_codes)
    )

    if missing_codes:
        print(
            "\nWARNING: Documented attack codes not "
            "observed in Test 2:"
        )

        print(
            sorted(missing_codes)
        )

    print(
        "PASS: All observed attack codes are "
        "documented."
    )

    # -----------------------------------------------------
    # Create binary multi-label columns
    # -----------------------------------------------------

    print("\n[3] Creating multi-label target columns")

    for code in ATTACK_CODES:

        data[code] = (
            data["attack_codes"]
            .apply(
                lambda value:
                    0
                    if value == "NORMAL"
                    else int(
                        code
                        in value.split("+")
                    )
            )
            .astype("int8")
        )

    print(
        "Created target labels:",
        len(ATTACK_CODES)
    )

    # -----------------------------------------------------
    # Validation: NORMAL rows
    # -----------------------------------------------------

    print("\n[4] Validating NORMAL rows")

    normal_mask = (
        data["attack_label"] == 0
    )

    normal_target_sum = (
        data.loc[
            normal_mask,
            ATTACK_CODES
        ]
        .sum(axis=1)
    )

    invalid_normal_rows = (
        normal_target_sum != 0
    )

    invalid_normal_count = int(
        invalid_normal_rows.sum()
    )

    print(
        "Invalid NORMAL rows:",
        invalid_normal_count
    )

    if invalid_normal_count != 0:
        fail(
            "NORMAL rows contain positive attack-code targets."
        )

    print(
        "PASS: All NORMAL rows have zero attack-code labels."
    )

    # -----------------------------------------------------
    # Validation: attack rows
    # -----------------------------------------------------

    print("\n[5] Validating attack rows")

    attack_mask = (
        data["attack_label"] == 1
    )

    attack_target_sum = (
        data.loc[
            attack_mask,
            ATTACK_CODES
        ]
        .sum(axis=1)
    )

    invalid_attack_rows = (
        attack_target_sum < 1
    )

    invalid_attack_count = int(
        invalid_attack_rows.sum()
    )

    print(
        "Attack rows without a target code:",
        invalid_attack_count
    )

    if invalid_attack_count != 0:
        fail(
            "Some attack rows have no positive "
            "attack-code target."
        )

    print(
        "PASS: Every attack row has at least "
        "one positive attack-code target."
    )

    # -----------------------------------------------------
    # Validation: single vs combination attacks
    # -----------------------------------------------------

    print(
        "\n[6] Validating single and combination attacks"
    )

    target_counts = (
        data[ATTACK_CODES]
        .sum(axis=1)
    )

    attack_code_counts = (
        data.loc[
            attack_mask
        ]
        .assign(
            target_count=
                target_counts.loc[attack_mask]
        )[
            [
                "scenario_id",
                "attack_codes",
                "target_count",
            ]
        ]
        .drop_duplicates()
    )

    invalid_count_rows = (
        attack_code_counts.apply(
            lambda row:
                row["target_count"]
                != len(
                    row["attack_codes"].split("+")
                ),
            axis=1,
        )
    )

    invalid_count = int(
        invalid_count_rows.sum()
    )

    if invalid_count != 0:

        print(
            attack_code_counts[
                invalid_count_rows
            ].to_string(index=False)
        )

        fail(
            "Multi-label target count does not "
            "match attack-code definition."
        )

    print(
        "PASS: Single and combination attack "
        "targets are correctly encoded."
    )

    # -----------------------------------------------------
    # Validation: specific combination scenarios
    # -----------------------------------------------------

    print(
        "\n[7] Validating documented combinations"
    )

    expected_combinations = {
        "A224": ["AP14", "AP26"],
        "A225": ["AP16", "AP32"],
        "A226": ["AP04", "AP11"],
        "A227": ["AP09", "AP14"],
        "A228": ["AP05", "AP30"],
        "A229": ["AP45", "AP01"],
        "A230": ["AP19", "AP02"],
        "A231": ["AP08", "AP35"],
        "A232": ["AP45", "AP27"],
        "A233": ["AP44", "AP47"],
    }

    for scenario_id, expected_codes in (
        expected_combinations.items()
    ):

        scenario_rows = data[
            data["scenario_id"] == scenario_id
        ]

        if scenario_rows.empty:
            fail(
                f"{scenario_id} not found."
            )

        for code in ATTACK_CODES:

            expected_value = int(
                code in expected_codes
            )

            actual_values = set(
                scenario_rows[code]
            )

            if actual_values != {expected_value}:

                fail(
                    f"{scenario_id}: {code} "
                    f"expected {expected_value}, "
                    f"found {actual_values}"
                )

    print(
        "PASS: All 10 combination scenarios "
        "encoded correctly."
    )

    # -----------------------------------------------------
    # Validation: row count
    # -----------------------------------------------------

    print("\n[8] Row-count validation")

    if len(data) != 230400:
        fail(
            f"Expected 230,400 rows, "
            f"found {len(data)}."
        )

    print(
        "PASS: 230,400 rows preserved."
    )

    # -----------------------------------------------------
    # Validation: target distribution
    # -----------------------------------------------------

    print("\n[9] Multi-label distribution")

    print(
        "\nNumber of positive attack codes per row:"
    )

    print(
        target_counts
        .value_counts()
        .sort_index()
        .to_string()
    )

    normal_count = int(
        (target_counts == 0).sum()
    )

    single_attack_count = int(
        (target_counts == 1).sum()
    )

    combination_count = int(
        (target_counts > 1).sum()
    )

    print(
        "\nZero attack codes :",
        normal_count
    )

    print(
        "One attack code   :",
        single_attack_count
    )

    print(
        "Multiple codes    :",
        combination_count
    )

    if normal_count != 220500:
        fail(
            f"Expected 220,500 zero-label rows, "
            f"found {normal_count}."
        )

    if single_attack_count + combination_count != 9900:
        fail(
            "Multi-label attack rows do not sum "
            "to 9,900."
        )

    print(
        "PASS: Multi-label row distribution is correct."
    )

    # -----------------------------------------------------
    # Per-code distribution
    # -----------------------------------------------------

    print("\n[10] Attack-code distribution")

    code_distribution = (
        data[ATTACK_CODES]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        code_distribution.to_string()
    )

    # -----------------------------------------------------
    # Verify metadata columns remain unchanged
    # -----------------------------------------------------

    print(
        "\n[11] Metadata preservation"
    )

    if data["attack_label"].sum() != 9900:
        fail(
            "attack_label was modified unexpectedly."
        )

    if data["scenario_id"].eq(
        "NORMAL"
    ).sum() != 220500:
        fail(
            "scenario_id NORMAL count changed."
        )

    print(
        "PASS: Existing metadata preserved."
    )

    # -----------------------------------------------------
    # Verify no feature leakage
    # -----------------------------------------------------

    print(
        "\n[12] Feature-target separation"
    )

    target_columns = (
        set(ATTACK_CODES)
        | {
            "attack_label",
            "scenario_id",
            "attack_codes",
        }
    )

    overlap = (
        set(feature_columns)
        & target_columns
    )

    if overlap:
        fail(
            "Feature/target overlap detected: "
            + ", ".join(sorted(overlap))
        )

    print(
        "PASS: Feature and target columns are separated."
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print("\n[13] Saving multi-label dataset")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL MULTI-LABEL DATASET SUMMARY")
    print("=" * 90)

    print(
        "\nRows:",
        len(data)
    )

    print(
        "Original model features:",
        len(feature_columns)
    )

    print(
        "Attack-code targets:",
        len(ATTACK_CODES)
    )

    print(
        "Normal rows:",
        normal_count
    )

    print(
        "Attack rows:",
        single_attack_count + combination_count
    )

    print(
        "Single-code attack rows:",
        single_attack_count
    )

    print(
        "Multi-code attack rows:",
        combination_count
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )

    print("\n" + "=" * 90)
    print("ALL MULTI-LABEL VALIDATIONS PASSED")
    print("STEP 6.6 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()