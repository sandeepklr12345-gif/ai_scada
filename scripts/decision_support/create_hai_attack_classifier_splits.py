from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test2_multilabel_classifier_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
)

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 90)
    print("HAI 23.05 ATTACK CLASSIFIER SPLIT CREATION")
    print("=" * 90)

    # -----------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------

    print("\n[1] Loading validated multi-label dataset...")

    data = pd.read_csv(INPUT_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print("Input shape:", data.shape)

    # -----------------------------------------------------
    # 2. Basic validation
    # -----------------------------------------------------

    print("\n[2] Validating input")

    if data["timestamp"].duplicated().any():
        fail("Duplicate timestamps detected.")

    if not data["timestamp"].is_monotonic_increasing:
        data = (
            data
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    if data["attack_label"].isna().any():
        fail("Missing attack_label values detected.")

    print("Rows:", len(data))
    print(
        "Start:",
        data["timestamp"].min()
    )
    print(
        "End:",
        data["timestamp"].max()
    )

    # -----------------------------------------------------
    # 3. Define event-aware chronological boundaries
    #
    # These boundaries were validated in Step 6.7.
    # -----------------------------------------------------

    print(
        "\n[3] Defining validated chronological boundaries"
    )

    train_start = data["timestamp"].min()

    validation_start = pd.Timestamp(
        "2022-08-18 14:24:00.400000"
    )

    test_start = pd.Timestamp(
        "2022-08-19 03:12:00.200000"
    )

    test_end = data["timestamp"].max()

    print(
        "TRAIN:",
        train_start,
        "→",
        validation_start
    )

    print(
        "VALIDATION:",
        validation_start,
        "→",
        test_start
    )

    print(
        "TEST:",
        test_start,
        "→",
        test_end
    )

    # -----------------------------------------------------
    # 4. Create splits
    # -----------------------------------------------------

    print("\n[4] Creating splits")

    train = data[
        (data["timestamp"] >= train_start)
        & (data["timestamp"] < validation_start)
    ].copy()

    validation = data[
        (data["timestamp"] >= validation_start)
        & (data["timestamp"] < test_start)
    ].copy()

    test = data[
        (data["timestamp"] >= test_start)
        & (data["timestamp"] <= test_end)
    ].copy()

    print(
        "Train shape:",
        train.shape
    )

    print(
        "Validation shape:",
        validation.shape
    )

    print(
        "Test shape:",
        test.shape
    )

    # -----------------------------------------------------
    # 5. Row-count validation
    # -----------------------------------------------------

    print("\n[5] Row-count validation")

    total_split_rows = (
        len(train)
        + len(validation)
        + len(test)
    )

    print(
        "Original rows:",
        len(data)
    )

    print(
        "Split rows:",
        total_split_rows
    )

    if total_split_rows != len(data):
        fail(
            "Split row count does not equal "
            "original row count."
        )

    print(
        "PASS: All rows preserved."
    )

    # -----------------------------------------------------
    # 6. Timestamp overlap validation
    # -----------------------------------------------------

    print("\n[6] Timestamp overlap validation")

    train_times = set(train["timestamp"])
    validation_times = set(validation["timestamp"])
    test_times = set(test["timestamp"])

    train_validation = (
        train_times & validation_times
    )

    train_test = (
        train_times & test_times
    )

    validation_test = (
        validation_times & test_times
    )

    print(
        "Train / Validation overlap:",
        len(train_validation)
    )

    print(
        "Train / Test overlap:",
        len(train_test)
    )

    print(
        "Validation / Test overlap:",
        len(validation_test)
    )

    if (
        train_validation
        or train_test
        or validation_test
    ):
        fail(
            "Timestamp overlap detected."
        )

    print(
        "PASS: No timestamp overlap."
    )

    # -----------------------------------------------------
    # 7. Scenario leakage validation
    # -----------------------------------------------------

    print("\n[7] Scenario leakage validation")

    def attack_scenarios(frame):

        return set(
            frame.loc[
                frame["attack_label"] == 1,
                "scenario_id"
            ]
            .dropna()
            .unique()
        )

    train_scenarios = attack_scenarios(train)
    validation_scenarios = attack_scenarios(
        validation
    )
    test_scenarios = attack_scenarios(test)

    print(
        "Train scenarios:",
        sorted(train_scenarios)
    )

    print(
        "Validation scenarios:",
        sorted(validation_scenarios)
    )

    print(
        "Test scenarios:",
        sorted(test_scenarios)
    )

    overlap_tv = (
        train_scenarios
        & validation_scenarios
    )

    overlap_tt = (
        train_scenarios
        & test_scenarios
    )

    overlap_vt = (
        validation_scenarios
        & test_scenarios
    )

    if overlap_tv:
        fail(
            f"Train/validation scenario leakage: "
            f"{overlap_tv}"
        )

    if overlap_tt:
        fail(
            f"Train/test scenario leakage: "
            f"{overlap_tt}"
        )

    if overlap_vt:
        fail(
            f"Validation/test scenario leakage: "
            f"{overlap_vt}"
        )

    print(
        "PASS: No scenario leakage."
    )

    # -----------------------------------------------------
    # 8. Attack / normal distribution
    # -----------------------------------------------------

    print(
        "\n[8] Attack / normal distribution"
    )

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        attack_count = int(
            (frame["attack_label"] == 1).sum()
        )

        normal_count = int(
            (frame["attack_label"] == 0).sum()
        )

        print(
            f"{name}: "
            f"rows={len(frame)}, "
            f"normal={normal_count}, "
            f"attack={attack_count}"
        )

    # -----------------------------------------------------
    # 9. Attack-code coverage
    # -----------------------------------------------------

    print(
        "\n[9] Attack-code coverage"
    )

    target_columns = [
        column
        for column in data.columns
        if column.startswith("AP")
        or column.startswith("AE")
    ]

    print(
        "Target columns:",
        len(target_columns)
    )

    all_codes = set()

    for column in target_columns:

        if column in data.columns:
            all_codes.add(column)

    print(
        "Total documented/observed codes:",
        len(all_codes)
    )

    def active_codes(frame):

        codes = set()

        for column in target_columns:

            if column not in frame.columns:
                continue

            if frame[column].sum() > 0:
                codes.add(column)

        return codes

    train_codes = active_codes(train)
    validation_codes = active_codes(validation)
    test_codes = active_codes(test)

    print(
        "\nTRAIN active codes:",
        sorted(train_codes)
    )

    print(
        "\nVALIDATION active codes:",
        sorted(validation_codes)
    )

    print(
        "\nTEST active codes:",
        sorted(test_codes)
    )

    print(
        "\nCodes absent from TEST:"
    )

    print(
        sorted(all_codes - test_codes)
    )

    # -----------------------------------------------------
    # 10. Combination attack validation
    # -----------------------------------------------------

    print(
        "\n[10] Combination attack validation"
    )

    combination_rows = data[
        data["attack_codes"].astype(str).str.contains(
            "+",
            regex=False
        )
    ]

    expected_combination_scenarios = sorted(
        combination_rows["scenario_id"]
        .dropna()
        .unique()
    )

    print(
        "Total combination scenarios:",
        len(expected_combination_scenarios)
    )

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        combinations = sorted(
            frame.loc[
                frame["attack_codes"]
                .astype(str)
                .str.contains(
                    "+",
                    regex=False
                ),
                "scenario_id"
            ]
            .dropna()
            .unique()
        )

        print(
            f"{name} combination scenarios:",
            combinations
        )

    # -----------------------------------------------------
    # 11. Feature-target separation
    # -----------------------------------------------------

    print(
        "\n[11] Feature-target separation"
    )

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
        and column not in target_columns
    ]

    print(
        "Feature columns:",
        len(feature_columns)
    )

    print(
        "Target columns:",
        len(target_columns)
    )

    leakage_columns = (
        set(feature_columns)
        & metadata_columns
    ) | (
        set(feature_columns)
        & set(target_columns)
    )

    if leakage_columns:
        fail(
            f"Feature/target leakage detected: "
            f"{leakage_columns}"
        )

    print(
        "PASS: Feature-target separation valid."
    )

    # -----------------------------------------------------
    # 12. Output directory
    # -----------------------------------------------------

    print(
        "\n[12] Creating output directory"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 13. Save splits
    # -----------------------------------------------------

    print("\n[13] Saving split datasets")

    train.to_csv(
        TRAIN_PATH,
        index=False
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False
    )

    test.to_csv(
        TEST_PATH,
        index=False
    )

    print(
        "TRAIN:",
        TRAIN_PATH
    )

    print(
        "VALIDATION:",
        VALIDATION_PATH
    )

    print(
        "TEST:",
        TEST_PATH
    )

    # -----------------------------------------------------
    # 14. Final verification
    # -----------------------------------------------------

    print(
        "\n[14] Final verification"
    )

    train_check = pd.read_csv(TRAIN_PATH)
    validation_check = pd.read_csv(
        VALIDATION_PATH
    )
    test_check = pd.read_csv(TEST_PATH)

    if (
        len(train_check)
        + len(validation_check)
        + len(test_check)
        != len(data)
    ):
        fail(
            "Saved split row counts do not "
            "match original dataset."
        )

    print(
        "Saved TRAIN rows:",
        len(train_check)
    )

    print(
        "Saved VALIDATION rows:",
        len(validation_check)
    )

    print(
        "Saved TEST rows:",
        len(test_check)
    )

    print(
        "PASS: Saved datasets verified."
    )

    print("=" * 90)
    print("STEP 6.8 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()