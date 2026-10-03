from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"


def fail(message):
    raise ValueError(message)


def get_target_columns(data):
    return [
        column
        for column in data.columns
        if column.startswith("AP")
        or column.startswith("AE")
    ]


def active_codes(data, target_columns):
    return {
        column
        for column in target_columns
        if column in data.columns
        and data[column].sum() > 0
    }


def code_counts(data, target_columns):
    return {
        column: int(data[column].sum())
        for column in target_columns
        if column in data.columns
    }


def main():

    print("=" * 95)
    print("HAI 23.05 ATTACK CLASSIFIER LABEL LEARNABILITY AUDIT")
    print("=" * 95)

    # -----------------------------------------------------
    # 1. Load splits
    # -----------------------------------------------------

    print("\n[1] Loading classifier splits")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)

    print("TRAIN:", train.shape)
    print("VALIDATION:", validation.shape)
    print("TEST:", test.shape)

    # -----------------------------------------------------
    # 2. Identify target columns
    # -----------------------------------------------------

    print("\n[2] Identifying attack-code targets")

    target_columns = get_target_columns(train)

    print(
        "Total target columns:",
        len(target_columns)
    )

    if len(target_columns) != 39:
        fail(
            f"Expected 39 attack-code targets, "
            f"found {len(target_columns)}."
        )

    print(
        "PASS: 39 target columns detected."
    )

    # -----------------------------------------------------
    # 3. Verify target columns exist everywhere
    # -----------------------------------------------------

    print(
        "\n[3] Target-column consistency"
    )

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        missing = [
            column
            for column in target_columns
            if column not in frame.columns
        ]

        if missing:
            fail(
                f"{name} is missing target columns: "
                f"{missing}"
            )

        print(
            f"{name}: all 39 target columns present"
        )

    # -----------------------------------------------------
    # 4. Count each label
    # -----------------------------------------------------

    print(
        "\n[4] Attack-code sample counts"
    )

    train_counts = code_counts(
        train,
        target_columns
    )

    validation_counts = code_counts(
        validation,
        target_columns
    )

    test_counts = code_counts(
        test,
        target_columns
    )

    print(
        "\nCODE       TRAIN    VALIDATION    TEST"
    )
    print(
        "-" * 45
    )

    for code in target_columns:

        print(
            f"{code:<8}"
            f"{train_counts[code]:>8}"
            f"{validation_counts[code]:>13}"
            f"{test_counts[code]:>9}"
        )

    # -----------------------------------------------------
    # 5. Determine learnability
    # -----------------------------------------------------

    print(
        "\n[5] Training-label learnability"
    )

    train_codes = active_codes(
        train,
        target_columns
    )

    validation_codes = active_codes(
        validation,
        target_columns
    )

    test_codes = active_codes(
        test,
        target_columns
    )

    unseen_in_training = (
        validation_codes
        | test_codes
    ) - train_codes

    print(
        "Codes present in TRAIN:",
        len(train_codes)
    )

    print(
        "Codes absent from TRAIN but "
        "present in evaluation:",
        len(unseen_in_training)
    )

    print(
        "Unseen evaluation codes:"
    )

    print(
        sorted(unseen_in_training)
    )

    # -----------------------------------------------------
    # 6. Codes available in all three splits
    # -----------------------------------------------------

    print(
        "\n[6] Cross-split label coverage"
    )

    common_codes = (
        train_codes
        & validation_codes
        & test_codes
    )

    print(
        "Codes present in TRAIN + VALIDATION + TEST:",
        len(common_codes)
    )

    print(
        sorted(common_codes)
    )

    # -----------------------------------------------------
    # 7. Codes only in training
    # -----------------------------------------------------

    train_only = (
        train_codes
        - validation_codes
        - test_codes
    )

    print(
        "\nCodes appearing only in TRAIN:"
    )

    print(
        sorted(train_only)
    )

    # -----------------------------------------------------
    # 8. Codes only in validation
    # -----------------------------------------------------

    validation_only = (
        validation_codes
        - train_codes
        - test_codes
    )

    print(
        "\nCodes appearing only in VALIDATION:"
    )

    print(
        sorted(validation_only)
    )

    # -----------------------------------------------------
    # 9. Codes only in test
    # -----------------------------------------------------

    test_only = (
        test_codes
        - train_codes
        - validation_codes
    )

    print(
        "\nCodes appearing only in TEST:"
    )

    print(
        sorted(test_only)
    )

    # -----------------------------------------------------
    # 10. Multi-label structure
    # -----------------------------------------------------

    print(
        "\n[10] Multi-label structure"
    )

    def label_cardinality(frame):

        return frame[target_columns].sum(
            axis=1
        )

    train_cardinality = label_cardinality(train)
    validation_cardinality = label_cardinality(
        validation
    )
    test_cardinality = label_cardinality(test)

    for name, cardinality in [
        ("TRAIN", train_cardinality),
        ("VALIDATION", validation_cardinality),
        ("TEST", test_cardinality),
    ]:

        print(
            f"\n{name}"
        )

        print(
            "0 labels:",
            int((cardinality == 0).sum())
        )

        print(
            "1 label:",
            int((cardinality == 1).sum())
        )

        print(
            "2+ labels:",
            int((cardinality >= 2).sum())
        )

    # -----------------------------------------------------
    # 11. Combination scenario distribution
    # -----------------------------------------------------

    print(
        "\n[11] Combination attack coverage"
    )

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        combinations = (
            frame[
                frame["attack_codes"]
                .fillna("")
                .str.contains(
                    "+",
                    regex=False
                )
            ]
        )

        print(
            f"{name}: "
            f"{len(combinations)} combination rows"
        )

        if not combinations.empty:

            print(
                sorted(
                    combinations[
                        "scenario_id"
                    ]
                    .dropna()
                    .unique()
                )
            )

    # -----------------------------------------------------
    # 12. Feature consistency
    # -----------------------------------------------------

    print(
        "\n[12] Feature consistency"
    )

    metadata_columns = {
        "timestamp",
        "attack_label",
        "scenario_id",
        "attack_codes",
    }

    feature_columns = [
        column
        for column in train.columns
        if column not in metadata_columns
        and column not in target_columns
    ]

    print(
        "Feature count:",
        len(feature_columns)
    )

    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        missing_features = [
            column
            for column in feature_columns
            if column not in frame.columns
        ]

        if missing_features:
            fail(
                f"{name} missing features: "
                f"{missing_features}"
            )

        missing_values = (
            frame[feature_columns]
            .isna()
            .sum()
            .sum()
        )

        print(
            f"{name}: "
            f"features={len(feature_columns)}, "
            f"missing_values={missing_values}"
        )

        if missing_values:
            fail(
                f"{name} contains missing feature values."
            )

    print(
        "PASS: Feature structure consistent."
    )

    # -----------------------------------------------------
    # 13. Final interpretation
    # -----------------------------------------------------

    print(
        "\n[13] Learnability conclusion"
    )

    print(
        """
This audit does NOT train a classifier.

It determines which attack-code labels are actually
observable in the chronological training split.

A label with zero positive training examples cannot
be learned by a supervised classifier from this
training split.

Therefore, labels absent from TRAIN must not be
reported as normally trainable attack classes.

The chronological split remains valid for event-level
evaluation, but label coverage must be considered
when interpreting classifier performance.
"""
    )

    # -----------------------------------------------------
    # 14. Final status
    # -----------------------------------------------------

    print("=" * 95)
    print("LABEL LEARNABILITY AUDIT COMPLETE")
    print("=" * 95)


if __name__ == "__main__":
    main()