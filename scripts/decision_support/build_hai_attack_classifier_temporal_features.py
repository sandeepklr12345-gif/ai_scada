from pathlib import Path

import numpy as np
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test2_attack_classifier_temporal_dataset.csv"
)


METADATA_COLUMNS = {
    "timestamp",
    "attack_label",
    "scenario_id",
    "attack_codes",
}


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 100)
    print("HAI 23.05 ATTACK CLASSIFIER TEMPORAL FEATURE CONSTRUCTION")
    print("=" * 100)

    # -----------------------------------------------------
    # 1. Load validated classifier dataset
    # -----------------------------------------------------

    print("\n[1] Loading validated classifier dataset")

    data = pd.read_csv(INPUT_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print(
        "Input shape:",
        data.shape
    )

    # -----------------------------------------------------
    # 2. Validate ordering
    # -----------------------------------------------------

    print("\n[2] Validating timestamp structure")

    if not data["timestamp"].is_monotonic_increasing:
        fail(
            "Input timestamps are not monotonically increasing."
        )

    if data["timestamp"].duplicated().any():
        fail(
            "Duplicate timestamps detected."
        )

    intervals = (
        data["timestamp"]
        .diff()
        .dropna()
    )

    unique_intervals = intervals.value_counts()

    print(
        "Most common sampling interval:"
    )

    print(
        unique_intervals.head(5)
    )

    if not (
        unique_intervals.index[0]
        == pd.Timedelta(seconds=1)
    ):
        fail(
            "Expected 1-second SCADA sampling."
        )

    print(
        "PASS: Timestamp ordering and 1-second sampling validated."
    )

    # -----------------------------------------------------
    # 3. Identify original features
    # -----------------------------------------------------

    print("\n[3] Identifying original SCADA features")

    target_columns = [
        column
        for column in data.columns
        if column.startswith("AP")
        or column.startswith("AE")
    ]

    feature_columns = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
        and column not in target_columns
    ]

    if len(feature_columns) != 68:
        fail(
            f"Expected 68 original features, "
            f"found {len(feature_columns)}."
        )

    print(
        "Original features:",
        len(feature_columns)
    )

    print(
        "Target columns:",
        len(target_columns)
    )

    # -----------------------------------------------------
    # 4. Preserve metadata
    # -----------------------------------------------------

    print("\n[4] Preserving metadata")

    result = data[
        [
            "timestamp",
            "attack_label",
            "scenario_id",
            "attack_codes",
        ]
        + target_columns
    ].copy()

    # -----------------------------------------------------
    # 5. Copy original features
    # -----------------------------------------------------

    print("\n[5] Copying original SCADA features")

    for column in feature_columns:
        result[column] = data[column].astype(float)

    # -----------------------------------------------------
    # 6. Create first-order differences
    #
    # Current value - previous 1-second value
    #
    # This uses ONLY historical/current information.
    # -----------------------------------------------------

    print(
        "\n[6] Creating first-order temporal differences"
    )

    diff_columns = []

    for column in feature_columns:

        new_column = (
            f"{column}__diff_1s"
        )

        result[new_column] = (
            data[column]
            .astype(float)
            .diff()
        )

        diff_columns.append(
            new_column
        )

    print(
        "Difference features:",
        len(diff_columns)
    )

    # -----------------------------------------------------
    # 7. Create short-term rolling mean
    #
    # Five observations = five seconds because the HAI
    # SCADA stream is sampled every second.
    #
    # min_periods=5 means incomplete history is not used.
    # -----------------------------------------------------

    print(
        "\n[7] Creating 5-second rolling means"
    )

    rolling_mean_columns = []

    for column in feature_columns:

        new_column = (
            f"{column}__rolling_mean_5s"
        )

        result[new_column] = (
            data[column]
            .astype(float)
            .rolling(
                window=5,
                min_periods=5,
            )
            .mean()
        )

        rolling_mean_columns.append(
            new_column
        )

    print(
        "Rolling-mean features:",
        len(rolling_mean_columns)
    )

    # -----------------------------------------------------
    # 8. Create short-term rolling standard deviation
    # -----------------------------------------------------

    print(
        "\n[8] Creating 5-second rolling standard deviations"
    )

    rolling_std_columns = []

    for column in feature_columns:

        new_column = (
            f"{column}__rolling_std_5s"
        )

        result[new_column] = (
            data[column]
            .astype(float)
            .rolling(
                window=5,
                min_periods=5,
            )
            .std()
        )

        rolling_std_columns.append(
            new_column
        )

    print(
        "Rolling-std features:",
        len(rolling_std_columns)
    )

    # -----------------------------------------------------
    # 9. Create deviation from rolling mean
    #
    # Current value - 5-second rolling mean.
    # -----------------------------------------------------

    print(
        "\n[9] Creating deviation-from-mean features"
    )

    deviation_columns = []

    for column in feature_columns:

        new_column = (
            f"{column}__deviation_5s"
        )

        result[new_column] = (
            result[column]
            - result[
                f"{column}__rolling_mean_5s"
            ]
        )

        deviation_columns.append(
            new_column
        )

    print(
        "Deviation features:",
        len(deviation_columns)
    )

    # -----------------------------------------------------
    # 10. Create second-order differences
    #
    # Change in the first-order difference.
    # -----------------------------------------------------

    print(
        "\n[10] Creating second-order temporal differences"
    )

    diff2_columns = []

    for column in feature_columns:

        new_column = (
            f"{column}__diff2_1s"
        )

        result[new_column] = (
            result[
                f"{column}__diff_1s"
            ].diff()
        )

        diff2_columns.append(
            new_column
        )

    print(
        "Second-difference features:",
        len(diff2_columns)
    )

    # -----------------------------------------------------
    # 11. Verify no feature uses labels
    # -----------------------------------------------------

    print(
        "\n[11] Feature-target leakage validation"
    )

    generated_columns = (
        diff_columns
        + rolling_mean_columns
        + rolling_std_columns
        + deviation_columns
        + diff2_columns
    )

    forbidden_tokens = [
        "attack",
        "scenario",
        "label",
        "code",
    ]

    suspicious_columns = []

    for column in generated_columns:

        lowered = column.lower()

        if any(
            token in lowered
            for token in forbidden_tokens
        ):
            suspicious_columns.append(
                column
            )

    if suspicious_columns:
        fail(
            "Suspicious target-related feature names found: "
            f"{suspicious_columns}"
        )

    print(
        "PASS: Generated feature names contain no target metadata."
    )

    # -----------------------------------------------------
    # 12. Verify feature count
    # -----------------------------------------------------

    print(
        "\n[12] Feature-count validation"
    )

    generated_feature_count = (
        len(feature_columns)
        + len(generated_columns)
    )

    expected_feature_count = (
        68
        + 68
        + 68
        + 68
        + 68
        + 68
    )

    print(
        "Original features:",
        len(feature_columns)
    )

    print(
        "Generated temporal features:",
        len(generated_columns)
    )

    print(
        "Expected total features:",
        expected_feature_count
    )

    print(
        "Actual total features:",
        generated_feature_count
    )

    if (
        generated_feature_count
        != expected_feature_count
    ):
        fail(
            "Unexpected feature count."
        )

    # -----------------------------------------------------
    # 13. Warm-up history validation
    # -----------------------------------------------------

    print(
        "\n[13] Temporal warm-up validation"
    )

    feature_only_columns = (
        feature_columns
        + generated_columns
    )

    missing_counts = (
        result[
            feature_only_columns
        ]
        .isna()
        .sum()
    )

    nonzero_missing = (
        missing_counts[
            missing_counts > 0
        ]
    )

    print(
        "Features with missing values:",
        len(nonzero_missing)
    )

    if len(nonzero_missing) > 0:

        print(
            nonzero_missing.head(20)
        )

    # -----------------------------------------------------
    # 14. Remove incomplete temporal history
    #
    # Five-second features require five observations.
    # We remove only the initial warm-up rows.
    # -----------------------------------------------------

    print(
        "\n[14] Removing temporal warm-up rows"
    )

    before_rows = len(result)

    result = result.dropna(
        subset=feature_only_columns
    ).copy()

    after_rows = len(result)

    removed_rows = (
        before_rows
        - after_rows
    )

    print(
        "Rows before:",
        before_rows
    )

    print(
        "Rows after:",
        after_rows
    )

    print(
        "Warm-up rows removed:",
        removed_rows
    )

    if removed_rows != 4:
        fail(
            f"Expected 5 warm-up rows removed, "
            f"found {removed_rows}."
        )

    # -----------------------------------------------------
    # 15. Validate finite numeric values
    # -----------------------------------------------------

    print(
        "\n[15] Numeric validity validation"
    )

    numeric_values = result[
        feature_only_columns
    ].to_numpy()

    if not np.isfinite(
        numeric_values
    ).all():

        fail(
            "NaN or infinite values remain."
        )

    print(
        "PASS: All temporal features are finite."
    )

    # -----------------------------------------------------
    # 16. Validate timestamp continuity after warm-up
    # -----------------------------------------------------

    print(
        "\n[16] Post-warm-up timestamp validation"
    )

    post_intervals = (
        result["timestamp"]
        .diff()
        .dropna()
    )

    invalid_intervals = (
        post_intervals
        != pd.Timedelta(seconds=1)
    ).sum()

    print(
        "Invalid timestamp intervals:",
        invalid_intervals
    )

    if invalid_intervals != 0:
        fail(
            "Timestamp continuity broken."
        )

    print(
        "PASS: Post-warm-up timestamps remain continuous."
    )

    # -----------------------------------------------------
    # 17. Validate metadata preservation
    # -----------------------------------------------------

    print(
        "\n[17] Metadata preservation"
    )

    if not (
        result["attack_label"].isin([0, 1]).all()
    ):
        fail(
            "Invalid attack labels."
        )

    if result[
        "scenario_id"
    ].isna().any():

        fail(
            "Scenario metadata unexpectedly missing."
        )

    for column in target_columns:

        if column not in data.columns:
            fail(
                f"Target column disappeared: {column}"
            )

    print(
        "PASS: Metadata and target vocabulary preserved."
    )

    # -----------------------------------------------------
    # 18. Verify temporal features use only past/current
    # -----------------------------------------------------

    print(
        "\n[18] Temporal causality validation"
    )

    # Verify that first-order differences equal
    # current - previous.
    sample_column = feature_columns[0]

    expected_diff = (
        data[sample_column]
        .astype(float)
        .diff()
        .iloc[4:]
        .to_numpy()
    )

    actual_diff = (
        result[
            f"{sample_column}__diff_1s"
        ]
        .to_numpy()
    )

    if not np.allclose(
        expected_diff,
        actual_diff,
        rtol=1e-10,
        atol=1e-10,
    ):
        fail(
            "First-order difference causality check failed."
        )

    print(
        "PASS: Difference features use current/previous observations only."
    )

    # -----------------------------------------------------
    # 19. Check attack/normal counts
    # -----------------------------------------------------

    print(
        "\n[19] Attack/normal distribution after warm-up"
    )

    normal_rows = int(
        (result["attack_label"] == 0).sum()
    )

    attack_rows = int(
        (result["attack_label"] == 1).sum()
    )

    print(
        "Normal rows:",
        normal_rows
    )

    print(
        "Attack rows:",
        attack_rows
    )

    print(
        "Total:",
        len(result)
    )

    if (
        normal_rows
        + attack_rows
        != len(result)
    ):
        fail(
            "Attack/normal counts do not sum to total rows."
        )

    # -----------------------------------------------------
    # 20. Save dataset
    # -----------------------------------------------------

    print(
        "\n[20] Saving temporal classifier dataset"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "Output:",
        OUTPUT_PATH
    )

    print(
        "Output shape:",
        result.shape
    )

    # -----------------------------------------------------
    # 21. Final validation
    # -----------------------------------------------------

    print(
        "\n[21] Final validation"
    )

    saved = pd.read_csv(
        OUTPUT_PATH
    )

    if len(saved) != len(result):
        fail(
            "Saved row count differs from in-memory result."
        )

    if saved.shape[1] != result.shape[1]:
        fail(
            "Saved column count differs from in-memory result."
        )

    print(
        "Saved rows:",
        len(saved)
    )

    print(
        "Saved columns:",
        len(saved.columns)
    )

    print(
        "PASS: Saved dataset verified."
    )

    print("\n" + "=" * 100)
    print("STEP 6.14 TEMPORAL FEATURE CONSTRUCTION COMPLETE")
    print("=" * 100)

    print(
        """
No classifier was trained.

No Candidate C artifact was modified.

No raw HAI data was modified.

The new feature representation contains:
    68 original SCADA features
    68 first-order difference features
    68 rolling-mean features
    68 rolling-standard-deviation features
    68 deviation-from-mean features
    68 second-order difference features

Total:
    408 classifier features

The first five observations are removed only because
the temporal features require five observations of history.
"""
    )


if __name__ == "__main__":
    main()