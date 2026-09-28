import pandas as pd
from pathlib import Path


# ============================================================
# 8J - FINAL FORECASTING DATASET VALIDATION
# ============================================================

DATA_DIR = Path(
    "data/features/600mw/candidates"
)

EXPECTED_FILES = [
    "feature_set_A_power_history.csv",
    "feature_set_B_process_core.csv",
    "feature_set_C_thermal_process.csv",
    "feature_set_D_full_candidate_pool.csv"
]

EXPECTED_ROWS = 5649

TARGET_COLUMNS = [
    "target_power_2min",
    "target_power_10min",
    "target_power_30min"
]

TIME_COLUMN = "Time"


# ============================================================
# Helper functions
# ============================================================

def check(condition, pass_message, fail_message):
    """
    Print validation result and return True/False.
    """
    if condition:
        print(f"PASS  {pass_message}")
        return True

    print(f"FAIL  {fail_message}")
    return False


def validate_dataset(file_name):

    print("\n" + "=" * 70)
    print(f"VALIDATING: {file_name}")
    print("=" * 70)

    file_path = DATA_DIR / file_name

    # --------------------------------------------------------
    # 1. File existence
    # --------------------------------------------------------

    if not file_path.exists():

        print("FAIL  File does not exist")
        return False

    print(f"File: {file_path}")

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    try:
        df = pd.read_csv(file_path)
    except Exception as error:

        print(f"FAIL  Could not read dataset: {error}")
        return False

    print(f"Shape: {df.shape}")

    passed = True

    # --------------------------------------------------------
    # 3. Row count
    # --------------------------------------------------------

    if not check(
        len(df) == EXPECTED_ROWS,
        f"Row count = {len(df)}",
        f"Expected {EXPECTED_ROWS} rows, found {len(df)}"
    ):
        passed = False

    # --------------------------------------------------------
    # 4. Column uniqueness
    # --------------------------------------------------------

    duplicate_columns = df.columns[
        df.columns.duplicated()
    ].tolist()

    if not check(
        len(duplicate_columns) == 0,
        "Column names are unique",
        f"Duplicate columns found: {duplicate_columns}"
    ):
        passed = False

    # --------------------------------------------------------
    # 5. Required Time column
    # --------------------------------------------------------

    if not check(
        TIME_COLUMN in df.columns,
        "Time column exists",
        "Time column is missing"
    ):
        passed = False

    # --------------------------------------------------------
    # 6. Required target columns
    # --------------------------------------------------------

    missing_targets = [
        target
        for target in TARGET_COLUMNS
        if target not in df.columns
    ]

    if not check(
        len(missing_targets) == 0,
        "All target columns exist",
        f"Missing target columns: {missing_targets}"
    ):
        passed = False

    # --------------------------------------------------------
    # 7. No missing values anywhere
    # --------------------------------------------------------

    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())

    if not check(
        total_missing == 0,
        "No missing values",
        f"{total_missing} missing values found"
    ):
        passed = False

        # Show only affected columns
        affected = missing_counts[
            missing_counts > 0
        ]

        print("\nMissing-value details:")

        for column, count in affected.items():
            print(f"  {column}: {count}")

    # --------------------------------------------------------
    # 8. Duplicate rows
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if not check(
        duplicate_rows == 0,
        "No duplicate rows",
        f"{duplicate_rows} duplicate rows found"
    ):
        passed = False

    # --------------------------------------------------------
    # 9. Timestamp validation
    # --------------------------------------------------------

    if TIME_COLUMN in df.columns:

        parsed_time = pd.to_datetime(
            df[TIME_COLUMN],
            errors="coerce"
        )

        invalid_timestamp_count = int(
            parsed_time.isna().sum()
        )

        if not check(
            invalid_timestamp_count == 0,
            "All timestamps are valid",
            f"{invalid_timestamp_count} invalid timestamps found"
        ):
            passed = False

        # ----------------------------------------------------
        # 10. Timestamp ordering
        # ----------------------------------------------------

        if invalid_timestamp_count == 0:

            time_differences = (
                parsed_time
                .sort_values()
                .diff()
                .dropna()
            )

            non_positive_intervals = (
                time_differences <= pd.Timedelta(0)
            ).sum()

            if not check(
                non_positive_intervals == 0,
                "Timestamps are strictly increasing",
                f"{non_positive_intervals} non-positive intervals found"
            ):
                passed = False

            # ------------------------------------------------
            # 11. Sampling interval
            # ------------------------------------------------

            interval_counts = (
                time_differences
                .value_counts()
            )

            print("\nSampling interval distribution:")

            for interval, count in interval_counts.items():
                print(
                    f"  {interval}: {count} intervals"
                )

            expected_interval = pd.Timedelta(
                minutes=2
            )

            unexpected_intervals = (
                time_differences != expected_interval
            ).sum()

            if not check(
                unexpected_intervals == 0,
                "All sampling intervals are 2 minutes",
                (
                    f"{unexpected_intervals} "
                    "unexpected sampling intervals found"
                )
            ):
                passed = False

    # --------------------------------------------------------
    # 12. Target numeric validation
    # --------------------------------------------------------

    if all(
        target in df.columns
        for target in TARGET_COLUMNS
    ):

        non_numeric_targets = [
            target
            for target in TARGET_COLUMNS
            if not pd.api.types.is_numeric_dtype(
                df[target]
            )
        ]

        if not check(
            len(non_numeric_targets) == 0,
            "All targets are numeric",
            f"Non-numeric targets: {non_numeric_targets}"
        ):
            passed = False

    # --------------------------------------------------------
    # 13. Feature/target separation
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in TARGET_COLUMNS
    ]

    leaked_targets = [
        target
        for target in TARGET_COLUMNS
        if target in feature_columns
    ]

    if not check(
        len(leaked_targets) == 0,
        "Targets are separated from input features",
        f"Target leakage detected: {leaked_targets}"
    ):
        passed = False

    # --------------------------------------------------------
    # 14. Future-target sanity check
    # --------------------------------------------------------
    if all(
        target in df.columns
        for target in TARGET_COLUMNS
    ):
        if "Power output\n（MW）" in df.columns:

            current_power = df[
                "Power output\n（MW）"
            ]

            target_power = df[
                "target_power_2min"
            ]

            expected_target = current_power.shift(-1)

            # Compare only rows where both values exist.
            valid_target_rows = (
                target_power.notna()
                & expected_target.notna()
            )

            actual_values = target_power[
                valid_target_rows
            ].to_numpy()

            expected_values = expected_target[
                valid_target_rows
            ].to_numpy()

            mismatches = (
                ~pd.Series(
                    actual_values
                ).combine(
                    pd.Series(expected_values),
                    lambda actual, expected:
                    abs(actual - expected) <= 1e-9
                )
            ).sum()

            if not check(
                mismatches == 0,
                "2-minute target alignment is correct",
                f"{mismatches} target alignment mismatches found"
            ):
                passed = False

    # --------------------------------------------------------
    # 15. Numeric feature validation
    # --------------------------------------------------------

    non_numeric_features = []

    for column in feature_columns:

        if column == TIME_COLUMN:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            non_numeric_features.append(
                column
            )

    if not check(
        len(non_numeric_features) == 0,
        "All ML input features are numeric",
        (
            "Non-numeric feature columns: "
            f"{non_numeric_features}"
        )
    ):
        passed = False

    # --------------------------------------------------------
    # 16. Infinite-value validation
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    infinite_count = int(
        df[numeric_columns]
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    if not check(
        infinite_count == 0,
        "No infinite numeric values",
        f"{infinite_count} infinite values found"
    ):
        passed = False

    # --------------------------------------------------------
    # 17. Feature count summary
    # --------------------------------------------------------

    input_feature_count = len(
        feature_columns
    )

    print(
        f"\nInput features: {input_feature_count}"
    )

    print(
        f"Targets: {len(TARGET_COLUMNS)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Final dataset result
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    if passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAILED")

    return passed


# ============================================================
# Run validation
# ============================================================

print("=" * 70)
print("8J - FINAL FORECASTING DATASET VALIDATION")
print("=" * 70)

results = {}

for file_name in EXPECTED_FILES:

    results[file_name] = validate_dataset(
        file_name
    )


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("8J FINAL SUMMARY")
print("=" * 70)

all_passed = True

for file_name, result in results.items():

    status = "PASS" if result else "FAILED"

    print(
        f"{file_name}: {status}"
    )

    if not result:
        all_passed = False


print("\n" + "=" * 70)

if all_passed:

    print(
        "ALL FORECASTING DATASETS PASSED FINAL VALIDATION"
    )

    print(
        "\nSTEP 8 DATASET CREATION: COMPLETE"
    )

else:

    print(
        "SOME DATASETS FAILED FINAL VALIDATION"
    )

    print(
        "\nSTEP 8 DATASET CREATION: NOT COMPLETE"
    )

print("=" * 70)