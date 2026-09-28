from pathlib import Path
import pandas as pd
import numpy as np


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


FILES = [
    "hai-test1_aligned.csv",
    "hai-test2_aligned.csv",
]


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate_dataset(file_path):

    print("\n" + "=" * 70)
    print(f"VALIDATING: {file_path.name}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    # --------------------------------------------------------
    # 1. Shape
    # --------------------------------------------------------

    print("\n[1] Dataset Shape")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # 2. Column information
    # --------------------------------------------------------

    print("\n[2] Column Types")

    print(f"Numeric columns    : {df.select_dtypes(include=np.number).shape[1]}")
    print(f"Non-numeric columns: {df.select_dtypes(exclude=np.number).shape[1]}")

    # --------------------------------------------------------
    # 3. Timestamp validation
    # --------------------------------------------------------

    print("\n[3] Timestamp Validation")

    if "timestamp" not in df.columns:
        print("FAIL: timestamp column missing")
        return False

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_timestamps = timestamps.isna().sum()

    print(f"Invalid timestamps : {invalid_timestamps}")

    if invalid_timestamps > 0:
        print("FAIL: Invalid timestamps found")
        return False

    # --------------------------------------------------------
    # 4. Timestamp ordering
    # --------------------------------------------------------

    is_sorted = timestamps.is_monotonic_increasing

    print(f"Timestamp sorted   : {is_sorted}")

    if not is_sorted:
        print("FAIL: Timestamps are not sorted")
        return False

    # --------------------------------------------------------
    # 5. Duplicate timestamps
    # --------------------------------------------------------

    duplicate_timestamps = timestamps.duplicated().sum()

    print(f"Duplicate timestamps: {duplicate_timestamps}")

    if duplicate_timestamps > 0:
        print("FAIL: Duplicate timestamps found")
        return False

    # --------------------------------------------------------
    # 6. Sampling interval
    # --------------------------------------------------------

    intervals = timestamps.diff().dropna().dt.total_seconds()

    print("\n[4] Sampling Interval")

    print(f"Minimum interval : {intervals.min()} sec")
    print(f"Maximum interval : {intervals.max()} sec")
    print(f"Unique intervals : {intervals.nunique()}")

    interval_counts = intervals.value_counts().sort_index()

    print("\nInterval distribution:")
    print(interval_counts.to_string())

    # --------------------------------------------------------
    # 7. Missing values
    # --------------------------------------------------------

    print("\n[5] Missing Values")

    missing = df.isna().sum()

    total_missing = missing.sum()

    print(f"Total missing values: {total_missing}")

    if total_missing > 0:
        print("\nColumns containing missing values:")
        print(missing[missing > 0])

        print("FAIL: Missing values found")
        return False

    # --------------------------------------------------------
    # 8. Duplicate rows
    # --------------------------------------------------------

    print("\n[6] Duplicate Rows")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_rows}")

    if duplicate_rows > 0:
        print("FAIL: Duplicate rows found")
        return False

    # --------------------------------------------------------
    # 9. Infinite values
    # --------------------------------------------------------

    print("\n[7] Infinite Values")

    numeric_df = df.select_dtypes(include=np.number)

    infinite_values = np.isinf(numeric_df).sum().sum()

    print(f"Infinite values: {infinite_values}")

    if infinite_values > 0:
        print("FAIL: Infinite values found")
        return False

    # --------------------------------------------------------
    # 10. Label validation
    # --------------------------------------------------------

    print("\n[8] Label Validation")

    if "label" not in df.columns:
        print("FAIL: label column missing")
        return False

    unique_labels = sorted(df["label"].unique())

    print(f"Unique labels: {unique_labels}")

    invalid_labels = set(unique_labels) - {0, 1}

    if invalid_labels:
        print(f"FAIL: Invalid labels found: {invalid_labels}")
        return False

    print("\nLabel distribution:")

    label_counts = df["label"].value_counts().sort_index()

    for label, count in label_counts.items():
        percentage = count / len(df) * 100

        print(
            f"Label {label}: "
            f"{count:,} rows "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 11. Numeric feature validation
    # --------------------------------------------------------

    print("\n[9] Numeric Feature Validation")

    feature_columns = [
        column
        for column in df.columns
        if column not in ["timestamp", "label"]
    ]

    non_numeric_features = df[feature_columns].select_dtypes(
        exclude=np.number
    ).columns.tolist()

    print(f"SCADA feature columns: {len(feature_columns)}")
    print(f"Non-numeric features : {len(non_numeric_features)}")

    if non_numeric_features:
        print("Non-numeric feature columns:")
        print(non_numeric_features)

        print("FAIL: Non-numeric SCADA features found")
        return False

    # --------------------------------------------------------
    # 12. Feature consistency
    # --------------------------------------------------------

    print("\n[10] Feature Structure")

    print("First 10 SCADA features:")

    for column in feature_columns[:10]:
        print(f"  - {column}")

    print(f"\nTotal SCADA features: {len(feature_columns)}")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"VALIDATION RESULT: {file_path.name}")

    print("PASS")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("HAI 23.05 PROCESSED DATA VALIDATION")
    print("=" * 70)

    results = []

    for filename in FILES:

        file_path = PROCESSED_DIR / filename

        if not file_path.exists():

            print("\n" + "=" * 70)
            print(f"FILE NOT FOUND: {file_path}")
            print("=" * 70)

            results.append(False)
            continue

        result = validate_dataset(file_path)

        results.append(result)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    for filename, result in zip(FILES, results):

        status = "PASS" if result else "FAIL"

        print(f"{filename}: {status}")

    if all(results):

        print("\nALL PROCESSED HAI 23.05 DATASETS PASSED VALIDATION.")

    else:

        print("\nONE OR MORE DATASETS FAILED VALIDATION.")

    print("=" * 70)


if __name__ == "__main__":
    main()