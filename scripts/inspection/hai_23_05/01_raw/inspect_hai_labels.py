import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_DIR = (
    PROJECT_ROOT
    / "data/raw/hai/hai-23.05"
)

FILES = [
    "label-test1.csv",
    "label-test2.csv"
]


print("=" * 80)
print("HAI 23.05 LABEL DATA INSPECTION")
print("=" * 80)


# --------------------------------------------------
# Inspect label files
# --------------------------------------------------

for filename in FILES:

    input_file = INPUT_DIR / filename

    print("\n" + "=" * 80)
    print("FILE:", filename)
    print("=" * 80)

    df = pd.read_csv(input_file)

    # --------------------------------------------------
    # Basic information
    # --------------------------------------------------

    print("\n--- BASIC INFORMATION ---")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # --------------------------------------------------
    # Columns
    # --------------------------------------------------

    print("\n--- COLUMN NAMES ---")

    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    # --------------------------------------------------
    # Data types
    # --------------------------------------------------

    print("\n--- DATA TYPES ---")

    print(df.dtypes)

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    print("\n--- MISSING VALUES ---")

    missing = df.isna().sum()

    missing_columns = missing[missing > 0]

    if len(missing_columns) == 0:
        print("No missing values.")
    else:
        print(missing_columns)

    # --------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------

    print("\n--- DUPLICATES ---")

    print(
        "Duplicate rows:",
        df.duplicated().sum()
    )

    # --------------------------------------------------
    # Timestamp inspection
    # --------------------------------------------------

    print("\n--- TIMESTAMP INSPECTION ---")

    timestamp_columns = [
        column
        for column in df.columns
        if "time" in column.lower()
        or "timestamp" in column.lower()
        or "date" in column.lower()
    ]

    if timestamp_columns:

        print(
            "Possible timestamp columns:",
            timestamp_columns
        )

        timestamp_column = timestamp_columns[0]

        timestamps = pd.to_datetime(
            df[timestamp_column],
            errors="coerce"
        )

        print(
            "Using timestamp column:",
            timestamp_column
        )

        print(
            "Invalid timestamps:",
            timestamps.isna().sum()
        )

        if timestamps.notna().any():

            print(
                "First timestamp:",
                timestamps.min()
            )

            print(
                "Last timestamp:",
                timestamps.max()
            )

            intervals = (
                timestamps
                .sort_values()
                .diff()
                .dropna()
                .dt.total_seconds()
            )

            if len(intervals) > 0:

                print(
                    "Most common sampling intervals:"
                )

                print(
                    intervals
                    .value_counts()
                    .head(10)
                )

    # --------------------------------------------------
    # Label distribution
    # --------------------------------------------------

    print("\n--- LABEL DISTRIBUTION ---")

    for column in df.columns:

        if column not in timestamp_columns:

            print(
                f"\nColumn: {column}"
            )

            print(
                "Unique values:",
                df[column].nunique(dropna=False)
            )

            print(
                df[column]
                .value_counts(dropna=False)
                .head(20)
            )


print("\n" + "=" * 80)
print("HAI 23.05 LABEL INSPECTION COMPLETE")
print("=" * 80)