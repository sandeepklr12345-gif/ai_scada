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
    "hai-test1.csv",
    "hai-test2.csv",
    "hai-train1.csv",
    "hai-train2.csv",
    "hai-train3.csv",
    "hai-train4.csv"
]


print("=" * 80)
print("HAI 23.05 RAW DATA INSPECTION")
print("=" * 80)


# --------------------------------------------------
# Inspect each file
# --------------------------------------------------

for filename in FILES:

    input_file = INPUT_DIR / filename

    print("\n" + "=" * 80)
    print("FILE:", filename)
    print("=" * 80)

    if not input_file.exists():
        print("ERROR: File not found")
        print("Path:", input_file)
        continue

    df = pd.read_csv(input_file)

    # --------------------------------------------------
    # Basic information
    # --------------------------------------------------

    print("\n--- BASIC INFORMATION ---")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # --------------------------------------------------
    # Column names
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

    duplicate_count = df.duplicated().sum()

    print("Duplicate rows:", duplicate_count)

    # --------------------------------------------------
    # Numeric / non-numeric columns
    # --------------------------------------------------

    print("\n--- COLUMN TYPES ---")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    non_numeric_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print("Numeric columns:", len(numeric_columns))
    print("Non-numeric columns:", len(non_numeric_columns))

    print("\nNon-numeric columns:")

    for column in non_numeric_columns:
        print("-", column)

    # --------------------------------------------------
    # Timestamp inspection
    # --------------------------------------------------

    print("\n--- TIMESTAMP INSPECTION ---")

    timestamp_candidates = [
        column
        for column in df.columns
        if "time" in column.lower()
        or "timestamp" in column.lower()
        or "date" in column.lower()
    ]

    if timestamp_candidates:

        print(
            "Possible timestamp columns:",
            timestamp_candidates
        )

        timestamp_column = timestamp_candidates[0]

        timestamps = pd.to_datetime(
            df[timestamp_column],
            errors="coerce"
        )

        print("Using timestamp column:", timestamp_column)

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

            # Sampling interval
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

    else:

        print("No obvious timestamp column detected.")

    # --------------------------------------------------
    # Infinite values
    # --------------------------------------------------

    print("\n--- INFINITE VALUES ---")

    numeric_df = df.select_dtypes(
        include="number"
    )

    if len(numeric_df.columns) > 0:

        infinite_count = (
            numeric_df
            .isin([float("inf"), float("-inf")])
            .sum()
            .sum()
        )

        print(
            "Total infinite values:",
            infinite_count
        )

    else:

        print("No numeric columns.")

    # --------------------------------------------------
    # Possible labels / anomaly columns
    # --------------------------------------------------

    print("\n--- POSSIBLE LABEL / ANOMALY COLUMNS ---")

    possible_label_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in [
                "label",
                "attack",
                "anomaly",
                "class",
                "event",
                "fault"
            ]
        )
    ]

    if possible_label_columns:

        for column in possible_label_columns:

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

    else:

        print(
            "No obvious label/anomaly column detected."
        )


print("\n" + "=" * 80)
print("HAI 23.05 RAW INSPECTION COMPLETE")
print("=" * 80)