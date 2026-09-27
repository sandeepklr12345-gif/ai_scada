import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data/raw/hai/hai-23.05"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/hai/hai-23.05"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("HAI 23.05 DATA CLEANING")
print("=" * 80)


files = sorted(RAW_DIR.glob("*.csv"))

print(f"\nFiles found: {len(files)}")


for file in files:

    print("\n" + "-" * 80)
    print(f"Processing: {file.name}")
    print("-" * 80)

    df = pd.read_csv(file)

    print(f"Original shape: {df.shape}")

    original_rows = len(df)

    # --------------------------------------------------
    # 1. Clean and convert timestamp
    # --------------------------------------------------

    timestamp_column = df.columns[0]

    timestamp_values = (
        df[timestamp_column]
        .astype(str)
        .str.strip()
    )

    converted_timestamp = pd.to_datetime(
        timestamp_values,
        errors="raise"
    )

    df = df.drop(columns=[timestamp_column])

    df.insert(
        0,
        timestamp_column,
        converted_timestamp
    )

    print("Timestamp conversion: OK")

    # --------------------------------------------------
    # 2. Missing values
    # --------------------------------------------------

    missing_count = df.isnull().sum().sum()

    print("Total missing values:", missing_count)

    # --------------------------------------------------
    # 3. Duplicate rows
    # --------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("Duplicate rows:", duplicate_count)

    # --------------------------------------------------
    # 4. Timestamp sorting
    # --------------------------------------------------

    is_sorted = df[timestamp_column].is_monotonic_increasing

    print("Timestamp sorted:", is_sorted)

    # --------------------------------------------------
    # 5. Timestamp intervals
    # --------------------------------------------------

    time_diff = df[timestamp_column].diff().dropna()

    print("\nTimestamp intervals:")
    print(time_diff.value_counts().head(10))

    # --------------------------------------------------
    # 6. Make sure row count did not change
    # --------------------------------------------------

    assert len(df) == original_rows, (
        "Row count changed unexpectedly."
    )

    # --------------------------------------------------
    # 7. Save cleaned file
    # --------------------------------------------------

    output_file = OUTPUT_DIR / file.name

    df.to_csv(
        output_file,
        index=False
    )

    print("\nSaved:", output_file)


print("\n" + "=" * 80)
print("HAI 23.05 CLEANING COMPLETE")
print("=" * 80)