import pandas as pd
from pathlib import Path

RAW = Path("data/raw")


def inspect_csv(path):
    print("\n" + "=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    try:
        df = pd.read_csv(path)

        print(f"Rows       : {len(df):,}")
        print(f"Columns    : {len(df.columns)}")
        print("\nColumns:")
        print(list(df.columns))

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        missing = df.isnull().sum()
        print(missing[missing > 0])

        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

    except Exception as e:
        print(f"ERROR: {e}")


def inspect_excel(path):
    print("\n" + "=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    try:
        excel = pd.ExcelFile(path)

        print("Sheets:")
        print(excel.sheet_names)

        for sheet in excel.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)

            print(f"\n--- Sheet: {sheet} ---")
            print(f"Rows    : {len(df):,}")
            print(f"Columns : {len(df.columns)}")
            print("Columns:")
            print(list(df.columns))

            print("\nFirst 3 rows:")
            print(df.head(3).to_string())

    except Exception as e:
        print(f"ERROR: {e}")

# Find all CSV files
csv_files = list(RAW.rglob("*.csv"))

print(f"Found {len(csv_files)} CSV files.")

for file in csv_files:
    inspect_csv(file)

# Find all Excel files
excel_files = list(RAW.rglob("*.xlsx"))

print(f"\nFound {len(excel_files)} Excel files.")

for file in excel_files:
    inspect_excel(file)