import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data/raw/uci_ccpp/Folds5x2_pp.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/uci_ccpp"
OUTPUT_FILE = OUTPUT_DIR / "uci_ccpp_clean.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("UCI CCPP DATA CLEANING")
print("=" * 80)


# --------------------------------------------------
# 1. Read Excel file
# --------------------------------------------------

df = pd.read_excel(RAW_FILE)

print("\nOriginal shape:", df.shape)

original_rows = len(df)
original_columns = len(df.columns)


# --------------------------------------------------
# 2. Display columns
# --------------------------------------------------

print("\nColumns:")
print(list(df.columns))


# --------------------------------------------------
# 3. Missing values
# --------------------------------------------------

missing_count = df.isnull().sum().sum()

print("\nTotal missing values:", missing_count)


# --------------------------------------------------
# 4. Duplicate rows
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# --------------------------------------------------
# 5. Data types
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 6. Check numeric columns
# --------------------------------------------------

numeric_columns = df.select_dtypes(include="number").columns

print("\nNumeric columns:")
print(list(numeric_columns))

print("\nNumber of numeric columns:", len(numeric_columns))


# --------------------------------------------------
# 7. Basic statistics
# --------------------------------------------------

print("\nBasic statistics:")
print(df.describe())


# --------------------------------------------------
# 8. Make sure row count did not change
# --------------------------------------------------

assert len(df) == original_rows, (
    "Row count changed unexpectedly."
)


# --------------------------------------------------
# 9. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("UCI CCPP CLEANING COMPLETE")
print("=" * 80)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Output:", OUTPUT_FILE)