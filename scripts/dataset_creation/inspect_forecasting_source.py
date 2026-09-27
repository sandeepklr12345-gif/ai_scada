import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/600mw/600mw_clean.csv"
)


print("=" * 80)
print("LOAD FORECASTING SOURCE INSPECTION")
print("=" * 80)


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")

print("Rows   :", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 2. Display all columns
# --------------------------------------------------

print("\n" + "-" * 80)
print("COLUMNS")
print("-" * 80)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# --------------------------------------------------
# 3. Data types
# --------------------------------------------------

print("\n" + "-" * 80)
print("DATA TYPES")
print("-" * 80)

print(df.dtypes)


# --------------------------------------------------
# 4. Missing values
# --------------------------------------------------

print("\n" + "-" * 80)
print("MISSING VALUES")
print("-" * 80)

missing = df.isnull().sum()

if missing.sum() == 0:
    print("No missing values.")
else:
    print(missing[missing > 0])


# --------------------------------------------------
# 5. Duplicate rows
# --------------------------------------------------

print("\n" + "-" * 80)
print("DUPLICATES")
print("-" * 80)

print("Duplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 6. Timestamp inspection
# --------------------------------------------------

print("\n" + "-" * 80)
print("TIMESTAMP")
print("-" * 80)

print("First timestamp:", df["Time"].iloc[0])
print("Last timestamp :", df["Time"].iloc[-1])


# --------------------------------------------------
# 7. Numeric columns
# --------------------------------------------------

numeric_columns = df.select_dtypes(
    include="number"
).columns

print("\n" + "-" * 80)
print("NUMERIC COLUMNS")
print("-" * 80)

print("Number of numeric columns:", len(numeric_columns))

for column in numeric_columns:
    print(column)


# --------------------------------------------------
# 8. Basic statistics for power-related columns
# --------------------------------------------------

power_columns = [
    column
    for column in df.columns
    if any(
        keyword in column.lower()
        for keyword in [
            "power",
            "load",
            "output"
        ]
    )
]

print("\n" + "-" * 80)
print("POWER / LOAD RELATED COLUMNS")
print("-" * 80)

for column in power_columns:
    print(column)

if power_columns:
    print("\nStatistics:")
    print(df[power_columns].describe())


print("\n" + "=" * 80)
print("SOURCE INSPECTION COMPLETE")
print("=" * 80)