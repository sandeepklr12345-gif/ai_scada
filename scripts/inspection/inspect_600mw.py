import pandas as pd
from pathlib import Path

RAW_FILE = Path(
    "data/raw/600mw/600 MW unit one-week operating data.xlsx"
)

print("=" * 80)
print("600 MW DATASET INSPECTION")
print("=" * 80)

# Load workbook
df = pd.read_excel(RAW_FILE)

print("\nDataset shape:")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print("\nColumn names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nLast 5 rows:")
print(df.tail().to_string())

print("\nTime column:")
print(df["Time"].head())

print("\nTime data type:")
print(df["Time"].dtype)

print("\nTime range:")
print("Start:", df["Time"].min())
print("End  :", df["Time"].max())

print("\nTime differences:")
time_diff = pd.to_datetime(df["Time"]).diff().dropna()

print(time_diff.value_counts().head(10))

print("\nInspection complete.")