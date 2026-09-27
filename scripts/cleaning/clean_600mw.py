from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data/raw/600mw/600 MW unit one-week operating data.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/600mw"
OUTPUT_FILE = OUTPUT_DIR / "600mw_clean.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("600 MW DATA CLEANING")
print("=" * 80)

df = pd.read_excel(RAW_FILE)

print(f"\nOriginal shape: {df.shape}")

original_rows = len(df)

df["Time"] = df["Time"].astype(str).str.strip()

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%m/%d %H:%M:%S"
)

print("\nTime conversion complete.")
print("Time dtype:", df["Time"].dtype)

missing_count = df.isnull().sum().sum()
print("\nTotal missing values:", missing_count)

duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)

is_sorted = df["Time"].is_monotonic_increasing
print("Timestamp sorted:", is_sorted)

time_diff = df["Time"].diff().dropna()

print("\nTimestamp intervals:")
print(time_diff.value_counts())

assert len(df) == original_rows, (
    "Row count changed unexpectedly."
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("CLEANING COMPLETE")
print("=" * 80)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Output:", OUTPUT_FILE)