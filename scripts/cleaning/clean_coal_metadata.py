import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data/raw/coal_plants.csv"

OUTPUT_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_FILE = OUTPUT_DIR / "coal_plant_metadata.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("COAL PLANT METADATA CLEANING")
print("=" * 80)


# --------------------------------------------------
# 1. Read CSV
# --------------------------------------------------

df = pd.read_csv(RAW_FILE)

print("\nOriginal shape:", df.shape)

original_rows = len(df)


# --------------------------------------------------
# 2. Display columns
# --------------------------------------------------

print("\nColumns:")
print(list(df.columns))


# --------------------------------------------------
# 3. Missing values
# --------------------------------------------------

print("\nMissing values:")

missing = df.isnull().sum()

if missing.sum() == 0:
    print("No missing values.")
else:
    print(missing[missing > 0])


# --------------------------------------------------
# 4. Duplicate rows
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)


# --------------------------------------------------
# 5. Duplicate plant IDs
# --------------------------------------------------

if "id" in df.columns:

    duplicate_ids = df["id"].duplicated().sum()

    print("Duplicate plant IDs:", duplicate_ids)


# --------------------------------------------------
# 6. Data types
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 7. Basic information about important fields
# --------------------------------------------------

for column in [
    "name",
    "type",
    "fuel",
    "region",
    "country",
    "status"
]:

    if column in df.columns:

        print(f"\nUnique values in '{column}':")

        print(
            df[column]
            .dropna()
            .astype(str)
            .unique()[:20]
        )


# --------------------------------------------------
# 8. Geographic fields
# --------------------------------------------------

if "lat" in df.columns and "lon" in df.columns:

    print("\nGeographic fields found:")
    print("Latitude:", df["lat"].dtype)
    print("Longitude:", df["lon"].dtype)


# --------------------------------------------------
# 9. Capacity information
# --------------------------------------------------

if "capacity_power" in df.columns:

    print("\nCapacity statistics:")

    print(
        df["capacity_power"].describe()
    )


# --------------------------------------------------
# 10. Row count verification
# --------------------------------------------------

assert len(df) == original_rows, (
    "Row count changed unexpectedly."
)


# --------------------------------------------------
# 11. Save processed metadata
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("COAL PLANT METADATA CLEANING COMPLETE")
print("=" * 80)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Output:", OUTPUT_FILE)