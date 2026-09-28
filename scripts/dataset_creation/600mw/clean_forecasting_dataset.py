import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_PATH = Path(
    "data/features/600mw/600mw_forecasting_with_targets.csv"
)

OUTPUT_PATH = Path(
    "data/features/600mw/600mw_forecasting_clean.csv"
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("8H - FORECASTING DATASET CLEANING")
print("=" * 60)

print(f"\nInput shape: {df.shape}")

# --------------------------------------------------
# Required target columns
# --------------------------------------------------

target_columns = [
    "target_power_2min",
    "target_power_10min",
    "target_power_30min"
]

# --------------------------------------------------
# Identify lag columns
# --------------------------------------------------

lag_columns = [
    column
    for column in df.columns
    if "_lag_" in column
]

print(f"\nNumber of lag columns: {len(lag_columns)}")
print(f"Target columns: {target_columns}")

# --------------------------------------------------
# Check missing values before cleaning
# --------------------------------------------------

missing_counts = df.isna().sum()

missing_columns = missing_counts[
    missing_counts > 0
]

print("\nColumns containing missing values:")

if missing_columns.empty:
    print("None")
else:
    print(missing_columns)

# --------------------------------------------------
# Check missing values in expected forecasting columns
# --------------------------------------------------

forecast_columns = lag_columns + target_columns

unexpected_missing = df[
    [column for column in df.columns if column not in forecast_columns]
].isna().sum()

unexpected_missing = unexpected_missing[
    unexpected_missing > 0
]

print("\nUnexpected missing values outside lag/target columns:")

if unexpected_missing.empty:
    print("None")
else:
    print(unexpected_missing)

# --------------------------------------------------
# Count invalid rows before cleaning
# --------------------------------------------------

invalid_lag_rows = df[lag_columns].isna().any(axis=1)

invalid_target_rows = df[target_columns].isna().any(axis=1)

invalid_rows = invalid_lag_rows | invalid_target_rows

print("\nInvalid rows before cleaning:")
print(f"Lag-related invalid rows: {invalid_lag_rows.sum()}")
print(f"Target-related invalid rows: {invalid_target_rows.sum()}")
print(f"Total invalid rows: {invalid_rows.sum()}")

# --------------------------------------------------
# Remove invalid forecasting rows
# --------------------------------------------------

df_clean = df.loc[~invalid_rows].copy()

# --------------------------------------------------
# Final validation
# --------------------------------------------------

remaining_missing = df_clean[forecast_columns].isna().sum()

remaining_missing = remaining_missing[
    remaining_missing > 0
]

print("\nRemaining missing values in lag/target columns:")

if remaining_missing.empty:
    print("None")
else:
    print(remaining_missing)

print(f"\nOutput shape: {df_clean.shape}")
print(f"Rows removed: {len(df) - len(df_clean)}")

# --------------------------------------------------
# Save cleaned dataset
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df_clean.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nCleaned dataset saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("8H COMPLETED")
print("=" * 60)