import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data/features/600mw/600mw_time_lag_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/features/600mw"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "600mw_forecasting_with_targets.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("600 MW FORECASTING TARGET CREATION")
print("=" * 80)


# --------------------------------------------------
# 1. Load time + lag feature dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded.")
print("Input shape:", df.shape)


# --------------------------------------------------
# 2. Convert timestamp
# --------------------------------------------------

df["Time"] = pd.to_datetime(
    df["Time"]
)


# --------------------------------------------------
# 3. Define target column
# --------------------------------------------------

target_column = "Power output\n（MW）"

if target_column not in df.columns:

    raise KeyError(
        f"Target column not found: {target_column}"
    )


# --------------------------------------------------
# 4. Create future targets
# --------------------------------------------------

# Dataset sampling interval = 2 minutes
#
# 1 step  = 2 minutes
# 5 steps = 10 minutes
# 15 steps = 30 minutes

df["target_power_2min"] = (
    df[target_column].shift(-1)
)

df["target_power_10min"] = (
    df[target_column].shift(-5)
)

df["target_power_30min"] = (
    df[target_column].shift(-15)
)


# --------------------------------------------------
# 5. Display target information
# --------------------------------------------------

print("\nFuture targets created:")

print(
    "target_power_2min  → Power at t + 2 minutes"
)

print(
    "target_power_10min → Power at t + 10 minutes"
)

print(
    "target_power_30min → Power at t + 30 minutes"
)


# --------------------------------------------------
# 6. Check target missing values
# --------------------------------------------------

target_columns = [
    "target_power_2min",
    "target_power_10min",
    "target_power_30min"
]

print("\nTarget missing values:")

for column in target_columns:

    print(
        column,
        ":",
        df[column].isnull().sum()
    )


# --------------------------------------------------
# 7. Check final shape
# --------------------------------------------------

print("\nOutput shape:", df.shape)


# --------------------------------------------------
# 8. Show final rows
# --------------------------------------------------

print("\nLast 5 rows of target columns:")

print(
    df[
        [
            "Time",
            target_column,
            "target_power_2min",
            "target_power_10min",
            "target_power_30min"
        ]
    ].tail(5).to_string(index=False)
)


# --------------------------------------------------
# 9. Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("FORECASTING TARGET CREATION COMPLETE")
print("=" * 80)

print("Output:", OUTPUT_FILE)