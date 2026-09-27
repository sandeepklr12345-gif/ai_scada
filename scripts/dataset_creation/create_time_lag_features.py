import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/600mw/600mw_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/features/600mw"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "600mw_time_lag_features.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("600 MW TIME AND LAG FEATURE CREATION")
print("=" * 80)


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded.")
print("Original shape:", df.shape)


# --------------------------------------------------
# 2. Convert timestamp
# --------------------------------------------------

df["Time"] = pd.to_datetime(
    df["Time"]
)


# --------------------------------------------------
# 3. Create time features
# --------------------------------------------------

df["hour"] = df["Time"].dt.hour

df["minute"] = df["Time"].dt.minute

df["day_of_week"] = df["Time"].dt.dayofweek

df["day_of_month"] = df["Time"].dt.day


# --------------------------------------------------
# 4. Define variables for lag creation
# --------------------------------------------------

lag_features = [
    "Power output\n（MW）",
    "Feed water flow rate\n（t/h）",
    "Plant auxiliary power\n（MW）",
    "Net power output\n（MW）",
    "Feedwater pressure\n（MPa)",
    "Feedwater temperature\n(℃)",
    "Fresh steam pressure\n(MPa)",
    "Drum pressure\n(MPa)",
    "A-side SH steam temperature\n(℃)",
    "B-side SH steam temperature\n(℃)",
    "Reheat steam flow rate\n(t/h)",
    "A-side flue gas oxygen content\n（%）",
    "B-side flue gas oxygen content\n（%）",
    "Boiler efficiency\n(%)",
    "unit efficiency\n（%）"
]


# --------------------------------------------------
# 5. Verify feature names
# --------------------------------------------------

missing_features = [
    feature
    for feature in lag_features
    if feature not in df.columns
]

if missing_features:

    print("\nFeatures not found:")

    for feature in missing_features:
        print("-", feature)

    raise KeyError(
        "One or more lag features were not found."
    )


# --------------------------------------------------
# 6. Define lag intervals
# --------------------------------------------------

lag_steps = [
    1,    # 2 minutes
    2,    # 4 minutes
    5,    # 10 minutes
    10    # 20 minutes
]


# --------------------------------------------------
# 7. Create lag features
# --------------------------------------------------

for feature in lag_features:

    for lag in lag_steps:

        new_column = (
            f"{feature}_lag_{lag}"
        )

        df[new_column] = (
            df[feature].shift(lag)
        )


# --------------------------------------------------
# 8. Show resulting shape
# --------------------------------------------------

print("\nFeature creation complete.")

print(
    "Shape before lag features:",
    (len(df), 78)
)

print(
    "Shape after lag features:",
    df.shape
)


# --------------------------------------------------
# 9. Count missing values created by lagging
# --------------------------------------------------

missing_after_lag = df.isnull().sum().sum()

print(
    "\nTotal missing values after lag creation:",
    missing_after_lag
)


# --------------------------------------------------
# 10. Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("TIME AND LAG FEATURE CREATION COMPLETE")
print("=" * 80)

print("Output:", OUTPUT_FILE)