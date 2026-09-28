import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    / "600mw_leakage_analysis.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("600 MW DATA LEAKAGE ANALYSIS")
print("=" * 80)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 2. Define forecasting target
# --------------------------------------------------

target_column = "Power output\n（MW）"

if target_column not in df.columns:
    raise KeyError(
        f"Target column not found: {target_column}"
    )

print("\nCurrent target:")
print(target_column)


# --------------------------------------------------
# 3. Define variables that require special attention
# --------------------------------------------------

potentially_derived = [
    "Net power output\n（MW）",
    "Plant auxiliary power\n（MW）",
    "Boiler efficiency\n(%)",
    "Standard coal consumption rate for power generation\n(g/kWh)",
    "turbine heat rate\n(kJ/kWh)",
    "unit efficiency\n（%）"
]


# --------------------------------------------------
# 4. Create leakage assessment
# --------------------------------------------------

analysis_rows = []

for column in df.columns:

    if column == "Time":

        role = "timestamp"
        leakage_risk = "none"
        reason = (
            "Used only for temporal ordering and "
            "time-based feature construction."
        )

    elif column == target_column:

        role = "forecast_target"
        leakage_risk = "target"
        reason = (
            "This is the variable being predicted. "
            "Its future value must not be used as an input."
        )

    elif column in potentially_derived:

        role = "candidate_feature"
        leakage_risk = "review"
        reason = (
            "May be derived from other plant measurements "
            "or operational calculations. Requires engineering "
            "and temporal review before final model use."
        )

    else:

        role = "candidate_feature"
        leakage_risk = "low"
        reason = (
            "Observed process variable. Can be considered "
            "as a forecasting input using only information "
            "available at prediction time."
        )

    analysis_rows.append({
        "feature_name": column,
        "role": role,
        "leakage_risk": leakage_risk,
        "reason": reason
    })


leakage_df = pd.DataFrame(
    analysis_rows
)


# --------------------------------------------------
# 5. Save analysis
# --------------------------------------------------

leakage_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 6. Display summary
# --------------------------------------------------

print("\n" + "-" * 80)
print("LEAKAGE ASSESSMENT SUMMARY")
print("-" * 80)

print(
    leakage_df["leakage_risk"]
    .value_counts()
)


print("\n" + "-" * 80)
print("VARIABLES REQUIRING REVIEW")
print("-" * 80)

review_features = leakage_df[
    leakage_df["leakage_risk"] == "review"
]

print(
    review_features[
        ["feature_name", "reason"]
    ].to_string(index=False)
)


print("\n" + "-" * 80)
print("FORECAST TARGET")
print("-" * 80)

print(target_column)


print("\n" + "=" * 80)
print("DATA LEAKAGE ANALYSIS COMPLETE")
print("=" * 80)

print("\nOutput:", OUTPUT_FILE)