from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_runtime_forecast_results_optimized.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_ramp_error_analysis.csv"
)


print("=" * 70)
print("600 MW RAMP-RATE vs FORECAST ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print(
    f"\nRows loaded : {len(df)}"
)


# ============================================================
# ACTUAL POWER CHANGE
# ============================================================

df["power_change_MW"] = (
    df["actual_power"]
    .diff()
)

df["absolute_ramp_MW"] = (
    df["power_change_MW"]
    .abs()
)


# ============================================================
# FORECAST ERRORS
# ============================================================

FORECASTS = {
    "2min": "forecast_2min",
    "10min": "forecast_10min",
    "30min": "forecast_30min",
}


for horizon, column in FORECASTS.items():

    df[f"{horizon}_error_MW"] = (
        df[column]
        - df["actual_power"]
    )

    df[f"{horizon}_absolute_error_MW"] = (
        df[f"{horizon}_error_MW"]
        .abs()
    )


# Remove first row because it has no previous value

analysis = df.dropna(
    subset=["absolute_ramp_MW"]
).copy()


# ============================================================
# RAMP CATEGORIES
# ============================================================

analysis["ramp_category"] = pd.cut(
    analysis["absolute_ramp_MW"],
    bins=[
        -np.inf,
        2,
        5,
        10,
        20,
        50,
        np.inf
    ],
    labels=[
        "0-2 MW",
        "2-5 MW",
        "5-10 MW",
        "10-20 MW",
        "20-50 MW",
        ">50 MW"
    ]
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RAMP CATEGORY SUMMARY")
print("=" * 70)


for horizon in FORECASTS:

    error_column = (
        f"{horizon}_absolute_error_MW"
    )

    print(
        f"\n{horizon} FORECAST"
    )

    summary = (
        analysis
        .groupby(
            "ramp_category",
            observed=True
        )[error_column]
        .agg(
            count="count",
            mean_error="mean",
            max_error="max"
        )
    )

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("RAMP vs ABSOLUTE FORECAST ERROR")
print("=" * 70)


for horizon in FORECASTS:

    error_column = (
        f"{horizon}_absolute_error_MW"
    )

    correlation = (
        analysis[
            [
                "absolute_ramp_MW",
                error_column
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    print(
        f"{horizon:>5} : "
        f"{correlation:.6f}"
    )


# ============================================================
# TOP RAMP EVENTS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 POWER RAMP EVENTS")
print("=" * 70)


top_ramps = (
    analysis
    .sort_values(
        "absolute_ramp_MW",
        ascending=False
    )
    .head(20)
)


columns = [
    "timestamp",
    "actual_power",
    "power_change_MW",
    "absolute_ramp_MW",
    "2min_absolute_error_MW",
    "10min_absolute_error_MW",
    "30min_absolute_error_MW",
]


print(
    top_ramps[
        columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

analysis.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("RAMP ANALYSIS SAVED")
print("=" * 70)

print(
    f"\n{OUTPUT_FILE}"
)

print(
    "\n600 MW RAMP ERROR ANALYSIS: PASS"
)

print("=" * 70)