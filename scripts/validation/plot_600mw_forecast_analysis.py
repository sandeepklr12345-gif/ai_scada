from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_runtime_forecast_results_optimized.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("600 MW FORECAST VISUAL ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print(
    f"\nRows loaded : {len(df)}"
)


# ============================================================
# CREATE ERROR / RAMP DATA
# ============================================================

df["power_change_MW"] = (
    df["actual_power"].diff()
)

df["absolute_ramp_MW"] = (
    df["power_change_MW"].abs()
)

forecasts = {
    "2min": "forecast_2min",
    "10min": "forecast_10min",
    "30min": "forecast_30min",
}


for horizon, column in forecasts.items():

    df[f"{horizon}_absolute_error"] = (
        (
            df[column]
            - df["actual_power"]
        )
        .abs()
    )


# ============================================================
# PLOT 1
# ACTUAL VS FORECAST
# ============================================================

print("\nCreating actual vs forecast plot...")

plt.figure(
    figsize=(16, 7)
)

plt.plot(
    df["actual_power"],
    label="Actual Power",
    linewidth=1.5
)

plt.plot(
    df["forecast_2min"],
    label="2-Min Forecast",
    linewidth=1
)

plt.plot(
    df["forecast_10min"],
    label="10-Min Forecast",
    linewidth=1
)

plt.plot(
    df["forecast_30min"],
    label="30-Min Forecast",
    linewidth=1
)

plt.xlabel(
    "Replay Step"
)

plt.ylabel(
    "Power (MW)"
)

plt.title(
    "600 MW Plant: Actual vs Forecasted Power"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "actual_vs_forecast.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2-4
# RAMP VS ERROR
# ============================================================

for horizon in forecasts:

    print(
        f"Creating ramp vs error plot: {horizon}"
    )

    error_column = (
        f"{horizon}_absolute_error"
    )

    valid = df[
        [
            "absolute_ramp_MW",
            error_column
        ]
    ].dropna()

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        valid["absolute_ramp_MW"],
        valid[error_column],
        alpha=0.35,
        s=12
    )

    plt.xlabel(
        "Absolute Power Ramp (MW / 2 min)"
    )

    plt.ylabel(
        "Absolute Forecast Error (MW)"
    )

    plt.title(
        f"{horizon} Forecast: "
        "Power Ramp vs Forecast Error"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"ramp_vs_error_{horizon}.png",
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT 5
# ERROR DISTRIBUTION
# ============================================================

print(
    "Creating forecast error distribution..."
)

plt.figure(
    figsize=(12, 7)
)

for horizon in forecasts:

    plt.hist(
        df[
            f"{horizon}_absolute_error"
        ].dropna(),
        bins=50,
        alpha=0.45,
        label=horizon
    )

plt.xlabel(
    "Absolute Forecast Error (MW)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "600 MW Forecast Error Distribution"
)

plt.legend(
    title="Forecast Horizon"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "forecast_error_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PLOTS CREATED")
print("=" * 70)

for file in sorted(OUTPUT_DIR.glob("*.png")):

    print(
        f"\n{file.name}"
    )

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\n600 MW VISUAL ANALYSIS: PASS"
)

print("=" * 70)