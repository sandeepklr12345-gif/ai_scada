from pathlib import Path
import pandas as pd
import numpy as np


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_forecast_error_analysis.csv"
)


FORECASTS = {
    "2min": "forecast_2min",
    "10min": "forecast_10min",
    "30min": "forecast_30min",
}


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("600 MW FORECAST ERROR ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nRows loaded : {len(df)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required = [
    "timestamp",
    "actual_power",
    *FORECASTS.values()
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

actual = df["actual_power"].astype(float)


# ============================================================
# ANALYZE EACH HORIZON
# ============================================================

all_error_rows = []

for horizon, forecast_column in FORECASTS.items():

    prediction = (
        df[forecast_column]
        .astype(float)
    )

    signed_error = (
        prediction - actual
    )

    absolute_error = (
        np.abs(signed_error)
    )

    percentage_error = np.where(
        actual != 0,
        (
            absolute_error
            / np.abs(actual)
        ) * 100,
        np.nan
    )

    # --------------------------------------------------------
    # Temporary error columns
    # --------------------------------------------------------

    temp = pd.DataFrame({

        "timestamp":
            df["timestamp"],

        "replay_step":
            df["replay_step"],

        "actual_power":
            actual,

        "forecast":
            prediction,

        "signed_error_MW":
            signed_error,

        "absolute_error_MW":
            absolute_error,

        "percentage_error":
            percentage_error,

        "horizon":
            horizon,
    })

    # --------------------------------------------------------
    # Top 20 errors
    # --------------------------------------------------------

    top_errors = (
        temp
        .sort_values(
            "absolute_error_MW",
            ascending=False
        )
        .head(20)
    )

    all_error_rows.append(
        top_errors
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"{horizon} FORECAST"
    )

    print("-" * 70)

    print(
        f"Mean absolute error : "
        f"{absolute_error.mean():.4f} MW"
    )

    print(
        f"Maximum error       : "
        f"{absolute_error.max():.4f} MW"
    )

    print(
        f"Mean percentage err : "
        f"{np.nanmean(percentage_error):.4f}%"
    )

    print(
        "\nTOP 10 ERRORS"
    )

    display_columns = [
        "timestamp",
        "actual_power",
        "forecast",
        "signed_error_MW",
        "percentage_error",
    ]

    print(
        top_errors[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# COMBINE
# ============================================================

error_analysis = pd.concat(
    all_error_rows,
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

error_analysis.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("ERROR ANALYSIS SAVED")
print("=" * 70)

print(
    f"\n{OUTPUT_FILE}"
)

print(
    "\n600 MW ERROR ANALYSIS: PASS"
)

print("=" * 70)