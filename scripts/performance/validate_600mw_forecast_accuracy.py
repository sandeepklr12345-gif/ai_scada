from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    / "600mw_forecast_accuracy_report.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "actual_power"

FORECASTS = {
    "2min": "forecast_2min",
    "10min": "forecast_10min",
    "30min": "forecast_30min",
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("600 MW FORECAST ACCURACY VALIDATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput rows : {len(df)}")

required = [
    TARGET,
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


# ============================================================
# VALIDATE NUMERIC DATA
# ============================================================

values = df[required].to_numpy(dtype=float)

if not np.isfinite(values).all():
    raise ValueError(
        "NaN or infinite values detected."
    )

print("Numeric validation : PASS")


# ============================================================
# CALCULATE METRICS
# ============================================================

actual = df[TARGET].to_numpy(dtype=float)

rows = []

for horizon, column in FORECASTS.items():

    predicted = df[column].to_numpy(dtype=float)

    error = predicted - actual
    absolute_error = np.abs(error)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    # Avoid division by zero
    nonzero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (
                    actual[nonzero]
                    - predicted[nonzero]
                )
                / actual[nonzero]
            )
        )
        * 100
    )

    mean_error = np.mean(error)

    max_error = np.max(
        absolute_error
    )

    std_error = np.std(error)

    within_5 = (
        np.mean(
            absolute_error
            <= 0.05 * np.abs(actual)
        )
        * 100
    )

    within_10 = (
        np.mean(
            absolute_error
            <= 0.10 * np.abs(actual)
        )
        * 100
    )

    rows.append({
        "horizon": horizon,
        "samples": len(actual),
        "MAE_MW": mae,
        "RMSE_MW": rmse,
        "MAPE_percent": mape,
        "R2": r2,
        "mean_error_MW": mean_error,
        "max_absolute_error_MW": max_error,
        "error_std_MW": std_error,
        "within_5_percent": within_5,
        "within_10_percent": within_10,
    })


# ============================================================
# REPORT
# ============================================================

report = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("FORECAST ACCURACY RESULTS")
print("=" * 70)

for _, row in report.iterrows():

    print(
        f"\n{row['horizon']} FORECAST"
    )

    print(
        f"  Samples             : "
        f"{int(row['samples'])}"
    )

    print(
        f"  MAE                 : "
        f"{row['MAE_MW']:.4f} MW"
    )

    print(
        f"  RMSE                : "
        f"{row['RMSE_MW']:.4f} MW"
    )

    print(
        f"  MAPE                : "
        f"{row['MAPE_percent']:.4f}%"
    )

    print(
        f"  R²                  : "
        f"{row['R2']:.6f}"
    )

    print(
        f"  Mean error          : "
        f"{row['mean_error_MW']:.4f} MW"
    )

    print(
        f"  Max absolute error  : "
        f"{row['max_absolute_error_MW']:.4f} MW"
    )

    print(
        f"  Error std           : "
        f"{row['error_std_MW']:.4f} MW"
    )

    print(
        f"  Within ±5%          : "
        f"{row['within_5_percent']:.2f}%"
    )

    print(
        f"  Within ±10%         : "
        f"{row['within_10_percent']:.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

report.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("REPORT SAVED")
print("=" * 70)

print(
    f"\n{OUTPUT_FILE}"
)

print("\n600 MW FORECAST ACCURACY VALIDATION: PASS")
print("=" * 70)