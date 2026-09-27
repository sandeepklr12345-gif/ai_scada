# Forecasting target
TARGET_COLUMN = "Power output\n（MW）"

# Dataset sampling interval
SAMPLING_MINUTES = 2

# Forecast horizons in minutes
FORECAST_HORIZONS = [2, 10, 30]

# Convert horizons into dataset rows
HORIZON_STEPS = {
    horizon: horizon // SAMPLING_MINUTES
    for horizon in FORECAST_HORIZONS
}