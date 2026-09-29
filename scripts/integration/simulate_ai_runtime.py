from pathlib import Path
import sys
import pandas as pd
import numpy as np

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts" / "integration")
)

from ai_model_integration import AIModelIntegration


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_DATA = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "600mw"
    / "candidates"
    / "feature_set_D_full_candidate_pool.csv"
)

HAI_DATA = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "model_ready"
    / "hai-test1_model_ready.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "integrated_runtime_simulation.csv"
)

FORECAST_ROWS = 100
HAI_ROWS = 100


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD MODEL INTEGRATION
# ============================================================

print_section("AI RUNTIME SIMULATION")

print("Loading integrated AI model...")

integration = AIModelIntegration()

print("Integrated model loaded successfully.")


# ============================================================
# LOAD DATA
# ============================================================

print_section("LOADING RUNTIME INPUT DATA")

forecast_df = pd.read_csv(FORECAST_DATA)
hai_df = pd.read_csv(HAI_DATA)

print(f"600 MW rows available : {len(forecast_df)}")
print(f"HAI rows available    : {len(hai_df)}")


# ============================================================
# PREPARE INPUTS
# ============================================================

forecast_sample = forecast_df.head(FORECAST_ROWS).copy()

# ============================================================
# HAI INPUT FEATURES
# ============================================================

# The HAI inference engine expects the original 58 SCADA
# features. The engine itself creates the Candidate C
# temporal features internally.

hai_features = [
    column
    for column in hai_df.columns
    if column not in ["timestamp", "label"]
]

if len(hai_features) != 58:
    raise RuntimeError(
        f"Expected 58 HAI input features, "
        f"found {len(hai_features)}"
    )

missing_features = [
    feature
    for feature in hai_features
    if feature not in hai_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing HAI input features: {missing_features}"
    )

hai_sample = hai_df[
    hai_features
].head(HAI_ROWS).copy()

print(
    f"HAI runtime input features : "
    f"{len(hai_features)}"
)


# ============================================================
# RUNTIME SIMULATION
# ============================================================

print_section("STARTING CONTINUOUS RUNTIME SIMULATION")

runtime_results = []

for i in range(HAI_ROWS):

    # --------------------------------------------------------
    # 600 MW forecast
    # --------------------------------------------------------

    forecast_row = forecast_sample.iloc[
        i % len(forecast_sample)
    ]

    forecast_input = pd.DataFrame(
        [forecast_row]
    )

    forecast_result = integration.predict_forecast(
        forecast_input
    )

    # --------------------------------------------------------
    # HAI streaming inference
    # --------------------------------------------------------

    hai_row = hai_sample.iloc[
        i
    ]

    hai_input = pd.DataFrame(
        [hai_row]
    )

    hai_result = integration.hai_detector.predict_stream(
        hai_input
    )

    # predict_stream returns a DataFrame
    hai_result_row = hai_result.iloc[0]

    # --------------------------------------------------------
    # Unified runtime record
    # --------------------------------------------------------

    runtime_results.append({
        "runtime_step": i + 1,

        "power_2min":
            forecast_result["power_2min"].iloc[0],

        "power_10min":
            forecast_result["power_10min"].iloc[0],

        "power_30min":
            forecast_result["power_30min"].iloc[0],

        "anomaly_score":
            hai_result_row["anomaly_score"],

        "anomaly_prediction":
            hai_result_row["prediction"],

        "anomaly_status":
            hai_result_row["status"],
    })

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (i + 1) % 10 == 0:
        print(
            f"Processed runtime step "
            f"{i + 1}/{HAI_ROWS}"
        )


# ============================================================
# CREATE OUTPUT
# ============================================================

results_df = pd.DataFrame(
    runtime_results
)


# ============================================================
# VALIDATION
# ============================================================

print_section("RUNTIME VALIDATION")

expected_rows = HAI_ROWS

actual_rows = len(results_df)

print(f"Expected output rows : {expected_rows}")
print(f"Actual output rows   : {actual_rows}")

if actual_rows != expected_rows:
    raise RuntimeError(
        "Runtime output row count mismatch."
    )


required_columns = [
    "runtime_step",
    "power_2min",
    "power_10min",
    "power_30min",
    "anomaly_score",
    "anomaly_prediction",
    "anomaly_status",
]

missing_output_columns = [
    column
    for column in required_columns
    if column not in results_df.columns
]

if missing_output_columns:
    raise RuntimeError(
        f"Missing output columns: "
        f"{missing_output_columns}"
    )


# Forecast values must be finite
forecast_columns = [
    "power_2min",
    "power_10min",
    "power_30min",
]

for column in forecast_columns:

    values = pd.to_numeric(
        results_df[column],
        errors="coerce"
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Invalid forecast values in {column}"
        )


# HAI history validation
insufficient_history = (
    results_df["anomaly_status"]
    == "INSUFFICIENT_HISTORY"
).sum()

expected_history_rows = (
    integration.hai_detector.HISTORY_SIZE
)

print(
    f"Expected insufficient-history rows : "
    f"{expected_history_rows}"
)

print(
    f"Actual insufficient-history rows   : "
    f"{insufficient_history}"
)

if insufficient_history != expected_history_rows:
    raise RuntimeError(
        "Unexpected HAI history initialization."
    )


# Remaining rows must have predictions
inference_rows = (
    results_df["anomaly_status"]
    != "INSUFFICIENT_HISTORY"
)

prediction_values = (
    results_df.loc[
        inference_rows,
        "anomaly_prediction"
    ]
    .dropna()
    .unique()
)

invalid_predictions = [
    value
    for value in prediction_values
    if value not in [0, 1]
]

if invalid_predictions:
    raise RuntimeError(
        f"Invalid HAI predictions: "
        f"{invalid_predictions}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print_section("RUNTIME SIMULATION SUMMARY")

print(
    f"Runtime steps processed : "
    f"{len(results_df)}"
)

print(
    f"Insufficient history    : "
    f"{insufficient_history}"
)

print(
    f"Actual inference rows   : "
    f"{inference_rows.sum()}"
)

normal_count = (
    results_df["anomaly_prediction"]
    == 0
).sum()

anomaly_count = (
    results_df["anomaly_prediction"]
    == 1
).sum()

print(
    f"Predicted NORMAL        : "
    f"{normal_count}"
)

print(
    f"Predicted ANOMALY       : "
    f"{anomaly_count}"
)

print()
print("Sample runtime output:")
print(results_df.head(10).to_string(index=False))

print()
print(f"Output saved to:")
print(OUTPUT_FILE)

print()
print("=" * 80)
print("INTEGRATED RUNTIME SIMULATION: PASS")
print("=" * 80)