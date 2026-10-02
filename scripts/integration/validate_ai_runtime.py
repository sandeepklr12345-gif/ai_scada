from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "integration"
    / "ai_runtime.py"
)

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


# ============================================================
# LOAD RUNTIME
# ============================================================

print("=" * 80)
print("UNIFIED AI RUNTIME FUNCTIONAL VALIDATION")
print("=" * 80)

print()
print("Loading unified runtime...")

spec = importlib.util.spec_from_file_location(
    "ai_runtime",
    RUNTIME_PATH
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Could not load runtime from: {RUNTIME_PATH}"
    )

runtime_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime_module)

AIRuntime = runtime_module.AIRuntime

runtime = AIRuntime()

print("Unified runtime loaded.")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading validation inputs...")

forecast_df = pd.read_csv(
    FORECAST_DATA
)

hai_df = pd.read_csv(
    HAI_DATA
)

print(
    f"600 MW rows available : "
    f"{len(forecast_df)}"
)

print(
    f"HAI rows available    : "
    f"{len(hai_df)}"
)


# ============================================================
# PREPARE INPUTS
# ============================================================

forecast_row = forecast_df.iloc[[100]].copy()

hai_features = [
    column
    for column in hai_df.columns
    if column not in ["timestamp", "label"]
]

if len(hai_features) != 58:
    raise RuntimeError(
        f"Expected 58 HAI features, "
        f"found {len(hai_features)}"
    )

# Use rows 0-4 so the HAI runtime receives
# its required history sequentially.
hai_rows = hai_df[
    hai_features
].iloc[:5].copy()


# ============================================================
# TEST 1: FORECAST()
# ============================================================

print()
print("=" * 80)
print("TEST 1: FORECAST INTERFACE")
print("=" * 80)

forecast_result = runtime.forecast(
    forecast_row
)

print("Forecast result:")
print(forecast_result)


required_forecast_keys = [
    "power_2min",
    "power_10min",
    "power_30min",
]

if set(forecast_result.keys()) != set(
    required_forecast_keys
):
    raise RuntimeError(
        "Forecast output schema mismatch."
    )

for key in required_forecast_keys:

    if not np.isfinite(
        forecast_result[key]
    ):
        raise RuntimeError(
            f"Invalid forecast value: {key}"
        )

print("Forecast interface: PASS")


# ============================================================
# TEST 2: DETECT_ANOMALY()
# ============================================================

print()
print("=" * 80)
print("TEST 2: ANOMALY INTERFACE")
print("=" * 80)

# Feed the first four rows to initialize history.
for i in range(4):

    runtime.detect_anomaly(
        hai_rows.iloc[[i]]
    )

anomaly_result = runtime.detect_anomaly(
    hai_rows.iloc[[4]]
)

print("Anomaly result:")
print(anomaly_result)

required_anomaly_keys = [
    "anomaly_score",
    "prediction",
    "status",
]

if set(anomaly_result.keys()) != set(
    required_anomaly_keys
):
    raise RuntimeError(
        "Anomaly output schema mismatch."
    )

if anomaly_result["prediction"] not in [0, 1, None]:
    raise RuntimeError(
        "Invalid anomaly prediction."
    )

if not isinstance(
    anomaly_result["status"],
    str
):
    raise RuntimeError(
        "Invalid anomaly status."
    )

if (
    anomaly_result["anomaly_score"]
    is not None
    and not np.isfinite(
        anomaly_result["anomaly_score"]
    )
):
    raise RuntimeError(
        "Invalid anomaly score."
    )

print("Anomaly interface: PASS")


# ============================================================
# TEST 3: FRESH RUNTIME FOR UNIFIED PREDICT()
# ============================================================

print()
print("=" * 80)
print("TEST 3: UNIFIED PREDICT INTERFACE")
print("=" * 80)

runtime_unified = AIRuntime()

# Initialize HAI history.
for i in range(4):

    runtime_unified.detect_anomaly(
        hai_rows.iloc[[i]]
    )

unified_result = runtime_unified.predict(
    forecast_row,
    hai_rows.iloc[[4]]
)

print("Unified result:")
print(unified_result)


if set(unified_result.keys()) != {
    "forecast",
    "anomaly",
}:
    raise RuntimeError(
        "Unified output schema mismatch."
    )

if set(
    unified_result["forecast"].keys()
) != set(required_forecast_keys):
    raise RuntimeError(
        "Unified forecast schema mismatch."
    )

if set(
    unified_result["anomaly"].keys()
) != set(required_anomaly_keys):
    raise RuntimeError(
        "Unified anomaly schema mismatch."
    )

print("Unified predict interface: PASS")


# ============================================================
# TEST 4: PREDICT_SINGLE()
# ============================================================

print()
print("=" * 80)
print("TEST 4: PREDICT_SINGLE INTERFACE")
print("=" * 80)

runtime_single = AIRuntime()

# Initialize HAI history.
for i in range(4):

    runtime_single.detect_anomaly(
        hai_rows.iloc[[i]]
    )

single_result = runtime_single.predict_single(
    forecast_row,
    hai_rows.iloc[[4]]
)

print("Single-row result:")
print(single_result)


if set(single_result.keys()) != {
    "forecast",
    "anomaly",
}:
    raise RuntimeError(
        "predict_single schema mismatch."
    )

print("predict_single interface: PASS")


# ============================================================
# TEST 5: FORECAST CONSISTENCY
# ============================================================

print()
print("=" * 80)
print("TEST 5: FORECAST CONSISTENCY")
print("=" * 80)

for key in required_forecast_keys:

    difference = abs(
        forecast_result[key]
        - unified_result["forecast"][key]
    )

    print(
        f"{key}: difference = "
        f"{difference:.18f}"
    )

    if difference > 1e-12:
        raise RuntimeError(
            f"Forecast mismatch for {key}"
        )

print("Forecast consistency: PASS")


# ============================================================
# TEST 6: ANOMALY CONSISTENCY
# ============================================================

print()
print("=" * 80)
print("TEST 6: ANOMALY CONSISTENCY")
print("=" * 80)

direct_anomaly = anomaly_result
unified_anomaly = unified_result["anomaly"]

if (
    direct_anomaly["prediction"]
    != unified_anomaly["prediction"]
):
    raise RuntimeError(
        "Anomaly prediction mismatch."
    )

if (
    direct_anomaly["status"]
    != unified_anomaly["status"]
):
    raise RuntimeError(
        "Anomaly status mismatch."
    )

direct_score = direct_anomaly[
    "anomaly_score"
]

unified_score = unified_anomaly[
    "anomaly_score"
]

if direct_score is None and unified_score is None:
    score_difference = 0.0

elif direct_score is not None and unified_score is not None:
    score_difference = abs(
        direct_score
        - unified_score
    )

else:
    raise RuntimeError(
        "Anomaly score None/value mismatch."
    )

print(
    "Anomaly score difference : "
    f"{score_difference:.18f}"
)

if score_difference > 1e-12:
    raise RuntimeError(
        "Anomaly score mismatch."
    )

print("Anomaly consistency: PASS")


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 80)
print("UNIFIED AI RUNTIME VALIDATION: PASS")
print("=" * 80)

print()
print("Verified:")
print("  Forecast interface       : PASS")
print("  Anomaly interface        : PASS")
print("  Unified predict()        : PASS")
print("  predict_single()         : PASS")
print("  Forecast consistency     : PASS")
print("  Anomaly consistency      : PASS")
print()
print("STATUS: AI RUNTIME INTERFACE VALIDATED")