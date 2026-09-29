from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_runtime_forecast_results.csv"
)

OPTIMIZED = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_runtime_forecast_results_optimized.csv"
)


PREDICTION_COLUMNS = [
    "forecast_2min",
    "forecast_10min",
    "forecast_30min",
]


print("=" * 70)
print("600 MW ORIGINAL vs BATCH PREDICTION VALIDATION")
print("=" * 70)


original = pd.read_csv(ORIGINAL)
optimized = pd.read_csv(OPTIMIZED)


print(f"\nOriginal rows   : {len(original)}")
print(f"Optimized rows  : {len(optimized)}")


# ============================================================
# ROW COUNT
# ============================================================

if len(original) != len(optimized):
    raise RuntimeError(
        "Row count mismatch."
    )


# ============================================================
# TIMESTAMP CHECK
# ============================================================

if not original["timestamp"].equals(
    optimized["timestamp"]
):
    raise RuntimeError(
        "Timestamp sequence mismatch."
    )


# ============================================================
# REPLAY STEP CHECK
# ============================================================

if not original["replay_step"].equals(
    optimized["replay_step"]
):
    raise RuntimeError(
        "Replay step mismatch."
    )


# ============================================================
# ACTUAL POWER CHECK
# ============================================================

if not np.allclose(
    original["actual_power"].to_numpy(),
    optimized["actual_power"].to_numpy(),
    rtol=0,
    atol=1e-12
):
    raise RuntimeError(
        "Actual power mismatch."
    )


# ============================================================
# PREDICTION COMPARISON
# ============================================================

print("\nPREDICTION COMPARISON")

all_pass = True

for column in PREDICTION_COLUMNS:

    original_values = original[
        column
    ].to_numpy(dtype=float)

    optimized_values = optimized[
        column
    ].to_numpy(dtype=float)

    absolute_difference = np.abs(
        original_values - optimized_values
    )

    max_difference = (
        absolute_difference.max()
    )

    mean_difference = (
        absolute_difference.mean()
    )

    exact_matches = np.sum(
        original_values == optimized_values
    )

    close_matches = np.sum(
        np.isclose(
            original_values,
            optimized_values,
            rtol=1e-10,
            atol=1e-10
        )
    )

    print(f"\n{column}")

    print(
        f"  Exact matches       : "
        f"{exact_matches}/{len(original_values)}"
    )

    print(
        f"  Within tolerance    : "
        f"{close_matches}/{len(original_values)}"
    )

    print(
        f"  Max absolute diff   : "
        f"{max_difference:.15f}"
    )

    print(
        f"  Mean absolute diff  : "
        f"{mean_difference:.15f}"
    )

    if not np.allclose(
        original_values,
        optimized_values,
        rtol=1e-10,
        atol=1e-10
    ):
        all_pass = False


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

if all_pass:

    print(
        "PREDICTION EQUIVALENCE: PASS"
    )

    print(
        "\nOriginal and batch predictions "
        "are numerically equivalent."
    )

else:

    print(
        "PREDICTION EQUIVALENCE: FAIL"
    )

print("=" * 70)