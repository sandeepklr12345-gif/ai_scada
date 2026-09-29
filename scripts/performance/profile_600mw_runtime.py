import time
import os
import sys
import runpy


SCRIPT = r"scripts/integration/run_600mw_runtime_forecast.py"


# ================================================================
# 600 MW RUNTIME PROFILER
# ================================================================

print("=" * 70)
print("600 MW RUNTIME PROFILER")
print("=" * 70)

print("\nProfiling current runtime...")
print("No forecasting logic is being modified.\n")


# ------------------------------------------------
# PROFILE THE COMPLETE SCRIPT
# ------------------------------------------------

start = time.perf_counter()

runpy.run_path(
    SCRIPT,
    run_name="__main__"
)

end = time.perf_counter()

total_time = end - start


# ------------------------------------------------
# RESULTS
# ------------------------------------------------

output_file = (
    r"data/integration/runtime_simulation/"
    r"600mw_runtime_forecast_results.csv"
)

print("\n" + "=" * 70)
print("PROFILE SUMMARY")
print("=" * 70)

print(f"\nTotal runtime : {total_time:.4f} seconds")

if os.path.exists(output_file):

    import pandas as pd

    df = pd.read_csv(output_file)

    rows = len(df)

    print(f"Rows          : {rows}")
    print(f"Rows/sec      : {rows / total_time:.2f}")
    print(f"Time/row      : {(total_time / rows) * 1000:.4f} ms")

print("\n" + "=" * 70)
print("NEXT STEP")
print("=" * 70)

print("""
The complete runtime has now been measured.

Next we will profile the individual stages:
    1. Data loading
    2. Feature preparation
    3. Model input preparation
    4. Chronological inference
    5. Result generation
    6. CSV output
""")

print("=" * 70)