import time
import os
import sys
import subprocess
import pandas as pd


# ================================================================
# 600 MW RUNTIME PERFORMANCE BENCHMARK
# ================================================================

SCRIPT = r"scripts/integration/run_600mw_runtime_forecast.py"

print("=" * 70)
print("600 MW RUNTIME PERFORMANCE BENCHMARK")
print("=" * 70)

print("\nStarting runtime...")
print("This benchmark measures the current CPU implementation.")
print("Do not modify the forecasting engine before this test.\n")

start = time.perf_counter()

result = subprocess.run(
    [sys.executable, "-X", "utf8", SCRIPT],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace"
)

end = time.perf_counter()

elapsed = end - start

print(result.stdout)

if result.returncode != 0:
    print("\nRUNTIME FAILED")
    print(result.stderr)
    sys.exit(result.returncode)

# ================================================================
# READ OUTPUT
# ================================================================

output_file = r"data/integration/runtime_simulation/600mw_runtime_forecast_results.csv"

if not os.path.exists(output_file):
    print("\nERROR: Runtime output file was not found.")
    sys.exit(1)

df = pd.read_csv(output_file)

rows = len(df)

# ================================================================
# RESULTS
# ================================================================

print("\n" + "=" * 70)
print("PERFORMANCE RESULTS")
print("=" * 70)

print(f"\nRows processed       : {rows}")
print(f"Total runtime        : {elapsed:.4f} seconds")

if elapsed > 0:
    print(f"Rows / second        : {rows / elapsed:.2f}")
    print(f"Time / row           : {(elapsed / rows) * 1000:.4f} ms")

print("\n" + "=" * 70)
print("SYSTEM")
print("=" * 70)

print(f"Python               : {sys.version.split()[0]}")

try:
    import torch

    print(f"PyTorch              : {torch.__version__}")
    print(f"CUDA available       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU                  : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU                  : Not being used by PyTorch")

except ImportError:
    print("PyTorch              : Not installed")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)