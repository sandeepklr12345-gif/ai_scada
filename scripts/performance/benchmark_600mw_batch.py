import time
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "performance"
    / "run_600mw_batch_optimized.py"
)


print("=" * 70)
print("600 MW BATCH OPTIMIZATION BENCHMARK")
print("=" * 70)

print("\nRunning optimized implementation...\n")

start = time.perf_counter()

result = subprocess.run(
    [
        sys.executable,
        "-X",
        "utf8",
        str(SCRIPT)
    ],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace"
)

end = time.perf_counter()

elapsed = end - start

print(result.stdout)

if result.returncode != 0:

    print("\nOPTIMIZED RUNTIME FAILED")
    print(result.stderr)

    sys.exit(result.returncode)


# ============================================================
# RESULTS
# ============================================================

ROWS = 5649

original_time = 290.3856

rows_per_second = ROWS / elapsed

time_per_row = (
    elapsed / ROWS
) * 1000

speedup = (
    original_time / elapsed
)

time_saved = (
    original_time - elapsed
)

percentage_reduction = (
    time_saved / original_time
) * 100


print("\n" + "=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

print(
    f"\nOriginal runtime      : "
    f"{original_time:.4f} seconds"
)

print(
    f"Optimized runtime     : "
    f"{elapsed:.4f} seconds"
)

print(
    f"Time saved            : "
    f"{time_saved:.4f} seconds"
)

print(
    f"Runtime reduction     : "
    f"{percentage_reduction:.2f}%"
)

print(
    f"Speedup               : "
    f"{speedup:.2f}x"
)

print(
    f"\nRows processed        : "
    f"{ROWS}"
)

print(
    f"Optimized rows/sec    : "
    f"{rows_per_second:.2f}"
)

print(
    f"Optimized ms/row     : "
    f"{time_per_row:.4f} ms"
)

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)