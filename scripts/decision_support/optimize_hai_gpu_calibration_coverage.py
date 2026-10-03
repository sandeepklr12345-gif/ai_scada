from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import itertools
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
    / "temporal_gpu"
    / "train.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
    / "temporal_gpu"
    / "calibration_scenario_selection.csv"
)


# ============================================================
# PRIMARY TRAINING SCENARIOS
# ============================================================

PRIMARY_SCENARIOS = {
    "A202", "A204", "A205", "A206", "A207",
    "A208", "A210", "A211", "A212", "A215",
    "A216", "A217", "A218", "A219", "A220",
    "A221", "A222", "A223", "A224", "A225",
    "A226", "A227", "A228", "A229", "A230",
    "A231", "A232", "A233", "A234", "A237",
    "A238",
}


# ============================================================
# FINAL EVALUATION SCENARIOS
# ============================================================

FINAL_EVALUATION_SCENARIOS = {
    "A201",
    "A203",
    "A209",
    "A213",
    "A214",
    "A235",
    "A236",
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("HAI 23.05 GPU CALIBRATION COVERAGE OPTIMIZATION")
print("=" * 100)

print("\n[1] Loading primary training data")

df = pd.read_csv(INPUT_PATH)

print(f"Input shape: {df.shape}")


# ============================================================
# TARGET COLUMNS
# ============================================================

target_columns = [
    column
    for column in df.columns
    if column.startswith("AP") or column.startswith("AE")
]

if len(target_columns) != 39:
    raise ValueError(
        f"Expected 39 attack-code columns, found {len(target_columns)}"
    )

print(f"Attack mechanisms: {len(target_columns)}")


# ============================================================
# BUILD SCENARIO INCIDENCE MATRIX
# ============================================================

print("\n[2] Building scenario/code incidence matrix")

scenario_rows = []

for scenario in sorted(PRIMARY_SCENARIOS):

    scenario_df = df[
        df["scenario_id"].astype(str) == scenario
    ]

    if scenario_df.empty:
        raise ValueError(
            f"No rows found for primary scenario {scenario}"
        )

    active_codes = [
        code
        for code in target_columns
        if scenario_df[code].sum() > 0
    ]

    scenario_rows.append(
        {
            "scenario_id": scenario,
            "row_count": len(scenario_df),
            "attack_codes": ",".join(sorted(active_codes)),
            "code_count": len(active_codes),
            "_codes": set(active_codes),
        }
    )


# ============================================================
# VERIFY GLOBAL COVERAGE
# ============================================================

all_codes = set(target_columns)

covered_by_primary = set()

for row in scenario_rows:
    covered_by_primary.update(row["_codes"])

missing_global = all_codes - covered_by_primary

if missing_global:
    raise ValueError(
        "Primary training scenarios do not cover all mechanisms: "
        f"{sorted(missing_global)}"
    )

print(
    f"Primary scenarios cover all {len(all_codes)} mechanisms."
)


# ============================================================
# FIND MINIMUM SET COVER
# ============================================================

print("\n[3] Searching for minimum scenario coverage")

scenario_names = [
    row["scenario_id"]
    for row in scenario_rows
]

scenario_sets = {
    row["scenario_id"]: row["_codes"]
    for row in scenario_rows
}


# Exact search by increasing number of scenarios.
#
# There are only 31 primary scenarios, so combinations are
# manageable for the small minimum cover sizes expected here.

best_solution = None

for size in range(1, len(scenario_names) + 1):

    print(f"Checking combinations of size {size}...")

    found = None

    for combination in itertools.combinations(
        scenario_names,
        size,
    ):

        covered = set()

        for scenario in combination:
            covered.update(scenario_sets[scenario])

        if covered == all_codes:
            found = combination
            break

    if found is not None:
        best_solution = found
        print(
            f"Minimum coverage found with {size} scenarios."
        )
        break


if best_solution is None:
    raise RuntimeError(
        "Could not find a scenario set covering all mechanisms."
    )


# ============================================================
# BUILD SELECTION TABLE
# ============================================================

selected = set(best_solution)

selection_rows = []

covered_so_far = set()

for row in scenario_rows:

    scenario = row["scenario_id"]

    if scenario in selected:

        new_codes = row["_codes"] - covered_so_far
        covered_so_far.update(row["_codes"])

        selection_rows.append(
            {
                "scenario_id": scenario,
                "selected_for_calibration": True,
                "row_count": row["row_count"],
                "attack_codes": row["attack_codes"],
                "code_count": row["code_count"],
                "new_codes_added": ",".join(
                    sorted(new_codes)
                ),
                "new_code_count": len(new_codes),
            }
        )


selection_df = pd.DataFrame(selection_rows)


# ============================================================
# VERIFY FINAL EVALUATION IS DISJOINT
# ============================================================

if selected & FINAL_EVALUATION_SCENARIOS:
    raise ValueError(
        "Calibration selection overlaps final evaluation scenarios"
    )


# ============================================================
# COVERAGE SUMMARY
# ============================================================

covered_codes = set()

for scenario in selected:
    covered_codes.update(
        scenario_sets[scenario]
    )

missing_codes = all_codes - covered_codes

print("\n[4] Coverage result")

print(
    f"Selected calibration scenarios: "
    f"{len(selected)}"
)

print(
    "Selected scenarios:"
)

for scenario in sorted(selected):
    print(f"  {scenario}")


print(
    f"\nCovered mechanisms: "
    f"{len(covered_codes)}/{len(all_codes)}"
)

if missing_codes:
    print(
        "Missing mechanisms:",
        sorted(missing_codes),
    )
else:
    print(
        "All 39 attack mechanisms are covered."
    )


# ============================================================
# SAVE
# ============================================================

print("\n[5] Saving selection")

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

selection_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"Saved: {OUTPUT_PATH}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("STEP 6.16E.1 COMPLETE")
print("=" * 100)

print("\nNo model was trained.")
print("No GPU model was modified.")
print("Final evaluation scenarios remain untouched.")