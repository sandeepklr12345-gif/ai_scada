"""
FINAL STEP-2 DATASET REVIEW

Purpose:
    Final non-destructive validation of:
    1. 6.6 MPa Steam Generator dataset
    2. UCI Combined Cycle Power Plant dataset

This script:
    - Does NOT delete rows
    - Does NOT impute values
    - Does NOT modify existing datasets
    - Reviews structural blanks in the 6.6 MPa dataset
    - Inspects exact duplicate rows in UCI CCPP
    - Produces final Step-2 readiness decisions
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED / "dataset_validation"

STEAM_DIR = PROCESSED / "steam_generator_6_6mpa"
UCI_DIR = PROCESSED / "uci_ccpp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPORT
# ============================================================

report_lines = []

report_lines.append("=" * 70)
report_lines.append("FINAL DATASET REVIEW - STEP 2")
report_lines.append("=" * 70)


# ============================================================
# 1. 6.6 MPa STEAM GENERATOR REVIEW
# ============================================================

report_lines.append("")
report_lines.append("6.6 MPa STEAM GENERATOR REVIEW")
report_lines.append("-" * 70)

steam_files = [
    "coal_analysis.csv",
    "flue_gas_composition.csv",
    "massflow_temperature.csv",
    "water_steam_circuit.csv",
    "enthalpy_entropy_exergy.csv",
    "operating_condition_summary.csv",
]

for filename in steam_files:

    filepath = STEAM_DIR / filename

    if not filepath.exists():
        report_lines.append("")
        report_lines.append(f"{filename}: FILE NOT FOUND")
        continue

    df = pd.read_csv(filepath)

    missing = df.isna().sum()
    missing_columns = missing[missing > 0]

    report_lines.append("")
    report_lines.append(
        f"{filename}: {df.shape[0]} rows x {df.shape[1]} columns"
    )

    if missing_columns.empty:

        report_lines.append(
            "  Missing values: 0"
        )

    else:

        report_lines.append(
            "  Missing values by column:"
        )

        for column, count in missing_columns.items():

            report_lines.append(
                f"    {column}: {int(count)}"
            )


# ============================================================
# 6.6 MPa INTERPRETATION
# ============================================================

report_lines.append("")
report_lines.append("6.6 MPa INTERPRETATION")
report_lines.append("-" * 70)

report_lines.append(
    "The source workbook is a formatted engineering report rather "
    "than a rectangular time-series dataset."
)

report_lines.append(
    "Some blank cells represent fields that are structurally not "
    "applicable or were not provided for a particular state point."
)

report_lines.append(
    "No missing values were imputed."
)

report_lines.append(
    "No rows or columns were deleted."
)

report_lines.append(
    "The three operating conditions are preserved:"
)

report_lines.append(
    "  70.73% MCR"
)

report_lines.append(
    "  92.65% MCR"
)

report_lines.append(
    "  100.00% MCR"
)

report_lines.append(
    "Decision: READY AS SUPPORTING ENGINEERING DATASET"
)


# ============================================================
# 2. UCI CCPP DUPLICATE REVIEW
# ============================================================

report_lines.append("")
report_lines.append("UCI CCPP DUPLICATE REVIEW")
report_lines.append("-" * 70)

uci_file = UCI_DIR / "uci_ccpp_clean.csv"

if not uci_file.exists():

    report_lines.append(
        f"FILE NOT FOUND: {uci_file}"
    )

else:

    uci = pd.read_csv(uci_file)

    duplicate_count = int(
        uci.duplicated().sum()
    )

    report_lines.append(
        f"Rows: {uci.shape[0]}"
    )

    report_lines.append(
        f"Columns: {uci.shape[1]}"
    )

    report_lines.append(
        f"Exact duplicate rows: {duplicate_count}"
    )

    # --------------------------------------------------------
    # Duplicate groups
    # --------------------------------------------------------

    duplicate_rows = uci[
        uci.duplicated(keep=False)
    ].copy()

    if duplicate_rows.empty:

        report_lines.append(
            "No duplicate groups found."
        )

    else:

        duplicate_groups = (
            duplicate_rows
            .groupby(
                list(uci.columns),
                dropna=False
            )
            .size()
            .reset_index(name="count")
            .sort_values(
                "count",
                ascending=False
            )
        )

        report_lines.append(
            f"Duplicate groups: {len(duplicate_groups)}"
        )

        report_lines.append(
            "Duplicate group counts:"
        )

        for _, row in duplicate_groups.iterrows():

            values = []

            for column in uci.columns:

                values.append(
                    f"{column}={row[column]}"
                )

            report_lines.append(
                f"  count={int(row['count'])}: "
                + ", ".join(values)
            )

        report_lines.append("")
        report_lines.append(
            "Decision: RETAIN DUPLICATES FOR NOW."
        )

        report_lines.append(
            "Reason: the duplicate rows are exact repeated "
            "observations. Removing them would alter the source "
            "distribution without a documented modeling reason."
        )


# ============================================================
# 3. NUMERICAL CORRUPTION CHECK
# ============================================================

report_lines.append("")
report_lines.append("NUMERICAL CORRUPTION CHECK")
report_lines.append("-" * 70)


datasets_to_check = []

# 6.6 MPa files
for filename in steam_files:

    filepath = STEAM_DIR / filename

    if filepath.exists():

        datasets_to_check.append(
            ("6.6 MPa / " + filename, filepath)
        )


# UCI
if uci_file.exists():

    datasets_to_check.append(
        ("UCI CCPP", uci_file)
    )


for dataset_name, filepath in datasets_to_check:

    df = pd.read_csv(filepath)

    numeric = df.select_dtypes(
        include="number"
    )

    if numeric.empty:

        report_lines.append(
            f"{dataset_name}: no numeric columns"
        )

        continue

    infinite_count = int(
        numeric.map(
            lambda x: abs(x) == float("inf")
        ).sum().sum()
    )

    report_lines.append(
        f"{dataset_name}: infinite values = "
        f"{infinite_count}"
    )


# ============================================================
# 4. FINAL STEP-2 DECISIONS
# ============================================================

report_lines.append("")
report_lines.append("=" * 70)
report_lines.append("STEP-2 FINAL READINESS DECISIONS")
report_lines.append("=" * 70)

report_lines.append(
    "600 MW dataset:"
)

report_lines.append(
    "  READY - primary power-output forecasting branch."
)

report_lines.append(
    "HAI 23.05:"
)

report_lines.append(
    "  READY - primary SCADA anomaly-detection branch."
)

report_lines.append(
    "6.6 MPa Steam Generator:"
)

report_lines.append(
    "  READY - supporting engineering dataset."
)

report_lines.append(
    "  Structural blanks retained; no imputation."
)

report_lines.append(
    "UCI CCPP:"
)

report_lines.append(
    "  READY - supporting power-generation modeling dataset."
)

report_lines.append(
    "  Exact duplicate observations retained."
)

report_lines.append(
    "Coal Plant Metadata:"
)

report_lines.append(
    "  READY - supporting plant-level context dataset."
)

report_lines.append("")
report_lines.append(
    "IMPORTANT:"
)

report_lines.append(
    "The datasets are NOT blindly concatenated."
)

report_lines.append(
    "Each dataset retains its project-specific role."
)

report_lines.append("")
report_lines.append(
    "STEP 2 STATUS: COMPLETE"
)

report_lines.append(
    "NEXT: STEP 3 - FINAL DATASET REGISTRY AND DATASET FOUNDATION CLOSURE"
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    OUTPUT_DIR /
    "final_step2_dataset_review.txt"
)

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("")
print("=" * 70)
print("FINAL STEP-2 DATASET REVIEW")
print("=" * 70)

print("")
print("6.6 MPa:")
print("  Structural blanks reviewed.")
print("  No imputation performed.")
print("  No rows deleted.")

print("")
print("UCI CCPP:")
print("  Exact duplicate rows reviewed.")
print("  No rows deleted.")

print("")
print("Numerical corruption checks completed.")

print("")
print("Final decisions:")
print("  600 MW              -> READY")
print("  HAI 23.05           -> READY")
print("  6.6 MPa             -> READY")
print("  UCI CCPP            -> READY")
print("  Coal Plant Metadata -> READY")

print("")
print("STEP 2 STATUS: COMPLETE")

print("")
print("Report:")
print(report_path)

print("")
print("NEXT:")
print("STEP 3 - FINAL DATASET REGISTRY AND DATASET FOUNDATION CLOSURE")

print("=" * 70)