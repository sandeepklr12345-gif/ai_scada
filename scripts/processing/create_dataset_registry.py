"""
STEP 3 - FINAL DATASET REGISTRY

Creates the authoritative registry for the five core project datasets
and their important processed/feature artifacts.

This script is organizational only:
- does not modify datasets
- does not delete files
- does not retrain models
"""

from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "data" / "dataset_registry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

registry_csv = OUTPUT_DIR / "dataset_registry.csv"
registry_report = OUTPUT_DIR / "dataset_registry_report.txt"


# ============================================================
# CORE DATASET REGISTRY
# ============================================================

datasets = [
    {
        "dataset": "600 MW Unit One-Week Operating Data",
        "source": "Mendeley Data, DOI 10.17632/5w59bs5t59.1",
        "raw_location": "data/raw/600mw/600 MW unit one-week operating data.xlsx",
        "processed_location": "data/processed/600mw",
        "project_role": "Primary power-output forecasting / load-estimation component",
        "structure": "5674 source rows, 78 columns; 2-minute sampling",
        "target": "Power output (MW)",
        "processing": "Cleaning, timestamp normalization, redundancy analysis, leakage review, lag features, 2/10/30-minute targets, candidate feature sets, validation",
        "status": "READY",
    },
    {
        "dataset": "HAI 23.05",
        "source": "ICS Dataset / HAI 23.05",
        "raw_location": "data/raw/hai/hai-23.05",
        "processed_location": "data/processed/hai/hai-23.05",
        "project_role": "Primary SCADA anomaly detection / predictive anomaly analysis",
        "structure": "86 SCADA variables; 1-second sampling; train/test datasets with aligned labels",
        "target": "Anomaly / attack label for evaluation",
        "processing": "Label alignment, training-only feature selection, redundancy review, temporal representation, candidate-C selection, Isolation Forest baseline/final detector, inference-engine validation",
        "status": "READY",
    },
    {
        "dataset": "6.6 MPa Steam Generator",
        "source": "Mendeley Data engineering dataset",
        "raw_location": "data/raw/steam_generator_6_6mpa/Boiler_Operational_Dataset_Tables_REVISED.xlsx",
        "processed_location": "data/processed/steam_generator_6_6mpa",
        "project_role": "Supporting boiler, steam and thermodynamic engineering reference",
        "structure": "Six cleaned engineering tables; operating conditions at 70.73%, 92.65%, 100% MCR",
        "target": "None; supporting engineering reference",
        "processing": "Workbook table extraction, structural cleaning, preservation of source values, non-destructive validation",
        "status": "READY",
    },
    {
        "dataset": "UCI Combined Cycle Power Plant",
        "source": "UCI Machine Learning Repository",
        "raw_location": "data/raw/uci_ccpp/Folds5x2_pp.xlsx",
        "processed_location": "data/processed/uci_ccpp/uci_ccpp_clean.csv",
        "project_role": "Supporting power-generation modeling / cross-validation",
        "structure": "9568 rows, 5 variables; AT, V, AP, RH, PE",
        "target": "PE (net hourly electrical energy output)",
        "processing": "Column/schema inspection, missing-value validation, duplicate review, numerical validation",
        "status": "READY",
    },
    {
        "dataset": "Coal Plant Metadata",
        "source": "Coal plant metadata CSV",
        "raw_location": "data/raw/coal_plants.csv",
        "processed_location": "data/processed/coal_plant_metadata.csv",
        "project_role": "Supporting plant-level context and metadata",
        "structure": "107 plants, 20 columns",
        "target": "None; metadata/context",
        "processing": "Schema inspection, missing-value validation, duplicate-ID validation, type review",
        "status": "READY",
    },
]


# ============================================================
# IMPORTANT DERIVED ARTIFACTS
# ============================================================

artifacts = [
    {
        "dataset": "600 MW forecasting dataset",
        "source": "Derived from 600 MW raw dataset",
        "raw_location": "data/raw/600mw",
        "processed_location": "data/features/600mw",
        "project_role": "Model-ready forecasting feature/target candidates",
        "structure": "Final validated candidate datasets; 5649 valid rows in full engineered dataset",
        "target": "2-minute, 10-minute, 30-minute future power output",
        "processing": "Lagged features and future targets with boundary rows removed",
        "status": "READY",
    },
    {
        "dataset": "HAI 23.05 model-ready dataset",
        "source": "Derived from HAI 23.05",
        "raw_location": "data/raw/hai/hai-23.05",
        "processed_location": "data/features/hai/hai-23.05/model_ready",
        "project_role": "Baseline/model development input",
        "structure": "58 selected original features plus timestamp",
        "target": "Anomaly label only in evaluation datasets",
        "processing": "Training-only engineering feature resolution and feature selection",
        "status": "READY",
    },
    {
        "dataset": "HAI Candidate C temporal dataset",
        "source": "Derived from HAI 23.05 model-ready data",
        "raw_location": "data/features/hai/hai-23.05/model_ready",
        "processed_location": "data/features/hai/hai-23.05/temporal_representation/final_candidate",
        "project_role": "Final anomaly-detection candidate",
        "structure": "118 model features: 58 original + 30 abs-difference + 30 rolling-standard-deviation features",
        "target": "Anomaly label for evaluation",
        "processing": "Temporal feature engineering, redundancy review, candidate selection, robustness evaluation",
        "status": "READY",
    },
    {
        "dataset": "HAI Candidate C inference engine",
        "source": "Derived from final Candidate C pipeline",
        "raw_location": "data/features/hai/hai-23.05/temporal_representation/final_candidate",
        "processed_location": "scripts/inference/hai_candidate_C_inference.py",
        "project_role": "Reusable batch/streaming inference component",
        "structure": "118-feature inference schema",
        "target": "NORMAL / ANOMALY status",
        "processing": "Manifest-driven feature construction and deterministic batch/streaming validation",
        "status": "VALIDATED",
    },
]


# ============================================================
# WRITE CSV
# ============================================================

fieldnames = [
    "dataset",
    "source",
    "raw_location",
    "processed_location",
    "project_role",
    "structure",
    "target",
    "processing",
    "status",
]

with registry_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in datasets + artifacts:
        writer.writerow(row)


# ============================================================
# WRITE REPORT
# ============================================================

lines = []

lines.append("=" * 80)
lines.append("AI-SCADA DATASET REGISTRY")
lines.append("=" * 80)

lines.append("")
lines.append("PROJECT:")
lines.append("AI-BASED LOAD ESTIMATION AND PREDICTIVE OUTAGE ANALYSIS")
lines.append("INTEGRATED WITH SCADA IN SMART POWER SYSTEMS")

lines.append("")
lines.append("DATASET FOUNDATION STATUS: COMPLETE")

lines.append("")
lines.append("CORE DATASETS")
lines.append("-" * 80)

for i, row in enumerate(datasets, 1):
    lines.append("")
    lines.append(f"{i}. {row['dataset']}")
    lines.append(f"   Source      : {row['source']}")
    lines.append(f"   Raw         : {row['raw_location']}")
    lines.append(f"   Processed   : {row['processed_location']}")
    lines.append(f"   Role        : {row['project_role']}")
    lines.append(f"   Structure   : {row['structure']}")
    lines.append(f"   Target      : {row['target']}")
    lines.append(f"   Processing  : {row['processing']}")
    lines.append(f"   Status      : {row['status']}")

lines.append("")
lines.append("KEY DERIVED ARTIFACTS")
lines.append("-" * 80)

for i, row in enumerate(artifacts, 1):
    lines.append("")
    lines.append(f"{i}. {row['dataset']}")
    lines.append(f"   Source      : {row['source']}")
    lines.append(f"   Location    : {row['processed_location']}")
    lines.append(f"   Role        : {row['project_role']}")
    lines.append(f"   Structure   : {row['structure']}")
    lines.append(f"   Target      : {row['target']}")
    lines.append(f"   Processing  : {row['processing']}")
    lines.append(f"   Status      : {row['status']}")

lines.append("")
lines.append("DATASET ROLES")
lines.append("-" * 80)
lines.append("600 MW        -> primary power-output forecasting")
lines.append("HAI 23.05     -> primary SCADA anomaly detection")
lines.append("6.6 MPa       -> supporting engineering reference")
lines.append("UCI CCPP      -> supporting power-generation modeling")
lines.append("Coal metadata -> supporting plant context")

lines.append("")
lines.append("DATA INTEGRITY PRINCIPLES")
lines.append("-" * 80)
lines.append("1. Raw datasets are preserved.")
lines.append("2. Processed datasets are kept separate from raw data.")
lines.append("3. Dataset-specific processing is documented.")
lines.append("4. HAI labels are treated as anomaly/attack labels, not fabricated outage labels.")
lines.append("5. 600 MW power output is treated as the forecasting target serving the load-estimation component.")
lines.append("6. Supporting datasets are not blindly concatenated with the primary ML datasets.")
lines.append("7. Structural blanks in the 6.6 MPa engineering tables were not blindly imputed.")
lines.append("8. Exact UCI duplicate observations were retained after review.")

lines.append("")
lines.append("PHASE CLOSURE")
lines.append("-" * 80)
lines.append("STEP 1: Dataset cleaning                    COMPLETE")
lines.append("STEP 2: Dataset validation                  COMPLETE")
lines.append("STEP 3: Dataset registry and closure        COMPLETE")

lines.append("")
lines.append("DATASET FOUNDATION: CLOSED")
lines.append("")
lines.append("The project can now proceed to the next engineering phase.")
lines.append("=" * 80)

registry_report.write_text("\n".join(lines), encoding="utf-8")

print("")
print("=" * 80)
print("STEP 3 - FINAL DATASET REGISTRY")
print("=" * 80)
print("")
print("Core datasets registered: 5")
print("Derived artifacts registered: 4")
print("")
print("Registry:")
print(registry_csv)
print("")
print("Report:")
print(registry_report)
print("")
print("STEP 1 - CLEANING              COMPLETE")
print("STEP 2 - VALIDATION            COMPLETE")
print("STEP 3 - REGISTRY + CLOSURE    COMPLETE")
print("")
print("DATASET FOUNDATION: CLOSED")
print("=" * 80)