from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
    / "temporal_gpu"
)

TRAIN_OUTPUT = OUTPUT_DIR / "classifier_train.csv"
CALIBRATION_OUTPUT = OUTPUT_DIR / "calibration.csv"


# ============================================================
# PRIMARY TRAINING SCENARIOS
# ============================================================

TRAIN_SCENARIOS = {
    "A202", "A204", "A205", "A206", "A207",
    "A208", "A210", "A211", "A212", "A215",
    "A216", "A217", "A218", "A219", "A220",
    "A221", "A222", "A223", "A224", "A225",
    "A226", "A227", "A228", "A229", "A230",
    "A231", "A232", "A233", "A234", "A237",
    "A238",
}


# ============================================================
# CALIBRATION SCENARIOS
# ============================================================

# Select scenarios from the primary training pool.
# These remain completely separate from the final 7-scenario
# evaluation set.

CALIBRATION_SCENARIOS = {
    "A221",
    "A222",
    "A223",
    "A224",
    "A225",
    "A226",
    "A227",
}


# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("HAI 23.05 GPU CALIBRATION SPLIT DESIGN")
print("=" * 100)

print("\n[1] Loading primary training dataset")

df = pd.read_csv(INPUT_PATH)

print(f"Input shape: {df.shape}")


# ============================================================
# VALIDATION
# ============================================================

print("\n[2] Validating scenarios")

if "scenario_id" not in df.columns:
    raise ValueError("scenario_id column missing")

available = set(
    df["scenario_id"]
    .dropna()
    .astype(str)
    .unique()
)

missing_train = TRAIN_SCENARIOS - available

if missing_train:
    raise ValueError(
        f"Training scenarios missing: {sorted(missing_train)}"
    )

missing_calibration = CALIBRATION_SCENARIOS - TRAIN_SCENARIOS

if missing_calibration:
    raise ValueError(
        "Calibration scenario outside primary training pool: "
        f"{sorted(missing_calibration)}"
    )


# ============================================================
# SPLIT
# ============================================================

print("\n[3] Creating scenario-level calibration split")

calibration_mask = (
    df["scenario_id"]
    .astype(str)
    .isin(CALIBRATION_SCENARIOS)
)

train_mask = (
    df["scenario_id"]
    .astype(str)
    .isin(TRAIN_SCENARIOS - CALIBRATION_SCENARIOS)
)

classifier_train = df.loc[train_mask].copy()
calibration = df.loc[calibration_mask].copy()


# ============================================================
# VALIDATION
# ============================================================

print("\n[4] Validating split")

classifier_scenarios = set(
    classifier_train["scenario_id"]
    .dropna()
    .astype(str)
    .unique()
)

calibration_scenarios = set(
    calibration["scenario_id"]
    .dropna()
    .astype(str)
    .unique()
)

if classifier_scenarios & calibration_scenarios:
    raise ValueError("Scenario overlap detected")

if classifier_scenarios | calibration_scenarios != TRAIN_SCENARIOS:
    raise ValueError(
        "Split does not preserve all primary training scenarios"
    )

if len(classifier_train) == 0:
    raise ValueError("Classifier training split is empty")

if len(calibration) == 0:
    raise ValueError("Calibration split is empty")


# ============================================================
# TARGET CHECK
# ============================================================

target_columns = [
    c for c in df.columns
    if c.startswith("AP") or c.startswith("AE")
]

print(f"Target columns: {len(target_columns)}")

if len(target_columns) != 39:
    raise ValueError(
        f"Expected 39 targets, found {len(target_columns)}"
    )


def attack_count(frame):
    return int(frame[target_columns].sum().sum())


print("\nClassifier training:")
print(f"Rows: {len(classifier_train):,}")
print(f"Scenarios: {len(classifier_scenarios)}")
print(f"Attack-label count: {attack_count(classifier_train):,}")

print("\nCalibration:")
print(f"Rows: {len(calibration):,}")
print(f"Scenarios: {len(calibration_scenarios)}")
print(f"Attack-label count: {attack_count(calibration):,}")


# ============================================================
# CODE COVERAGE
# ============================================================

train_codes = {
    code
    for code in target_columns
    if classifier_train[code].sum() > 0
}

calibration_codes = {
    code
    for code in target_columns
    if calibration[code].sum() > 0
}

print("\nAttack-code coverage:")
print(
    f"Classifier training codes: "
    f"{len(train_codes)}/39"
)

print(
    f"Calibration codes: "
    f"{len(calibration_codes)}/39"
)

print(
    "Calibration codes:",
    sorted(calibration_codes),
)


# ============================================================
# SAVE
# ============================================================

print("\n[5] Saving split")

classifier_train.to_csv(
    TRAIN_OUTPUT,
    index=False,
)

calibration.to_csv(
    CALIBRATION_OUTPUT,
    index=False,
)

print(f"Classifier training: {TRAIN_OUTPUT}")
print(f"Calibration data:    {CALIBRATION_OUTPUT}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("STEP 6.16E SPLIT DESIGN COMPLETE")
print("=" * 100)

print("\nFinal evaluation scenarios remain untouched.")
print("No model was retrained.")