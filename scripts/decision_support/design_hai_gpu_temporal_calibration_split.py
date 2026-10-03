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

TRAIN_OUTPUT = OUTPUT_DIR / "classifier_train_temporal.csv"
CALIBRATION_OUTPUT = OUTPUT_DIR / "calibration_temporal.csv"
GUARD_OUTPUT = OUTPUT_DIR / "calibration_guard_temporal.csv"


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
# LOAD
# ============================================================

print("=" * 100)
print("HAI 23.05 GPU TEMPORAL CALIBRATION SPLIT DESIGN")
print("=" * 100)

print("\n[1] Loading primary training data")

df = pd.read_csv(INPUT_PATH)

if "timestamp" not in df.columns:
    raise ValueError("timestamp column missing")

if "scenario_id" not in df.columns:
    raise ValueError("scenario_id column missing")

df["timestamp"] = pd.to_datetime(df["timestamp"])

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
        f"Expected 39 target columns, found {len(target_columns)}"
    )

print(f"Attack mechanisms: {len(target_columns)}")


# ============================================================
# VALIDATE EVALUATION DISJOINTNESS
# ============================================================

scenario_values = (
    df["scenario_id"]
    .dropna()
    .astype(str)
)

overlap = (
    set(scenario_values.unique())
    & FINAL_EVALUATION_SCENARIOS
)

if overlap:
    raise ValueError(
        "Final evaluation scenarios leaked into calibration source: "
        f"{sorted(overlap)}"
    )


# ============================================================
# FIND PRIMARY TRAINING ATTACK SCENARIOS
# ============================================================

attack_scenarios = sorted(
    scenario_values.unique()
)

print(
    f"Primary attack scenarios available: "
    f"{len(attack_scenarios)}"
)


# ============================================================
# SPLIT MASKS
# ============================================================

classifier_train_parts = []
calibration_parts = []
guard_parts = []


# ============================================================
# SCENARIO-BY-SCENARIO TEMPORAL SPLIT
# ============================================================

print("\n[2] Creating temporal scenario splits")

for scenario in attack_scenarios:

    scenario_mask = (
        df["scenario_id"].astype(str) == scenario
    )

    scenario_df = (
        df.loc[scenario_mask]
        .sort_values("timestamp")
        .copy()
    )

    # Released HAI labels are minute-level. Work at minute
    # boundaries rather than splitting arbitrary second rows.
    minutes = (
        scenario_df["timestamp"]
        .dt.floor("min")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    minute_count = len(minutes)

    if minute_count < 2:
        raise ValueError(
            f"Scenario {scenario} has fewer than 2 attack minutes"
        )

    calibration_minute = minutes[-1]

    if minute_count >= 3:

        guard_minute = minutes[-2]
        training_minutes = minutes[:-2]

        guard_mask = (
            scenario_df["timestamp"]
            .dt.floor("min")
            == guard_minute
        )

        train_mask = (
            scenario_df["timestamp"]
            .dt.floor("min")
            .isin(training_minutes)
        )

        calibration_mask = (
            scenario_df["timestamp"]
            .dt.floor("min")
            == calibration_minute
        )

        guard_parts.append(
            scenario_df.loc[guard_mask]
        )

        classifier_train_parts.append(
            scenario_df.loc[train_mask]
        )

        calibration_parts.append(
            scenario_df.loc[calibration_mask]
        )

    else:
        # Two-minute scenario:
        # minute 1 -> classifier training
        # minute 2 -> calibration
        #
        # No guard minute is possible without eliminating the
        # training or calibration positive example.

        training_minute = minutes[0]

        train_mask = (
            scenario_df["timestamp"]
            .dt.floor("min")
            == training_minute
        )

        calibration_mask = (
            scenario_df["timestamp"]
            .dt.floor("min")
            == calibration_minute
        )

        classifier_train_parts.append(
            scenario_df.loc[train_mask]
        )

        calibration_parts.append(
            scenario_df.loc[calibration_mask]
        )


# ============================================================
# BUILD ATTACK-SIDE SPLITS
# ============================================================

attack_train = pd.concat(
    classifier_train_parts,
    ignore_index=True,
)

attack_calibration = pd.concat(
    calibration_parts,
    ignore_index=True,
)

if guard_parts:
    attack_guard = pd.concat(
        guard_parts,
        ignore_index=True,
    )
else:
    attack_guard = pd.DataFrame(
        columns=df.columns
    )


# ============================================================
# NORMAL DATA
# ============================================================

print("\n[3] Handling normal observations")

normal_df = df[
    df["scenario_id"].isna()
].copy()

# Normal observations are not assigned to a scenario.
# Keep them available for both classifier training and
# calibration, but use disjoint temporal rows.
#
# The chronological split is performed globally so that
# calibration normal observations come from a later period
# than classifier-training normal observations.

normal_df = normal_df.sort_values("timestamp")

normal_cut = int(len(normal_df) * 0.80)

normal_train = normal_df.iloc[:normal_cut].copy()
normal_calibration = normal_df.iloc[normal_cut:].copy()


# ============================================================
# COMBINE
# ============================================================

classifier_train = pd.concat(
    [normal_train, attack_train],
    ignore_index=True,
).sort_values("timestamp")

calibration = pd.concat(
    [normal_calibration, attack_calibration],
    ignore_index=True,
).sort_values("timestamp")

guard = attack_guard.sort_values("timestamp")


# ============================================================
# TARGET VALIDATION
# ============================================================

print("\n[4] Validating attack-code coverage")

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

missing_train = set(target_columns) - train_codes
missing_calibration = set(target_columns) - calibration_codes

print(
    f"Classifier training coverage: "
    f"{len(train_codes)}/39"
)

print(
    f"Calibration coverage: "
    f"{len(calibration_codes)}/39"
)

if missing_train:
    print(
        "Missing from classifier training:",
        sorted(missing_train),
    )

if missing_calibration:
    print(
        "Missing from calibration:",
        sorted(missing_calibration),
    )


# ============================================================
# ROW COUNTS
# ============================================================

print("\n[5] Split sizes")

print(
    f"Classifier training: {len(classifier_train):,}"
)

print(
    f"Calibration:         {len(calibration):,}"
)

print(
    f"Guard:               {len(guard):,}"
)

print(
    f"Training attacks: "
    f"{int(classifier_train[target_columns].sum().sum()):,}"
)

print(
    f"Calibration attacks: "
    f"{int(calibration[target_columns].sum().sum()):,}"
)


# ============================================================
# TIMESTAMP OVERLAP CHECK
# ============================================================

train_timestamps = set(
    classifier_train["timestamp"]
)

calibration_timestamps = set(
    calibration["timestamp"]
)

if train_timestamps & calibration_timestamps:
    raise ValueError(
        "Timestamp overlap detected between classifier training "
        "and calibration data"
    )

print(
    "\nTimestamp separation: PASS"
)


# ============================================================
# SCENARIO VALIDATION
# ============================================================

calibration_attack_scenarios = set(
    calibration["scenario_id"]
    .dropna()
    .astype(str)
)

if calibration_attack_scenarios & FINAL_EVALUATION_SCENARIOS:
    raise ValueError(
        "Calibration contains final evaluation scenarios"
    )

print(
    "Final evaluation scenario leakage: PASS"
)


# ============================================================
# SAVE
# ============================================================

print("\n[6] Saving temporal calibration split")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

classifier_train.to_csv(
    TRAIN_OUTPUT,
    index=False,
)

calibration.to_csv(
    CALIBRATION_OUTPUT,
    index=False,
)

guard.to_csv(
    GUARD_OUTPUT,
    index=False,
)

print(f"Classifier training: {TRAIN_OUTPUT}")
print(f"Calibration:         {CALIBRATION_OUTPUT}")
print(f"Guard:               {GUARD_OUTPUT}")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 100)
print("STEP 6.16E.2 COMPLETE")
print("=" * 100)

print("\nNo classifier was retrained.")
print("No GPU model was modified.")
print("Final evaluation scenarios remain untouched.")