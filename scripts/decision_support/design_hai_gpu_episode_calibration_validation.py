from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test2_attack_classifier_temporal_dataset.csv"
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

FIT_OUTPUT = OUTPUT_DIR / "calibration_episode_fit.csv"
VALIDATION_OUTPUT = OUTPUT_DIR / "calibration_episode_validation.csv"
GUARD_OUTPUT = OUTPUT_DIR / "calibration_episode_guard.csv"


FINAL_EVALUATION_SCENARIOS = {
    "A201",
    "A203",
    "A209",
    "A213",
    "A214",
    "A235",
    "A236",
}


TARGET_COLUMNS_EXPECTED = 39


print("=" * 100)
print("HAI 23.05 PER-EPISODE CALIBRATION VALIDATION DESIGN")
print("=" * 100)


# ============================================================
# LOAD
# ============================================================

print("\n[1] Loading primary training data")

df = pd.read_csv(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Input shape: {df.shape}")


target_columns = [
    c for c in df.columns
    if c.startswith("AP") or c.startswith("AE")
]

if len(target_columns) != TARGET_COLUMNS_EXPECTED:
    raise ValueError(
        f"Expected {TARGET_COLUMNS_EXPECTED} targets, "
        f"found {len(target_columns)}"
    )


# ============================================================
# SPLIT EACH ATTACK SCENARIO
# ============================================================

fit_parts = []
validation_parts = []
guard_parts = []

attack_scenarios = sorted(
    df.loc[
        df["scenario_id"].notna(),
        "scenario_id",
    ].astype(str).unique()
)

print(
    f"Attack scenarios available: "
    f"{len(attack_scenarios)}"
)

for scenario in attack_scenarios:

    if scenario in FINAL_EVALUATION_SCENARIOS:
        continue

    scenario_df = (
        df[df["scenario_id"].astype(str) == scenario]
        .sort_values("timestamp")
        .copy()
    )

    minutes = (
        scenario_df["timestamp"]
        .dt.floor("min")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    n = len(minutes)

    # --------------------------------------------------------
    # Handle short attack episodes explicitly.
    #
    # 2-minute episode:
    #   minute 1 -> calibration fit
    #   minute 2 -> validation
    #
    # No guard minute is possible without eliminating one
    # of the two positive attack minutes.
    # --------------------------------------------------------

    if n == 2:

        fit_minutes = minutes[:1]
        guard_minute = None
        validation_minutes = minutes[1:]

    else:

        # ----------------------------------------------------
        # 3+ minute episode:
        #   early portion -> fit
        #   one minute    -> guard
        #   later portion -> validation
        # ----------------------------------------------------

        fit_count = max(1, n // 2)

        fit_count = min(
            fit_count,
            n - 2,
        )

        fit_minutes = minutes[:fit_count]
        guard_minute = minutes[fit_count]
        validation_minutes = minutes[fit_count + 1:]

        # Ensure at least one minute remains for guard and
        # at least one minute for validation.
        fit_count = min(
            fit_count,
            n - 2,
        )

        fit_minutes = minutes[:fit_count]
        guard_minute = minutes[fit_count]
        validation_minutes = minutes[fit_count + 1:]

    if not validation_minutes:
        raise ValueError(
            f"{scenario}: no validation minutes remain"
        )

    fit_mask = (
        scenario_df["timestamp"]
        .dt.floor("min")
        .isin(fit_minutes)
    )

    guard_mask = (
        scenario_df["timestamp"]
        .dt.floor("min")
        == guard_minute
    )

    validation_mask = (
        scenario_df["timestamp"]
        .dt.floor("min")
        .isin(validation_minutes)
    )

    fit_parts.append(
        scenario_df.loc[fit_mask]
    )

    guard_parts.append(
        scenario_df.loc[guard_mask]
    )

    validation_parts.append(
        scenario_df.loc[validation_mask]
    )


# ============================================================
# NORMAL OBSERVATIONS
# ============================================================

print("\n[2] Splitting normal observations")

normal_df = (
    df[df["scenario_id"].isna()]
    .sort_values("timestamp")
    .copy()
)

normal_cut = int(len(normal_df) * 0.50)

normal_fit = normal_df.iloc[:normal_cut].copy()
normal_validation = normal_df.iloc[normal_cut:].copy()


# ============================================================
# COMBINE
# ============================================================

fit_df = pd.concat(
    [
        normal_fit,
        *fit_parts,
    ],
    ignore_index=True,
).sort_values("timestamp")

validation_df = pd.concat(
    [
        normal_validation,
        *validation_parts,
    ],
    ignore_index=True,
).sort_values("timestamp")

guard_df = pd.concat(
    guard_parts,
    ignore_index=True,
).sort_values("timestamp")


# ============================================================
# COVERAGE
# ============================================================

print("\n[3] Checking mechanism coverage")

fit_codes = {
    c for c in target_columns
    if fit_df[c].sum() > 0
}

validation_codes = {
    c for c in target_columns
    if validation_df[c].sum() > 0
}

print(
    f"Fit attack-code coverage: "
    f"{len(fit_codes)}/39"
)

print(
    f"Validation attack-code coverage: "
    f"{len(validation_codes)}/39"
)

missing_fit = set(target_columns) - fit_codes
missing_validation = set(target_columns) - validation_codes

if missing_fit:
    print(
        "Missing from fit:",
        sorted(missing_fit),
    )

if missing_validation:
    print(
        "Missing from validation:",
        sorted(missing_validation),
    )


# ============================================================
# TIMESTAMP DISJOINTNESS
# ============================================================

fit_timestamps = set(fit_df["timestamp"])
validation_timestamps = set(
    validation_df["timestamp"]
)
guard_timestamps = set(guard_df["timestamp"])

if fit_timestamps & validation_timestamps:
    raise ValueError(
        "Fit/validation timestamp overlap"
    )

if fit_timestamps & guard_timestamps:
    raise ValueError(
        "Fit/guard timestamp overlap"
    )

if validation_timestamps & guard_timestamps:
    raise ValueError(
        "Validation/guard timestamp overlap"
    )

print("\nTimestamp separation: PASS")


# ============================================================
# FINAL EVALUATION LEAKAGE
# ============================================================

for name, frame in [
    ("fit", fit_df),
    ("validation", validation_df),
    ("guard", guard_df),
]:

    scenarios = set(
        frame["scenario_id"]
        .dropna()
        .astype(str)
    )

    overlap = scenarios & FINAL_EVALUATION_SCENARIOS

    if overlap:
        raise ValueError(
            f"{name} contains final evaluation scenarios: "
            f"{sorted(overlap)}"
        )

print("Final evaluation scenario leakage: PASS")


# ============================================================
# SUMMARY
# ============================================================

print("\n[4] Split summary")

for name, frame in [
    ("Fit", fit_df),
    ("Validation", validation_df),
    ("Guard", guard_df),
]:

    attack_labels = int(
        frame[target_columns].sum().sum()
    )

    attack_rows = int(
        (frame[target_columns].sum(axis=1) > 0).sum()
    )

    print(
        f"{name:12s}: "
        f"rows={len(frame):,} | "
        f"attack_rows={attack_rows:,} | "
        f"attack_labels={attack_labels:,}"
    )


# ============================================================
# SAVE
# ============================================================

print("\n[5] Saving episode-level calibration split")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

fit_df.to_csv(
    FIT_OUTPUT,
    index=False,
)

validation_df.to_csv(
    VALIDATION_OUTPUT,
    index=False,
)

guard_df.to_csv(
    GUARD_OUTPUT,
    index=False,
)

print(f"Fit:        {FIT_OUTPUT}")
print(f"Validation: {VALIDATION_OUTPUT}")
print(f"Guard:      {GUARD_OUTPUT}")


print("\n" + "=" * 100)
print("STEP 6.16E.5 DESIGN COMPLETE")
print("=" * 100)

print("\nNo model was trained.")
print("No GPU model was modified.")
print("Final evaluation scenarios remain untouched.")