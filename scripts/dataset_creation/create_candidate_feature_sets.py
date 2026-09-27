import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_PATH = Path(
    "data/features/600mw/600mw_forecasting_clean.csv"
)

OUTPUT_DIR = Path(
    "data/features/600mw/candidates"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("8I - CANDIDATE FEATURE SET CREATION")
print("=" * 60)

print(f"\nInput shape: {df.shape}")


# --------------------------------------------------
# Target columns
# --------------------------------------------------

TARGET_COLUMNS = [
    "target_power_2min",
    "target_power_10min",
    "target_power_30min"
]


# --------------------------------------------------
# Time features
# --------------------------------------------------

TIME_FEATURES = [
    "hour",
    "minute",
    "day_of_week",
    "day_of_month"
]


# --------------------------------------------------
# Major process variables
# --------------------------------------------------

POWER_VARIABLES = [
    "Power output\n（MW）",
    "Plant auxiliary power\n（MW）",
    "Net power output\n（MW）"
]


ELECTRICAL_PROCESS_VARIABLES = [
    "Power output\n（MW）",
    "Plant auxiliary power\n（MW）",
    "Net power output\n（MW）",
    "Feed water flow rate\n（t/h）",
    "Feedwater pressure\n（MPa）",
    "Feedwater temperature\n（℃）"
]


THERMAL_PROCESS_VARIABLES = [
    "Feed water flow rate\n（t/h）",
    "Feedwater pressure\n（MPa）",
    "Feedwater temperature\n（℃）",
    "Fresh steam pressure\n（MPa）",
    "Drum pressure\n（MPa）",
    "A-side SH steam temperature\n（℃）",
    "B-side SH steam temperature\n（℃）",
    "Reheat steam flow rate\n（t/h）",
    "A-side flue gas oxygen content\n（%）",
    "B-side flue gas oxygen content\n（%）",
    "Boiler efficiency\n（%）",
    "unit efficiency\n（%）"
]


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def get_existing_columns(columns):
    """
    Keep only columns that actually exist in the dataset.
    """

    return [
        column
        for column in columns
        if column in df.columns
    ]


def add_lag_columns(base_columns):
    """
    Add all lag columns associated with selected variables.
    """

    selected = []

    for column in base_columns:

        selected.append(column)

        lag_columns = [
            dataset_column
            for dataset_column in df.columns
            if dataset_column.startswith(column + "_lag_")
        ]

        selected.extend(lag_columns)

    return selected


def remove_targets(columns):
    """
    Ensure target columns never become input features.
    """

    return [
        column
        for column in columns
        if column not in TARGET_COLUMNS
    ]


def save_feature_set(name, columns):

    columns = remove_targets(columns)

    # Remove duplicates while preserving order
    columns = list(dict.fromkeys(columns))

    # Always preserve Time for chronological validation
    # and future time-based train/validation splitting.
    if "Time" in df.columns:
        columns = [
            "Time"
        ] + [
            column
            for column in columns
            if column != "Time"
        ]

    output = df[columns + TARGET_COLUMNS].copy()

    output_path = OUTPUT_DIR / f"{name}.csv"

    output.to_csv(
        output_path,
        index=False
    )

    # Number of actual ML features
    ml_feature_columns = [
        column
        for column in columns
        if column != "Time"
    ]

    print(f"\n{name}")
    print(
        f"Feature columns: "
        f"{len(ml_feature_columns)}"
    )
    print(
        f"Total columns with targets: "
        f"{output.shape[1]}"
    )
    print(f"Saved: {output_path}")

    return columns


# --------------------------------------------------
# Set A: Power-history baseline
# --------------------------------------------------

set_a_base = get_existing_columns(
    POWER_VARIABLES
)

set_a = TIME_FEATURES + add_lag_columns(
    set_a_base
)

set_a = save_feature_set(
    "feature_set_A_power_history",
    set_a
)


# --------------------------------------------------
# Set B: Electrical/process core
# --------------------------------------------------

set_b_base = get_existing_columns(
    ELECTRICAL_PROCESS_VARIABLES
)

set_b = TIME_FEATURES + add_lag_columns(
    set_b_base
)

set_b = save_feature_set(
    "feature_set_B_process_core",
    set_b
)


# --------------------------------------------------
# Set C: Thermal/process
# --------------------------------------------------

set_c_base = get_existing_columns(
    POWER_VARIABLES + THERMAL_PROCESS_VARIABLES
)

set_c = TIME_FEATURES + add_lag_columns(
    set_c_base
)

set_c = save_feature_set(
    "feature_set_C_thermal_process",
    set_c
)


# --------------------------------------------------
# Set D: Full candidate pool
# --------------------------------------------------

set_d = [
    column
    for column in df.columns
    if column not in TARGET_COLUMNS
]

set_d = save_feature_set(
    "feature_set_D_full_candidate_pool",
    set_d
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("8I COMPLETED")
print("=" * 60)

print("\nCandidate feature sets created:")
print(f"A: {len(set_a)} features")
print(f"B: {len(set_b)} features")
print(f"C: {len(set_c)} features")
print(f"D: {len(set_d)} features")

print("\nTargets excluded from all feature sets:")
for target in TARGET_COLUMNS:
    print(f"  - {target}")

print("\nOutput directory:")
print(OUTPUT_DIR)