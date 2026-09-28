from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
)

TEST1_FILE = PROCESSED_DIR / "hai-test1_aligned.csv"
TEST2_FILE = PROCESSED_DIR / "hai-test2_aligned.csv"

STABILITY_FILE = FEATURE_DIR / "hai_2305_feature_stability.csv"

OUTPUT_FILE = FEATURE_DIR / "hai_2305_feature_roles.csv"


# ============================================================
# DOCUMENTED FEATURE ROLES
# ============================================================
#
# These roles are based on the official HAI 23.05
# technical documentation.
#
# Do NOT infer undocumented meanings from variable names.
# ============================================================

ROLE_MAP = {

    # --------------------------------------------------------
    # P1: Boiler / Water-treatment related process
    # --------------------------------------------------------

    "P1_FCV01D": "Valve Position Command",
    "P1_FCV01Z": "Valve Current Position",

    "P1_FCV02D": "Valve Position Command",
    "P1_FCV02Z": "Valve Current Position",

    "P1_FCV03D": "Valve Position Command",
    "P1_FCV03Z": "Valve Current Position",

    "P1_FT01": "Measured Flowrate",
    "P1_FT01Z": "Converted Flowrate",

    "P1_FT02": "Measured Flowrate",
    "P1_FT02Z": "Converted Flowrate",

    "P1_FT03": "Measured Flowrate",
    "P1_FT03Z": "Converted Flowrate",

    "P1_LCV01D": "Valve Position Command",
    "P1_LCV01Z": "Valve Current Position",

    "P1_LIT01": "Process Measurement",

    "P1_PCV01D": "Valve Position Command",
    "P1_PCV01Z": "Valve Current Position",

    "P1_PCV02D": "Valve Position Command",
    "P1_PCV02Z": "Valve Current Position",

    "P1_PIT01": "Process Measurement",
    "P1_PIT01_HH": "Process Threshold",

    "P1_PIT02": "Process Measurement",

    "P1_PP01AD": "Pump Start Command",
    "P1_PP01AR": "Pump Running State",

    "P1_PP01BD": "Pump Start Command",
    "P1_PP01BR": "Pump Running State",

    "P1_PP02D": "Pump Start Command",
    "P1_PP02R": "Pump Running State",

    "P1_PP04": "Control Output",
    "P1_PP04D": "Requires Documentation",
    "P1_PP04SP": "Setpoint",

    "P1_SOL01D": "Valve Open Command",
    "P1_SOL03D": "Valve Open Command",

    "P1_STSP": "Start/Stop Command",

    "P1_TIT01": "Process Measurement",
    "P1_TIT02": "Process Measurement",
    "P1_TIT03": "Process Measurement",


    # --------------------------------------------------------
    # P2: Turbine / speed control
    # --------------------------------------------------------

    "P2_24Vdc": "Electrical Measurement",

    "P2_ATSW_Lamp": "Status / Indicator",
    "P2_AutoGO": "Start Command",
    "P2_AutoSD": "Speed Demand",
    "P2_Emerg": "Emergency Status",

    "P2_MASW": "Mode Switch",
    "P2_MASW_Lamp": "Status / Indicator",

    "P2_ManualGO": "Start Command",
    "P2_ManualSD": "Speed Demand",

    "P2_OnOff": "On/Off Switch",

    "P2_RTR": "Trip Rate",

    "P2_SCO": "Controller Output",
    "P2_SCST": "Control Process Variable",

    "P2_SIT01": "Process Measurement",

    "P2_TripEx": "Trip Status",

    "P2_VIBTR01": "Vibration Measurement",
    "P2_VIBTR02": "Vibration Measurement",
    "P2_VIBTR03": "Vibration Measurement",
    "P2_VIBTR04": "Vibration Measurement",

    "P2_VT01": "Equipment Measurement",

    "P2_VTR01": "Vibration Limit",
    "P2_VTR02": "Vibration Limit",
    "P2_VTR03": "Vibration Limit",
    "P2_VTR04": "Vibration Limit",


    # --------------------------------------------------------
    # P3: Water treatment
    # --------------------------------------------------------

    "P3_FIT01": "Process Measurement",

    "P3_LCP01D": "Pump Speed Command",
    "P3_LCV01D": "Valve Position Command",

    "P3_LH01": "Level Setpoint",
    "P3_LIT01": "Process Measurement",
    "P3_LL01": "Level Setpoint",

    "P3_PIT01": "Process Measurement",


    # --------------------------------------------------------
    # P4: HIL / power generation
    # --------------------------------------------------------

    "P4_HT_FD": "Frequency Deviation",
    "P4_HT_PO": "Output Power",
    "P4_HT_PS": "Scheduled Power Demand",

    "P4_LD": "Total Electrical Load Demand",

    "P4_ST_FD": "Frequency Deviation",
    "P4_ST_GOV": "Gate Opening Rate",
    "P4_ST_LD": "Electrical Load Demand",
    "P4_ST_PO": "Output Power",
    "P4_ST_PS": "Scheduled Power Demand",

    "P4_ST_PT01": "Steam Pressure",
    "P4_ST_TT01": "Steam Temperature",
}


# ============================================================
# HAIEnd / INTERNAL CONTROL-LOGIC OUTPUTS
# ============================================================
#
# The dataset uses names such as:
# x1001_05_SETPOINT_OUT
# x1001_15_ASSIGN_OUT
#
# The official technical document identifies the underlying
# HAIEnd points as outputs of internal DCS functions.
#
# We therefore DO NOT infer their engineering meaning solely
# from the suffix used in our CSV column name.
# ============================================================

INTERNAL_OUTPUTS = {
    "x1001_05_SETPOINT_OUT",
    "x1001_15_ASSIGN_OUT",
    "x1002_07_SETPOINT_OUT",
    "x1002_08_SETPOINT_OUT",
    "x1003_10_SETPOINT_OUT",
    "x1003_18_SETPOINT_OUT",
    "x1003_24_SUM_OUT",
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HAI 23.05 DOCUMENT-BASED FEATURE ROLE CLASSIFICATION")
print("=" * 70)

test1 = pd.read_csv(TEST1_FILE)
test2 = pd.read_csv(TEST2_FILE)
stability = pd.read_csv(STABILITY_FILE)

features = [
    col
    for col in test1.columns
    if col not in ["timestamp", "label"]
]


# ============================================================
# VALIDATION
# ============================================================

print("\nValidating feature coverage...")

missing_roles = [
    feature
    for feature in features
    if feature not in ROLE_MAP
    and feature not in INTERNAL_OUTPUTS
]

if missing_roles:
    print("\nERROR: Features without documented role:")
    for feature in missing_roles:
        print(f"  {feature}")

    raise ValueError(
        "Feature-role mapping is incomplete."
    )

extra_roles = [
    feature
    for feature in ROLE_MAP
    if feature not in features
]

if extra_roles:
    print("\nWARNING: Role map contains features not present:")
    for feature in extra_roles:
        print(f"  {feature}")


# ============================================================
# BUILD ROLE TABLE
# ============================================================

rows = []

for feature in features:

    if feature in INTERNAL_OUTPUTS:
        role = "Internal Control-Logic Output"
    else:
        role = ROLE_MAP[feature]

    unique_values = test1[feature].nunique()

    if unique_values == 1:
        variability = "Constant"
    elif unique_values == 2:
        variability = "Two-State"
    else:
        variability = "Variable"

    rows.append({
        "feature": feature,
        "role": role,
        "variability": variability,
        "unique_values_test1": unique_values,
    })


result = pd.DataFrame(rows)


# ============================================================
# ADD STABILITY INFORMATION
# ============================================================

stability_columns = [
    "feature",
    "constant_test1",
    "constant_test2",
    "constant_both",
]

available_stability = [
    col
    for col in stability_columns
    if col in stability.columns
]

result = result.merge(
    stability[available_stability],
    on="feature",
    how="left"
)


# ============================================================
# ROLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DOCUMENT-BASED ROLE SUMMARY")
print("=" * 70)

print(
    result["role"]
    .value_counts()
    .to_string()
)


# ============================================================
# ALL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("ALL FEATURE CLASSIFICATIONS")
print("=" * 70)

print(
    result.to_string(index=False)
)


# ============================================================
# INTERNAL OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("INTERNAL CONTROL-LOGIC OUTPUTS")
print("=" * 70)

print(
    result[
        result["role"] == "Internal Control-Logic Output"
    ][
        ["feature", "role", "variability"]
    ].to_string(index=False)
)


# ============================================================
# CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CONSTANT IN BOTH TEST DATASETS")
print("=" * 70)

constant_both = result[
    result["constant_both"] == True
]

print(
    constant_both[
        ["feature", "role", "variability"]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DOCUMENT-BASED FEATURE ROLE CLASSIFICATION COMPLETE")
print("=" * 70)