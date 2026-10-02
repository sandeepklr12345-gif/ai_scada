from pathlib import Path
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HAI_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
    / "final_candidate"
    / "hai_2305_candidate_C_feature_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "integration"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_runtime_input_classification.csv"
)


# ============================================================
# DOCUMENTED HAI SIGNALS
#
# These classifications are based on the HAI technical
# documentation and the engineering-resolution work already
# completed for this project.
# ============================================================

DOCUMENTED_SIGNALS = {

    # --------------------------------------------------------
    # P1 Boiler Process
    # --------------------------------------------------------

    "P1_FCV01D": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Valve position command documented in HAI."
    ),

    "P1_FCV01Z": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Valve current position documented in HAI."
    ),

    "P1_FCV02D": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Valve position command documented in HAI."
    ),

    "P1_FCV02Z": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Valve current position documented in HAI."
    ),

    "P1_FCV03D": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Valve position command documented in HAI."
    ),

    "P1_FCV03Z": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Valve current position documented in HAI."
    ),

    "P1_FT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Measured flowrate documented in HAI."
    ),

    "P1_FT01Z": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Converted flowrate documented in HAI."
    ),

    "P1_FT02": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Measured flowrate documented in HAI."
    ),

    "P1_FT02Z": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Converted flowrate documented in HAI."
    ),

    "P1_FT03": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Measured flowrate documented in HAI."
    ),

    "P1_FT03Z": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Converted flowrate documented in HAI."
    ),

    "P1_LCV01D": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Level-control valve command documented in HAI."
    ),

    "P1_LCV01Z": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Level-control valve position documented in HAI."
    ),

    "P1_LIT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Level measurement documented in HAI."
    ),

    "P1_PCV01D": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Pressure-control valve command documented in HAI."
    ),

    "P1_PCV01Z": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Pressure-control valve position documented in HAI."
    ),

    "P1_PCV02D": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Pressure-control valve command documented in HAI."
    ),

    "P1_PCV02Z": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Pressure-control valve position documented in HAI."
    ),

    "P1_PIT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Outlet pressure documented in HAI."
    ),

    "P1_PIT02": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Pressure measurement documented in HAI."
    ),

    "P1_TIT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Temperature measurement documented in HAI."
    ),

    "P1_TIT02": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Temperature measurement documented in HAI."
    ),

    "P1_TIT03": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Temperature measurement documented in HAI."
    ),

    "P1_PP04": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Present in model-ready HAI feature set."
    ),

    "P1_PP04SP": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Present in model-ready HAI feature set."
    ),

    # --------------------------------------------------------
    # P2 Turbine Process
    # --------------------------------------------------------

    "P2_24Vdc": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "24Vdc signal documented in HAI."
    ),

    "P2_AutoSD": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Auto shutdown signal documented in HAI."
    ),

    "P2_ManualSD": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Manual shutdown signal documented in HAI."
    ),

    "P2_SCST": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine/process control signal documented in HAI."
    ),

    "P2_SIT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine signal documented in HAI."
    ),

    "P2_VIBTR01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine vibration signal documented in HAI."
    ),

    "P2_VIBTR02": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine vibration signal documented in HAI."
    ),

    "P2_VIBTR03": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine vibration signal documented in HAI."
    ),

    "P2_VIBTR04": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine vibration signal documented in HAI."
    ),

    "P2_VT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine process signal documented in HAI."
    ),

    "P2_SCO": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine control signal documented in HAI."
    ),

    "P2_ATSW_Lamp": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine control/status signal documented in HAI."
    ),

    "P2_AutoGO": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine operating signal documented in HAI."
    ),

    "P2_MASW": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine control/status signal documented in HAI."
    ),

    "P2_MASW_Lamp": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine control/status signal documented in HAI."
    ),

    "P2_ManualGO": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Turbine operating signal documented in HAI."
    ),

    # --------------------------------------------------------
    # P3 Water Treatment
    # --------------------------------------------------------

    "P3_FIT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Flow measurement from HAI P3 process."
    ),

    "P3_LCP01D": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Control command from HAI P3 process."
    ),

    "P3_LCV01D": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Valve/control command from HAI P3 process."
    ),

    "P3_LIT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Level measurement from HAI P3 process."
    ),

    "P3_PIT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Pressure measurement from HAI P3 process."
    ),

    # --------------------------------------------------------
    # P4 HIL Power System
    # --------------------------------------------------------

    "P4_HT_FD": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 power-system signal."
    ),

    "P4_HT_PS": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 power-system signal."
    ),

    "P4_HT_PO": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 power-system signal."
    ),

    "P4_LD": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "Electrical load signal documented in HAI."
    ),

    "P4_ST_FD": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 power-system signal."
    ),

    "P4_ST_PS": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 power-system signal."
    ),

    "P4_ST_GOV": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 governor signal."
    ),

    "P4_ST_LD": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 steam-turbine load signal."
    ),

    "P4_ST_PO": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 steam-turbine power output signal."
    ),

    "P4_ST_PT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 steam pressure signal."
    ),

    "P4_ST_TT01": (
        "DOCUMENTED_SCADA",
        "HAI_HISTORICAL_RUNTIME",
        False,
        "HAI P4 steam temperature signal."
    ),
}


# ============================================================
# LOAD AUTHORITATIVE 58 INPUTS
# ============================================================

def load_hai_features():

    if not HAI_MANIFEST.exists():
        raise FileNotFoundError(
            f"HAI manifest not found:\n{HAI_MANIFEST}"
        )

    df = pd.read_csv(HAI_MANIFEST)

    if "feature" not in df.columns:
        raise ValueError(
            "Expected 'feature' column in HAI manifest."
        )

    features = (
        df["feature"]
        .dropna()
        .astype(str)
        .tolist()
    )

    original_features = [
        feature
        for feature in features
        if "__abs_diff_1s" not in feature
        and "__rolling_std_5s" not in feature
        and "__diff_1s" not in feature
    ]

    return original_features


# ============================================================
# BUILD CLASSIFICATION
# ============================================================

def build_classification():

    features = load_hai_features()

    rows = []

    for feature in features:

        if feature in DOCUMENTED_SIGNALS:

            (
                documented_role,
                acquisition_class,
                hardware_candidate,
                notes,
            ) = DOCUMENTED_SIGNALS[feature]

            rows.append({
                "feature": feature,
                "documented_role": documented_role,
                "acquisition_class": acquisition_class,
                "historical_available": True,
                "hardware_candidate": hardware_candidate,
                "runtime_status": "HISTORICAL_READY",
                "notes": notes,
            })

        else:

            rows.append({
                "feature": feature,
                "documented_role": "REQUIRES_DOCUMENTATION",
                "acquisition_class": "REQUIRES_DOCUMENTATION",
                "historical_available": True,
                "hardware_candidate": False,
                "runtime_status": "DOCUMENTATION_REVIEW",
                "notes":
                    "Feature is in the frozen model input set "
                    "but its engineering role requires "
                    "additional documentation before assigning "
                    "a physical/runtime source.",
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HAI RUNTIME INPUT CLASSIFICATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = build_classification()

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCLASSIFICATION SUMMARY")

    print(
        f"Total HAI original inputs : {len(df)}"
    )

    print(
        "\nBy documented role:"
    )

    print(
        df["documented_role"]
        .value_counts()
        .to_string()
    )

    print(
        "\nBy acquisition class:"
    )

    print(
        df["acquisition_class"]
        .value_counts()
        .to_string()
    )

    print(
        "\nBy runtime status:"
    )

    print(
        df["runtime_status"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nOutput saved:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("HAI RUNTIME INPUT CLASSIFICATION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()