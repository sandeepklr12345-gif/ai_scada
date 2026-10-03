from pathlib import Path
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_model_input_acquisition_plan.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_raw_feature_mapping.csv"
)


# ============================================================
# CURRENT POSTGRESQL PARAMETER CATALOG
# ============================================================

POSTGRES_PARAMETERS = {
    "power_output": {
        "unit": "MW",
        "description": "Electrical power output",
    },
    "voltage": {
        "unit": "V",
        "description": "Electrical voltage",
    },
    "current": {
        "unit": "A",
        "description": "Electrical current",
    },
    "frequency": {
        "unit": "Hz",
        "description": "Electrical frequency",
    },
    "steam_pressure": {
        "unit": "MPa",
        "description": "Steam pressure",
    },
    "steam_temperature": {
        "unit": "C",
        "description": "Steam temperature",
    },
    "feedwater_flow": {
        "unit": "t/h",
        "description": "Feedwater flow rate",
    },
    "coal_flow": {
        "unit": "t/h",
        "description": "Coal/fuel flow rate",
    },
    "turbine_speed": {
        "unit": "RPM",
        "description": "Turbine rotational speed",
    },
    "vibration": {
        "unit": "unknown",
        "description": "Equipment vibration measurement",
    },
}


# ============================================================
# INITIAL CANDIDATE MAPPINGS
# ============================================================

CANDIDATE_MAPPINGS = {

    "Power output": (
        "power_output",
        "CANDIDATE_REQUIRES_VERIFICATION",
        "STRONG_CANDIDATE",
        "Same physical quantity and unit as PostgreSQL power_output."
    ),

    "Feed water flow rate": (
        "feedwater_flow",
        "CANDIDATE_REQUIRES_VERIFICATION",
        "STRONG_CANDIDATE",
        "Same physical quantity and unit as PostgreSQL feedwater_flow."
    ),

    "Feedwater pressure": (
        "steam_pressure",
        "CANDIDATE_REQUIRES_VERIFICATION",
        "LOW_CONFIDENCE",
        "PostgreSQL steam_pressure is generic and does not identify feedwater pressure."
    ),

    "Feedwater temperature": (
        "steam_temperature",
        "CANDIDATE_REQUIRES_VERIFICATION",
        "LOW_CONFIDENCE",
        "PostgreSQL steam_temperature is generic and does not identify feedwater temperature."
    ),

    "Fresh steam pressure": (
        "steam_pressure",
        "CANDIDATE_REQUIRES_VERIFICATION",
        "LOW_CONFIDENCE",
        "PostgreSQL steam_pressure is generic and does not identify fresh steam pressure."
    ),

    "A-side SH steam temperature": (
        "steam_temperature",
        "CANDIDATE_REQUIRES_VERIFICATION",
        "LOW_CONFIDENCE",
        "PostgreSQL steam_temperature does not distinguish the A-side SH steam measurement."
    ),

    "B-side SH steam temperature": (
        "steam_temperature",
        "CANDIDATE_REQUIRES_VERIFICATION",
        "LOW_CONFIDENCE",
        "PostgreSQL steam_temperature does not distinguish the B-side SH steam measurement."
    ),

    "Reheat steam flow rate": (
        "",
        "NO_CURRENT_SOURCE",
        "NOT_ESTABLISHED",
        "No dedicated reheat steam flow parameter exists in the current PostgreSQL catalog."
    ),

    "A-side flue gas oxygen content": (
        "",
        "NO_CURRENT_SOURCE",
        "NOT_ESTABLISHED",
        "No flue-gas oxygen parameter exists in the current PostgreSQL catalog."
    ),

    "B-side flue gas oxygen content": (
        "",
        "NO_CURRENT_SOURCE",
        "NOT_ESTABLISHED",
        "No flue-gas oxygen parameter exists in the current PostgreSQL catalog."
    ),
}

def normalize_feature_name(feature):
    """
    Convert the full model feature name into its
    physical signal name for mapping lookup.

    Example:
        'Power output\\n（MW）'
        -> 'Power output'
    """

    text = str(feature)

    # Normalize line breaks.
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Normalize whitespace.
    text = " ".join(text.split())

    # Remove Unicode/full-width unit suffix.
    if "（" in text:
        text = text.split("（", 1)[0].strip()

    # Remove ASCII unit suffix.
    elif "(" in text:
        text = text.split("(", 1)[0].strip()

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("600 MW RAW FEATURE MAPPING SPECIFICATION")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing acquisition plan:\n{INPUT_PATH}"
        )

    plan = pd.read_csv(INPUT_PATH)

    raw = plan[
        plan["feature_class"] == "RAW_SOURCE"
    ].copy()

    print(f"\nRAW_SOURCE features : {len(raw)}")

    rows = []

    for _, row in raw.iterrows():

        feature = row["feature"]

        # Extract unit from the feature name where possible.
        unit = ""

        if "（" in feature and "）" in feature:
            unit = feature.split("（")[-1].split("）")[0]

        elif "(" in feature and ")" in feature:
            unit = feature.split("(")[-1].split(")")[0]

        postgres_parameter = ""
        mapping_status = "NO_CURRENT_SOURCE"
        mapping_confidence = "NOT_ESTABLISHED"
        physical_reason = (
            "No current PostgreSQL parameter has been "
            "established for this feature."
        )

        mapping_key = normalize_feature_name(feature)

        if mapping_key in CANDIDATE_MAPPINGS:

            (
                postgres_parameter,
                mapping_status,
                mapping_confidence,
                physical_reason,
            ) = CANDIDATE_MAPPINGS[mapping_key]

        requires_new_parameter = (
            mapping_status == "NO_CURRENT_SOURCE"
        )

        rows.append({

            "feature": feature,

            "unit": unit,

            "postgres_parameter":
                postgres_parameter,

            "mapping_status":
                mapping_status,

            "mapping_confidence":
                mapping_confidence,

            "physical_reason":
                physical_reason,

            "requires_new_parameter":
                requires_new_parameter,

            "runtime_source":
                "SCADA_PARAMETER"
                if postgres_parameter
                else "NOT_AVAILABLE",

        })

    mapping = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert len(mapping) == 71

    assert mapping["feature"].is_unique

    # No mapping is allowed to be CONFIRMED yet.
    assert not (
        mapping["mapping_status"] == "CONFIRMED"
    ).any()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mapping.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nMAPPING STATUS")

    print(
        mapping["mapping_status"]
        .value_counts()
        .to_string()
    )

    print("\nPOSTGRESQL CANDIDATES")

    candidates = mapping[
        mapping["postgres_parameter"] != ""
    ]

    print(
        candidates[
            [
                "feature",
                "postgres_parameter",
                "mapping_confidence",
            ]
        ].to_string(index=False)
    )

    print("\nFEATURES WITHOUT CURRENT SOURCE")

    no_source = mapping[
        mapping["mapping_status"]
        == "NO_CURRENT_SOURCE"
    ]

    print(
        f"Count : {len(no_source)}"
    )

    print(
        f"\nOutput saved:\n{OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print(
        "600 MW RAW FEATURE MAPPING: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
