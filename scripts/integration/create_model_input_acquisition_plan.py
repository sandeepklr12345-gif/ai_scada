from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "600mw"
)

FORECAST_MANIFEST = (
    MODEL_DIR
    / "600mw_forecasting_feature_manifest.json"
)

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
    / "model_input_acquisition_plan.csv"
)


# ============================================================
# EXISTING POSTGRESQL PARAMETERS
# ============================================================

POSTGRES_PARAMETERS = {
    "power_output": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.power_output",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },

    "voltage": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.voltage",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },

    "current": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.current",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },

    "frequency": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.frequency",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },

    "steam_pressure": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.steam_pressure",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },

    "steam_temperature": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.steam_temperature",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },

    "feedwater_flow": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.feedwater_flow",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },

    "coal_flow": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.coal_flow",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },

    "turbine_speed": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.turbine_speed",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },

    "vibration": {
        "source_type": "DIRECT_DATABASE_PARAMETER",
        "source_name": "PostgreSQL parameters.vibration",
        "status": "AVAILABLE",
        "runtime_available": True,
        "notes": "Existing PostgreSQL SCADA parameter."
    },
}


# ============================================================
# LOAD 600 MW FEATURES
# ============================================================

def load_forecasting_features():

    with open(
        FORECAST_MANIFEST,
        "r",
        encoding="utf-8"
    ) as file:

        manifest = json.load(file)

    if "features" in manifest:
        features = manifest["features"]

    elif "feature_names" in manifest:
        features = manifest["feature_names"]

    else:
        raise ValueError(
            "600 MW manifest does not contain "
            "'features' or 'feature_names'."
        )

    return [str(feature) for feature in features]


# ============================================================
# LOAD HAI ORIGINAL FEATURES
# ============================================================

def load_hai_original_features():

    df = pd.read_csv(HAI_MANIFEST)

    print("\nHAI MANIFEST COLUMNS")

    for column in df.columns:
        print(f"  {column}")

    possible_columns = [
        "feature",
        "feature_name",
        "Feature",
        "Feature_Name",
    ]

    feature_column = None

    for column in possible_columns:

        if column in df.columns:
            feature_column = column
            break

    if feature_column is None:

        if len(df.columns) == 1:
            feature_column = df.columns[0]

        else:
            raise ValueError(
                "Could not determine HAI feature column."
            )

    all_features = (
        df[feature_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Candidate C contains:
    # 58 original + 30 abs_diff + 30 rolling_std.
    #
    # For acquisition planning, we only need the
    # original SCADA inputs.

    original_features = [
        feature
        for feature in all_features
        if "__abs_diff_1s" not in feature
        and "__rolling_std_5s" not in feature
        and "__diff_1s" not in feature
    ]

    return original_features


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(name):

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# BUILD PLAN
# ============================================================

def build_plan():

    forecasting_features = load_forecasting_features()
    hai_features = load_hai_original_features()

    postgres_names = set(
        normalize(name)
        for name in POSTGRES_PARAMETERS
    )

    rows = []

    # --------------------------------------------------------
    # 600 MW
    # --------------------------------------------------------

    for feature in forecasting_features:

        normalized_feature = normalize(feature)

        if normalized_feature in postgres_names:

            info = POSTGRES_PARAMETERS[
                normalized_feature
            ]

            rows.append({
                "model": "600MW_FORECASTING",
                "feature": feature,
                "source_type": info["source_type"],
                "source_name": info["source_name"],
                "mapping_status": "DIRECT_MATCH",
                "engineering_basis":
                    "Exact parameter-name match.",
                "runtime_available":
                    info["runtime_available"],
                "notes": info["notes"],
            })

        else:

            rows.append({
                "model": "600MW_FORECASTING",
                "feature": feature,
                "source_type": "NOT_YET_MAPPED",
                "source_name": "",
                "mapping_status": "REQUIRES_REVIEW",
                "engineering_basis":
                    "No exact PostgreSQL parameter-name match.",
                "runtime_available": False,
                "notes":
                    "Do not fabricate a mapping."
            })

    # --------------------------------------------------------
    # HAI
    # --------------------------------------------------------

    for feature in hai_features:

        normalized_feature = normalize(feature)

        if normalized_feature in postgres_names:

            info = POSTGRES_PARAMETERS[
                normalized_feature
            ]

            rows.append({
                "model": "HAI_CANDIDATE_C",
                "feature": feature,
                "source_type": info["source_type"],
                "source_name": info["source_name"],
                "mapping_status": "DIRECT_MATCH",
                "engineering_basis":
                    "Exact parameter-name match.",
                "runtime_available":
                    info["runtime_available"],
                "notes": info["notes"],
            })

        else:

            rows.append({
                "model": "HAI_CANDIDATE_C",
                "feature": feature,
                "source_type": "NOT_YET_MAPPED",
                "source_name": "",
                "mapping_status": "REQUIRES_REVIEW",
                "engineering_basis":
                    "No exact PostgreSQL parameter-name match.",
                "runtime_available": False,
                "notes":
                    "Do not fabricate a mapping."
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL INPUT ACQUISITION PLAN")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plan = build_plan()

    plan.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nPLAN SUMMARY")

    print(
        "\n600 MW FORECASTING"
    )

    forecast = plan[
        plan["model"] == "600MW_FORECASTING"
    ]

    print(
        f"Total features       : {len(forecast)}"
    )

    print(
        "Direct matches        : "
        f"{(forecast['mapping_status'] == 'DIRECT_MATCH').sum()}"
    )

    print(
        "Requires review       : "
        f"{(forecast['mapping_status'] == 'REQUIRES_REVIEW').sum()}"
    )

    print(
        "\nHAI CANDIDATE C"
    )

    hai = plan[
        plan["model"] == "HAI_CANDIDATE_C"
    ]

    print(
        f"Original inputs       : {len(hai)}"
    )

    print(
        "Direct matches        : "
        f"{(hai['mapping_status'] == 'DIRECT_MATCH').sum()}"
    )

    print(
        "Requires review       : "
        f"{(hai['mapping_status'] == 'REQUIRES_REVIEW').sum()}"
    )

    print(
        f"\nPlan saved:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("MODEL INPUT ACQUISITION PLAN: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()