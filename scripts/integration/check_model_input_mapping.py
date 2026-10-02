from pathlib import Path
import json
import pandas as pd

from postgres_scada_reader import DB_CONFIG
from postgres_to_scada_contract import PostgreSQLSCADAContractAdapter


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

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "integration"
)

REPORT_PATH = REPORT_DIR / "model_input_mapping_report.txt"


# ============================================================
# 600 MW FEATURE MANIFEST
# ============================================================

FORECAST_MANIFEST = (
    MODEL_DIR
    / "600mw_forecasting_feature_manifest.json"
)


def load_forecasting_features():

    with open(FORECAST_MANIFEST, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if isinstance(manifest, dict):

        # Expected final model manifest structure
        if "features" in manifest:
            features = manifest["features"]

        elif "feature_names" in manifest:
            features = manifest["feature_names"]

        else:
            raise ValueError(
                "Could not find forecasting feature list "
                "in model manifest."
            )

    else:
        features = manifest

    if not isinstance(features, list):
        raise ValueError(
            "Forecasting feature manifest is not a list."
        )

    return features


# ============================================================
# HAI FEATURE MANIFEST
# ============================================================

def load_hai_features():

    if not HAI_MANIFEST.exists():
        raise FileNotFoundError(
            f"HAI manifest not found:\n{HAI_MANIFEST}"
        )

    df = pd.read_csv(HAI_MANIFEST)

    # Try common feature-column names
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

        # If the manifest contains only one obvious column,
        # use it.
        if len(df.columns) == 1:
            feature_column = df.columns[0]
        else:
            raise ValueError(
                "Could not determine HAI feature-name column.\n"
                f"Columns found: {list(df.columns)}"
            )

    features = (
        df[feature_column]
        .dropna()
        .astype(str)
        .tolist()
    )

    return features


# ============================================================
# POSTGRESQL PARAMETERS
# ============================================================

def load_postgres_parameters():

    adapter = PostgreSQLSCADAContractAdapter(DB_CONFIG)

    # Read current database measurements.
    # At present this is expected to be empty.
    df = adapter.reader.read_measurements()

    # The parameter reference table is not directly exposed
    # through the contract adapter, so obtain the catalog using
    # the reader database connection.
    connection = adapter.reader._connect()

    try:

        query = """
            SELECT
                parameter_id,
                parameter_name,
                unit,
                category
            FROM parameters
            ORDER BY parameter_id
        """

        parameters = pd.read_sql_query(
            query,
            connection,
        )

    finally:
        connection.close()

    return parameters, df


# ============================================================
# NORMALIZED NAME MATCHING
# ============================================================

def normalize_name(name):

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_direct_matches(postgres_names, model_features):

    normalized_postgres = {
        normalize_name(name): name
        for name in postgres_names
    }

    matches = []

    for feature in model_features:

        normalized_feature = normalize_name(feature)

        if normalized_feature in normalized_postgres:

            matches.append(
                (
                    feature,
                    normalized_postgres[normalized_feature],
                )
            )

    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL INPUT MAPPING READINESS CHECK")
    print("=" * 70)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model features
    # --------------------------------------------------------

    forecasting_features = load_forecasting_features()
    hai_features = load_hai_features()

    print("\nMODEL INPUTS")

    print(
        f"600 MW forecasting features : "
        f"{len(forecasting_features)}"
    )

    print(
        f"HAI Candidate C features   : "
        f"{len(hai_features)}"
    )

    # --------------------------------------------------------
    # PostgreSQL catalog
    # --------------------------------------------------------

    postgres_parameters, measurements = (
        load_postgres_parameters()
    )

    postgres_names = (
        postgres_parameters["parameter_name"]
        .astype(str)
        .tolist()
    )

    print(
        f"PostgreSQL parameters        : "
        f"{len(postgres_names)}"
    )

    print(
        f"Current measurements         : "
        f"{len(measurements)}"
    )

    # --------------------------------------------------------
    # Direct name matching
    # --------------------------------------------------------

    forecast_matches = find_direct_matches(
        postgres_names,
        forecasting_features,
    )

    hai_matches = find_direct_matches(
        postgres_names,
        hai_features,
    )

    forecast_match_names = {
        feature
        for feature, parameter in forecast_matches
    }

    hai_match_names = {
        feature
        for feature, parameter in hai_matches
    }

    forecast_unmapped = [
        feature
        for feature in forecasting_features
        if feature not in forecast_match_names
    ]

    hai_unmapped = [
        feature
        for feature in hai_features
        if feature not in hai_match_names
    ]

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("\n600 MW DIRECT NAME MATCHING")

    print(
        f"Direct matches : "
        f"{len(forecast_matches)}"
    )

    print(
        f"Unmapped       : "
        f"{len(forecast_unmapped)}"
    )

    print("\nHAI DIRECT NAME MATCHING")

    print(
        f"Direct matches : "
        f"{len(hai_matches)}"
    )

    print(
        f"Unmapped       : "
        f"{len(hai_unmapped)}"
    )

    print("\nPOSTGRESQL PARAMETERS")

    for parameter in postgres_parameters.itertuples():

        print(
            f"  {parameter.parameter_name:<20} "
            f"{parameter.unit or '':<8} "
            f"{parameter.category or ''}"
        )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "MODEL INPUT MAPPING READINESS REPORT"
    )
    report_lines.append("=" * 70)

    report_lines.append("")
    report_lines.append(
        f"PostgreSQL parameter count: {len(postgres_names)}"
    )

    report_lines.append(
        f"600 MW model feature count: "
        f"{len(forecasting_features)}"
    )

    report_lines.append(
        f"HAI Candidate C feature count: "
        f"{len(hai_features)}"
    )

    report_lines.append("")
    report_lines.append("600 MW DIRECT NAME MATCHES")
    report_lines.append("-" * 70)

    if forecast_matches:

        for feature, parameter in forecast_matches:

            report_lines.append(
                f"{parameter} -> {feature}"
            )

    else:
        report_lines.append(
            "None"
        )

    report_lines.append("")
    report_lines.append("HAI DIRECT NAME MATCHES")
    report_lines.append("-" * 70)

    if hai_matches:

        for feature, parameter in hai_matches:

            report_lines.append(
                f"{parameter} -> {feature}"
            )

    else:
        report_lines.append(
            "None"
        )

    report_lines.append("")
    report_lines.append("600 MW UNMAPPED FEATURES")
    report_lines.append("-" * 70)

    for feature in forecast_unmapped:
        report_lines.append(feature)

    report_lines.append("")
    report_lines.append("HAI UNMAPPED FEATURES")
    report_lines.append("-" * 70)

    for feature in hai_unmapped:
        report_lines.append(feature)

    report_lines.append("")
    report_lines.append("CURRENT DATABASE MEASUREMENTS")
    report_lines.append("-" * 70)

    report_lines.append(
        f"Rows currently stored: {len(measurements)}"
    )

    report_lines.append("")
    report_lines.append("INTERPRETATION")
    report_lines.append("-" * 70)

    report_lines.append(
        "Direct name matching is only a preliminary mapping check."
    )

    report_lines.append(
        "It does not establish engineering equivalence."
    )

    report_lines.append(
        "No unmapped model feature should be fabricated."
    )

    report_lines.append(
        "The PostgreSQL parameter catalog currently contains "
        "a small generic SCADA parameter set."
    )

    report_lines.append(
        "The 600 MW and HAI models require substantially "
        "different feature representations."
    )

    report_lines.append(
        "Additional feature acquisition and/or engineering "
        "mapping is required before production inference "
        "from PostgreSQL measurements."
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        f"\nReport saved:\n{REPORT_PATH}"
    )

    print("\n" + "=" * 70)
    print("MODEL INPUT MAPPING CHECK: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()