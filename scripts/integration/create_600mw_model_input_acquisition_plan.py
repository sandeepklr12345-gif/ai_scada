from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "forecasting" / "600mw"
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "600mw"
INTEGRATION_DIR = PROJECT_ROOT / "data" / "integration"

MANIFEST_PATH = (
    MODEL_DIR /
    "600mw_forecasting_feature_manifest.json"
)

CLASSIFICATION_PATH = (
    INTEGRATION_DIR /
    "600mw_runtime_input_classification.csv"
)

OUTPUT_PATH = (
    INTEGRATION_DIR /
    "600mw_model_input_acquisition_plan.csv"
)


# ============================================================
# LOAD MODEL MANIFEST
# ============================================================

def load_manifest_features():

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    if "features" in manifest:
        features = manifest["features"]

    elif "feature_names" in manifest:
        features = manifest["feature_names"]

    else:
        raise ValueError(
            "Could not find feature list in model manifest."
        )

    return [str(x) for x in features]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("600 MW MODEL INPUT ACQUISITION PLAN")
    print("=" * 70)

    INTEGRATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load authoritative final model feature list
    # --------------------------------------------------------

    final_features = load_manifest_features()

    print(
        f"\nFinal model features : {len(final_features)}"
    )

    # --------------------------------------------------------
    # Load previous lineage classification
    # --------------------------------------------------------

    if not CLASSIFICATION_PATH.exists():

        raise FileNotFoundError(
            f"Missing classification file:\n"
            f"{CLASSIFICATION_PATH}"
        )

    classification = pd.read_csv(
        CLASSIFICATION_PATH
    )

    required_column = "feature"

    if required_column not in classification.columns:

        raise ValueError(
            "Classification file does not contain "
            "'feature' column."
        )

    classification_map = (
        classification
        .set_index("feature")
        .to_dict("index")
    )

    # --------------------------------------------------------
    # Build acquisition plan
    # --------------------------------------------------------

    rows = []

    for feature in final_features:

        info = classification_map.get(
            feature,
            {}
        )

        feature_class = info.get(
            "feature_class",
            "REQUIRES_REVIEW"
        )

        source_feature = info.get(
            "source_feature",
            ""
        )

        derivation = info.get(
            "derivation",
            ""
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Historical presence is NOT runtime availability.
        # ----------------------------------------------------

        if feature_class == "RAW_SOURCE":
            runtime_source = "NOT_CONFIRMED"
            acquisition_method = "Requires runtime SCADA source mapping"
            runtime_status = "HISTORICAL_ONLY_UNTIL_RUNTIME_SOURCE_MAPPED"
            mapping_confidence = "NOT_ESTABLISHED"
            notes = (
                "Historical feature availability does not establish "
                "live SCADA availability."
            )

        elif feature_class == "TIME_DERIVED":
            runtime_source = "RUNTIME_TIMESTAMP"
            acquisition_method = "Derive from runtime timestamp"
            runtime_status = "REQUIRES_RUNTIME_FEATURE_BUILDER"
            mapping_confidence = "NOT_APPLICABLE"
            notes = (
                "Derived directly from the runtime timestamp. "
                "No PostgreSQL parameter mapping is required."
            )

        elif feature_class == "LAG_DERIVED":
            runtime_source = "RUNTIME_SOURCE_PLUS_HISTORY_BUFFER"
            acquisition_method = "Generate from runtime source and historical buffer"
            runtime_status = "REQUIRES_RUNTIME_FEATURE_BUILDER"
            mapping_confidence = "NOT_ESTABLISHED"
            notes = (
                "Historical feature availability does not establish "
                "live SCADA availability."
            )

        else:

            runtime_source = "UNKNOWN"

            runtime_status = "REQUIRES_REVIEW"

            acquisition_method = (
                "Runtime source must be established"
            )

        rows.append({

            "model":
                "600MW_FORECASTING",

            "feature":
                feature,

            "feature_class":
                feature_class,

            "historical_source_feature":
                source_feature,

            "historical_derivation":
                derivation,

            "runtime_source":
                runtime_source,

            "acquisition_method":
                acquisition_method,

            "postgresql_parameter":
                "",

            "runtime_status":
                runtime_status,

            "mapping_confidence":
                mapping_confidence,

            "notes":
                notes
        })


    plan = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert len(plan) == len(final_features)

    assert (
        plan["feature"].tolist()
        == final_features
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plan.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nFEATURE CLASSIFICATION")

    print(
        plan["feature_class"]
        .value_counts()
        .to_string()
    )

    print("\nRUNTIME STATUS")

    print(
        plan["runtime_status"]
        .value_counts()
        .to_string()
    )

    print("\nPOSTGRESQL MAPPING")

    print(
        "Confirmed PostgreSQL mappings : 0"
    )

    print(
        "Reason: runtime parameter-to-feature "
        "mapping has not yet been established."
    )

    print(
        f"\nOutput saved:\n{OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print(
        "600 MW MODEL INPUT ACQUISITION PLAN: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
