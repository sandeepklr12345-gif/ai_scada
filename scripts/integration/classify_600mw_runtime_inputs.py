from pathlib import Path
import json
import re
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

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "600mw"
)

FORECAST_MANIFEST = (
    MODEL_DIR
    / "600mw_forecasting_feature_manifest.json"
)

RAW_SOURCE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "600mw"
    / "600 MW unit one-week operating data.xlsx"
)
CANDIDATE_DATASET = (
    FEATURE_DIR
    / "candidates"
    / "feature_set_D_full_candidate_pool.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "integration"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "600mw_runtime_input_classification.csv"
)


# ============================================================
# LOAD FINAL MODEL FEATURES
# ============================================================

def load_final_features():

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
            "Could not find feature list in "
            "600 MW model manifest."
        )

    return [str(feature) for feature in features]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(name):

    return (
        str(name)
        .strip()
        .lower()
        .replace("\n", "")
        .replace("\r", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# LAG DETECTION
# ============================================================

LAG_PATTERNS = [

    # Examples:
    # variable_lag_1
    # variable_lag1
    r"^(.*)_lag_?(\d+)$",

    # Examples:
    # variable_lag_2
    r"^(.*)_lag(\d+)$",

    # Examples:
    # variable_tminus_1
    # variable_tminus1
    r"^(.*)_tminus_?(\d+)$",

    # Examples:
    # variable_t_1
    r"^(.*)_t_?minus_?(\d+)$",

    # Examples:
    # variable_t-1
    r"^(.*)_t_(\d+)$",
]


def detect_lag_feature(feature):
    """
    Detect lag features of the form:

        <base_feature>_lag_1
        <base_feature>_lag_2
        <base_feature>_lag_5
        <base_feature>_lag_10

    The base feature may contain spaces, newlines,
    Unicode characters and measurement units.
    """

    feature_text = str(feature).strip()

    match = re.search(
        r"_lag_(\d+)\s*$",
        feature_text,
        flags=re.IGNORECASE
    )

    if match:

        lag_step = match.group(1)

        base_feature = feature_text[
            :match.start()
        ].strip()

        return (
            True,
            base_feature,
            lag_step
        )

    return (
        False,
        None,
        None
    )


# ============================================================
# BUILD FEATURE LINEAGE
# ============================================================

def classify_feature(
    feature,
    original_columns,
):

    normalized_original = {
        normalize(column): column
        for column in original_columns
    }

    normalized_feature = normalize(feature)

    # --------------------------------------------------------
    # 1. Time-derived feature
    # --------------------------------------------------------

    TIME_FEATURES = {
        "hour",
        "minute",
        "day_of_week",
        "day_of_month",
    }

    if normalized_feature in TIME_FEATURES:

        return {
            "feature_class": "TIME_DERIVED",
            "source_feature": "Time",
            "derivation":
                f"Derived from timestamp using "
                f"{normalized_feature}.",
            "lag_step": "",
            "runtime_requirement":
                "Current timestamp required.",
            "notes":
                "Generated online from the runtime timestamp."
        }

    # --------------------------------------------------------
    # 2. Lag-derived feature
    # --------------------------------------------------------

    is_lag, base_feature, lag_step = (
        detect_lag_feature(feature)
    )

    if is_lag:

        original_base = normalized_original.get(
            normalize(base_feature)
        )

        if original_base is not None:

            return {
                "feature_class": "LAG_DERIVED",
                "source_feature": original_base,
                "derivation":
                    f"Lag of source variable by "
                    f"{lag_step} step(s).",
                "lag_step": lag_step,
                "runtime_requirement":
                    "Current SCADA variable plus "
                    "historical runtime buffer required.",
                "notes":
                    "Must be generated online from "
                    "previous SCADA observations."
            }

        return {
            "feature_class": "LAG_DERIVED_REVIEW",
            "source_feature": base_feature,
            "derivation":
                f"Lag of {base_feature} by "
                f"{lag_step} step(s).",
            "lag_step": lag_step,
            "runtime_requirement":
                "Source variable must first be mapped.",
            "notes":
                "Lag pattern detected but base variable "
                "was not found in the raw 600 MW dataset."
        }

    # --------------------------------------------------------
    # 3. Exact raw source feature
    # --------------------------------------------------------

    if normalized_feature in normalized_original:

        return {
            "feature_class": "RAW_SOURCE",
            "source_feature": normalized_original[
                normalized_feature
            ],
            "derivation": "None",
            "lag_step": "",
            "runtime_requirement":
                "Current SCADA source variable required.",
            "notes":
                "Feature exists directly in the original "
                "600 MW raw source dataset."
        }

    # --------------------------------------------------------
    # 4. Explicit engineered feature
    # --------------------------------------------------------

    engineered_terms = [
        "rolling",
        "diff",
        "delta",
        "change",
        "mean",
        "std",
        "min",
        "max",
        "rate",
        "ratio",
        "avg",
    ]

    if any(
        term in normalized_feature
        for term in engineered_terms
    ):

        return {
            "feature_class": "ENGINEERED",
            "source_feature": "",
            "derivation":
                "Engineered feature pattern detected.",
            "lag_step": "",
            "runtime_requirement":
                "Feature-engineering logic required.",
            "notes":
                "Requires verification against the "
                "600 MW feature-generation pipeline."
        }

    # --------------------------------------------------------
    # 5. Unknown
    # --------------------------------------------------------

    return {
        "feature_class": "REQUIRES_REVIEW",
        "source_feature": "",
        "derivation": "",
        "lag_step": "",
        "runtime_requirement":
            "Runtime source/derivation not established.",
        "notes":
            "Feature could not be classified automatically."
    }


# ============================================================
# MAIN CLASSIFICATION
# ============================================================

def build_classification():

    if not RAW_SOURCE_DATASET.exists():
        raise FileNotFoundError(
            f"Raw 600 MW source dataset not found:\n"
            f"{RAW_SOURCE_DATASET}"
        )

    if not CANDIDATE_DATASET.exists():
        raise FileNotFoundError(
            f"Candidate dataset not found:\n"
            f"{CANDIDATE_DATASET}"
        )

    original_df = pd.read_excel(
    RAW_SOURCE_DATASET,
    nrows=2
)

    candidate_df = pd.read_csv(
        CANDIDATE_DATASET,
        nrows=2
    )

    original_columns = [
        column
        for column in original_df.columns
        if normalize(column) not in {
            "time",
            "timestamp"
        }
    ]

    candidate_columns = list(
        candidate_df.columns
    )

    final_features = load_final_features()

    rows = []

    for feature in final_features:

        result = classify_feature(
            feature,
            original_columns
        )

        rows.append({
            "model":
                "600MW_FORECASTING",

            "feature":
                feature,

            "feature_class":
                result["feature_class"],

            "source_feature":
                result["source_feature"],

            "derivation":
                result["derivation"],

            "lag_step":
                result["lag_step"],

            "runtime_requirement":
                result["runtime_requirement"],

            "notes":
                result["notes"],
        })

    return pd.DataFrame(rows), candidate_columns


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("600 MW FEATURE LINEAGE CLASSIFICATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df, candidate_columns = (
        build_classification()
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFINAL MODEL")

    print(
        f"Total final features : {len(df)}"
    )

    print("\nFEATURE LINEAGE")

    print(
        df["feature_class"]
        .value_counts()
        .to_string()
    )

    print("\nLAG FEATURES")

    lag_df = df[
        df["feature_class"]
        == "LAG_DERIVED"
    ]

    print(
        f"Lag-derived features : {len(lag_df)}"
    )

    if not lag_df.empty:

        print("\nLag distribution:")

        print(
            lag_df["lag_step"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print("\nRAW FEATURES")

    raw_df = df[
        df["feature_class"]
        == "RAW_SOURCE"
    ]

    print(
        f"Raw source features  : {len(raw_df)}"
    )

    print("\nENGINEERED FEATURES")

    engineered_df = df[
        df["feature_class"]
        == "ENGINEERED"
    ]

    print(
        f"Engineered features  : "
        f"{len(engineered_df)}"
    )

    print("\nREQUIRES REVIEW")

    review_df = df[
        df["feature_class"]
        .isin([
            "REQUIRES_REVIEW",
            "LAG_DERIVED_REVIEW"
        ])
    ]

    print(
        f"Review features      : "
        f"{len(review_df)}"
    )

    if not review_df.empty:

        print("\nFeatures requiring review:")

        for feature in review_df["feature"]:
            print(f"  {feature}")

    print(
        f"\nOutput saved:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print(
        "600 MW FEATURE LINEAGE CLASSIFICATION: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()