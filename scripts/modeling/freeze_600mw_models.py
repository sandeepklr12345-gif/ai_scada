"""
600 MW FINAL MODEL FREEZE + INFERENCE ENGINE

Freezes the validated leakage-safe 600 MW forecasting design.

Final design:
    Feature Set D
    119 leakage-safe numeric features
    Linear Regression
    Targets:
        target_power_2min
        target_power_10min
        target_power_30min

This script:
    1. Rebuilds the authoritative 119-feature manifest.
    2. Trains one final Linear Regression model per horizon
       on all valid rows.
    3. Saves final .joblib artifacts.
    4. Saves model metadata.
    5. Creates a reusable inference engine.
    6. Runs a self-test against known rows.

It does NOT modify raw or processed datasets.
It does NOT overwrite the previous benchmark artifacts.
"""

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT /
    "data/features/600mw/candidates/"
    "feature_set_D_full_candidate_pool.csv"
)

MODEL_DIR = (
    ROOT /
    "models/forecasting/600mw"
)

INFERENCE_DIR = (
    ROOT /
    "scripts/inference"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INFERENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGETS = [
    "target_power_2min",
    "target_power_10min",
    "target_power_30min"
]

FLAGGED_PATTERNS = {
    "Plant auxiliary power": [
        r"auxiliary.*power",
        r"aux.*power",
        r"plant.*auxiliary"
    ],
    "Net power output": [
        r"net.*power.*output",
        r"net.*output"
    ],
    "Boiler efficiency": [
        r"boiler.*efficiency"
    ],
    "Standard coal consumption rate": [
        r"standard.*coal.*consumption",
        r"coal.*consumption.*rate"
    ],
    "Turbine heat rate": [
        r"turbine.*heat.*rate"
    ],
    "Unit efficiency": [
        r"unit.*efficiency"
    ],
}


def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(name).lower()
    ).strip()


def find_flagged_columns(columns):

    matched = set()

    for column in columns:

        clean = normalize_name(column)

        for patterns in FLAGGED_PATTERNS.values():

            for pattern in patterns:

                if re.search(pattern, clean):
                    matched.add(column)
                    break

    return matched


# ============================================================
# LOAD AUTHORITATIVE FEATURE SET D
# ============================================================

print("")
print("=" * 80)
print("600 MW FINAL MODEL FREEZE")
print("=" * 80)
print("")

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Feature Set D not found:\n{FEATURE_FILE}"
    )

df = pd.read_csv(
    FEATURE_FILE
)

print(
    f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns"
)


# ============================================================
# TIMESTAMP
# ============================================================

if "Time" in df.columns:
    timestamp_column = "Time"
elif "timestamp" in df.columns:
    timestamp_column = "timestamp"
else:
    timestamp_column = None

if timestamp_column:
    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )
    df = df.sort_values(
        timestamp_column
    ).reset_index(drop=True)


# ============================================================
# BUILD LEAKAGE-SAFE FEATURE LIST
# ============================================================

flagged_columns = find_flagged_columns(
    df.columns
)

excluded = set(TARGETS)

if timestamp_column:
    excluded.add(timestamp_column)

excluded.update(flagged_columns)

feature_columns = [
    column
    for column in df.columns
    if column not in excluded
    and pd.api.types.is_numeric_dtype(
        df[column]
    )
]

print(
    f"Excluded leakage-review columns: "
    f"{len(flagged_columns)}"
)

print(
    f"Final feature count: {len(feature_columns)}"
)

if len(feature_columns) != 119:

    raise RuntimeError(
        "Expected exactly 119 leakage-safe features, "
        f"but found {len(feature_columns)}."
    )


# ============================================================
# SAVE AUTHORITATIVE FEATURE MANIFEST
# ============================================================

manifest_path = (
    MODEL_DIR /
    "600mw_forecasting_feature_manifest.json"
)

feature_manifest = {
    "dataset": "600 MW Unit One-Week Operating Data",
    "feature_set": "D",
    "feature_count": len(feature_columns),
    "features": feature_columns,
    "excluded_leakage_review_columns": sorted(
        flagged_columns
    ),
    "timestamp_column": timestamp_column,
    "targets": TARGETS,
    "model_type": "LinearRegression",
    "validation": {
        "split": "chronological",
        "temporal_robustness": "3 chronological folds",
        "feature_count": 119
    }
}

manifest_path.write_text(
    json.dumps(
        feature_manifest,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# TRAIN FINAL MODELS
# ============================================================

model_records = {}

for target in TARGETS:

    print("")
    print(
        f"Training final model: {target}"
    )

    training = (
        df[
            feature_columns + [target]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    X = training[
        feature_columns
    ]

    y = training[
        target
    ]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    horizon = target.replace(
        "target_power_",
        ""
    )

    model_path = (
        MODEL_DIR /
        f"600mw_{horizon}_final.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    model_records[target] = {
        "model_path": str(
            model_path.relative_to(ROOT)
        ),
        "training_rows": len(training),
        "feature_count": len(feature_columns),
        "model_type": "LinearRegression"
    }

    print(
        f"  Rows: {len(training)}"
    )

    print(
        f"  Features: {len(feature_columns)}"
    )

    print(
        f"  Saved: {model_path}"
    )


# ============================================================
# MODEL METADATA
# ============================================================

metadata = {
    "dataset": "600 MW Unit One-Week Operating Data",
    "role": "Primary power-output forecasting / load-estimation component",
    "model_family": "LinearRegression",
    "feature_set": "D leakage-safe",
    "feature_count": 119,
    "targets": TARGETS,
    "models": model_records,
    "validation_summary": {
        "2_min": {
            "temporal_robustness_mean_r2": 0.998253,
            "temporal_robustness_mean_rmse": 3.639241
        },
        "10_min": {
            "temporal_robustness_mean_r2": 0.974143,
            "temporal_robustness_mean_rmse": 12.598730
        },
        "30_min": {
            "temporal_robustness_mean_r2": 0.814217,
            "temporal_robustness_mean_rmse": 34.491407
        }
    },
    "status": "FROZEN FOR INTEGRATION",
    "note": (
        "Validation statistics are from chronological holdout folds. "
        "Final models are retrained on all valid rows using the "
        "validated 119-feature schema."
    )
}

metadata_path = (
    MODEL_DIR /
    "600mw_forecasting_model_metadata.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# CREATE REUSABLE INFERENCE ENGINE
# ============================================================

engine_code = r'''
"""
600 MW Forecasting Inference Engine

Loads the frozen 600 MW forecasting models and the authoritative
119-feature manifest.

Expected input:
    pandas DataFrame containing the 119 model features.

Output:
    pandas DataFrame containing:
        power_2min
        power_10min
        power_30min
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


class Forecasting600MW:

    def __init__(self, model_dir=None):

        # ----------------------------------------------------
        # MODEL DIRECTORY
        # ----------------------------------------------------

        if model_dir is None:

            model_dir = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "forecasting"
                / "600mw"
            )

        self.model_dir = Path(model_dir)

        # ----------------------------------------------------
        # LOAD FEATURE MANIFEST
        # ----------------------------------------------------

        manifest_path = (
            self.model_dir
            / "600mw_forecasting_feature_manifest.json"
        )

        if not manifest_path.exists():

            raise FileNotFoundError(
                "600 MW feature manifest not found:\n"
                f"{manifest_path}"
            )

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.manifest = json.load(file)

        # ----------------------------------------------------
        # LOAD MODEL METADATA
        # ----------------------------------------------------

        metadata_path = (
            self.model_dir
            / "600mw_forecasting_model_metadata.json"
        )

        if not metadata_path.exists():

            raise FileNotFoundError(
                "600 MW model metadata not found:\n"
                f"{metadata_path}"
            )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        # ----------------------------------------------------
        # FEATURE SCHEMA
        # ----------------------------------------------------

        self.features = self.manifest[
            "features"
        ]

        expected_count = self.manifest[
            "feature_count"
        ]

        if expected_count != 119:

            raise ValueError(
                "Invalid 600 MW manifest. "
                f"Expected 119 features, got {expected_count}."
            )

        if len(self.features) != 119:

            raise ValueError(
                "Feature manifest contains an unexpected "
                f"number of features: {len(self.features)}"
            )

        # ----------------------------------------------------
        # LOAD FINAL MODELS
        # ----------------------------------------------------

        self.models = {}

        for target in self.manifest[
            "targets"
        ]:

            horizon = target.replace(
                "target_power_",
                ""
            )

            model_path = (
                self.model_dir
                / f"600mw_{horizon}_final.joblib"
            )

            if not model_path.exists():

                raise FileNotFoundError(
                    "600 MW model not found:\n"
                    f"{model_path}"
                )

            self.models[target] = joblib.load(
                model_path
            )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def validate_input(
        self,
        data
    ):

        if not isinstance(
            data,
            pd.DataFrame
        ):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if data.empty:

            raise ValueError(
                "Input DataFrame is empty."
            )

        # ----------------------------------------------------
        # CHECK REQUIRED FEATURES
        # ----------------------------------------------------

        missing = [
            feature
            for feature in self.features
            if feature not in data.columns
        ]

        if missing:

            raise ValueError(
                "Missing required 600 MW features:\n"
                + "\n".join(missing)
            )

        # ----------------------------------------------------
        # CHECK NUMERIC FEATURES
        # ----------------------------------------------------

        non_numeric = [
            feature
            for feature in self.features
            if not pd.api.types.is_numeric_dtype(
                data[feature]
            )
        ]

        if non_numeric:

            raise TypeError(
                "The following required features "
                "are not numeric:\n"
                + "\n".join(non_numeric)
            )

        # ----------------------------------------------------
        # CHECK MISSING VALUES
        # ----------------------------------------------------

        missing_values = (
            data[
                self.features
            ]
            .isna()
            .sum()
            .sum()
        )

        if missing_values > 0:

            raise ValueError(
                "Input contains "
                f"{missing_values} missing feature values."
            )

        # ----------------------------------------------------
        # CHECK INFINITE / INVALID VALUES
        # ----------------------------------------------------

        numeric_block = data[
            self.features
        ]

        if not np.isfinite(
            numeric_block.to_numpy(dtype=float)
        ).all():

            raise ValueError(
                "Input contains infinite or invalid "
                "numeric feature values."
            )

        return True

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        data
    ):

        self.validate_input(
            data
        )

        X = data[
            self.features
        ].copy()

        predictions = pd.DataFrame(
            index=data.index
        )

        # ----------------------------------------------------
        # 2-MINUTE FORECAST
        # ----------------------------------------------------

        predictions[
            "power_2min"
        ] = self.models[
            "target_power_2min"
        ].predict(X)

        # ----------------------------------------------------
        # 10-MINUTE FORECAST
        # ----------------------------------------------------

        predictions[
            "power_10min"
        ] = self.models[
            "target_power_10min"
        ].predict(X)

        # ----------------------------------------------------
        # 30-MINUTE FORECAST
        # ----------------------------------------------------

        predictions[
            "power_30min"
        ] = self.models[
            "target_power_30min"
        ].predict(X)

        return predictions

    # ========================================================
    # SINGLE-ROW PREDICTION
    # ========================================================

    def predict_dict(
        self,
        data
    ):

        if len(data) != 1:

            raise ValueError(
                "predict_dict() expects exactly one row."
            )

        result = self.predict(
            data
        )

        row = result.iloc[0]

        return {
            "power_2min": float(
                row["power_2min"]
            ),
            "power_10min": float(
                row["power_10min"]
            ),
            "power_30min": float(
                row["power_30min"]
            )
        }
'''


# ============================================================
# SAVE INFERENCE ENGINE
# ============================================================

engine_path = (
    INFERENCE_DIR /
    "forecasting_600mw.py"
)

engine_path.write_text(
    engine_code.strip() + "\n",
    encoding="utf-8"
)

print("")
print(
    f"Inference engine created: {engine_path}"
)


# ============================================================
# INFERENCE ENGINE SELF-TEST
# ============================================================

print("")
print("=" * 80)
print("RUNNING 600 MW INFERENCE ENGINE SELF-TEST")
print("=" * 80)
print("")


# ------------------------------------------------------------
# Dynamically import the newly created inference engine
# ------------------------------------------------------------

import importlib.util

spec = importlib.util.spec_from_file_location(
    "forecasting_600mw",
    engine_path
)

if spec is None or spec.loader is None:

    raise RuntimeError(
        "Could not load the generated inference engine."
    )

module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    module
)


# ------------------------------------------------------------
# Create inference engine
# ------------------------------------------------------------

engine = module.Forecasting600MW(
    MODEL_DIR
)

print(
    f"Loaded feature count: "
    f"{len(engine.features)}"
)

print(
    f"Loaded models: "
    f"{len(engine.models)}"
)


# ------------------------------------------------------------
# Prepare one valid row
# ------------------------------------------------------------

test_input = (
    df[
        feature_columns
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna()
    .head(1)
)


if test_input.empty:

    raise RuntimeError(
        "Could not find a valid row for "
        "the inference self-test."
    )


# ------------------------------------------------------------
# Validate input
# ------------------------------------------------------------

engine.validate_input(
    test_input
)

print(
    "Input validation: PASS"
)


# ------------------------------------------------------------
# Run prediction
# ------------------------------------------------------------

result = engine.predict(
    test_input
)


# ------------------------------------------------------------
# Verify output schema
# ------------------------------------------------------------

expected_outputs = [
    "power_2min",
    "power_10min",
    "power_30min"
]

actual_outputs = list(
    result.columns
)

if actual_outputs != expected_outputs:

    raise RuntimeError(
        "Unexpected inference output schema.\n"
        f"Expected: {expected_outputs}\n"
        f"Actual:   {actual_outputs}"
    )


# ------------------------------------------------------------
# Verify prediction values
# ------------------------------------------------------------

if result.isna().any().any():

    raise RuntimeError(
        "Inference produced NaN predictions."
    )

if not np.isfinite(
    result.to_numpy()
).all():

    raise RuntimeError(
        "Inference produced infinite predictions."
    )


# ------------------------------------------------------------
# Print prediction
# ------------------------------------------------------------

print("")
print("Inference result:")

print(
    f"  power_2min  = "
    f"{result.iloc[0]['power_2min']:.6f} MW"
)

print(
    f"  power_10min = "
    f"{result.iloc[0]['power_10min']:.6f} MW"
)

print(
    f"  power_30min = "
    f"{result.iloc[0]['power_30min']:.6f} MW"
)


# ------------------------------------------------------------
# Test predict_dict()
# ------------------------------------------------------------

dictionary_result = engine.predict_dict(
    test_input
)

if set(dictionary_result.keys()) != set(
    expected_outputs
):

    raise RuntimeError(
        "predict_dict() returned an unexpected schema."
    )

print("")
print(
    "Single-row dictionary interface: PASS"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("")
print("=" * 80)
print("600 MW FINAL MODEL VALIDATION")
print("=" * 80)

print("")
print(
    f"Feature count       : {len(engine.features)}"
)

print(
    f"Model count         : {len(engine.models)}"
)

print(
    f"2-min model         : LOADED"
)

print(
    f"10-min model        : LOADED"
)

print(
    f"30-min model        : LOADED"
)

print(
    f"Input validation    : PASS"
)

print(
    f"Prediction schema   : PASS"
)

print(
    f"Prediction values   : PASS"
)

print(
    f"Dictionary API      : PASS"
)

print("")
print("=" * 80)
print("600 MW MODEL FREEZE COMPLETE")
print("=" * 80)

print("")
print("Final artifacts:")

print(
    f"  Models directory : {MODEL_DIR}"
)

print(
    f"  Feature manifest : {manifest_path}"
)

print(
    f"  Model metadata   : {metadata_path}"
)

print(
    f"  Inference engine : {engine_path}"
)

print("")
print("Architecture:")

print(
    "  119 leakage-safe features"
)

print(
    "          ↓"
)

print(
    "  Linear Regression × 3"
)

print(
    "          ↓"
)

print(
    "  2-min / 10-min / 30-min"
)

print(
    "          ↓"
)

print(
    "  Power-output forecasts"
)

print("")
print(
    "STATUS: 600 MW MODEL READY FOR INTEGRATION"
)

print("=" * 80)