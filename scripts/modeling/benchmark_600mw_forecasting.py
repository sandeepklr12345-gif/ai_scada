"""
600 MW FORECASTING MODEL BENCHMARK

Purpose:
    Benchmark the existing 600 MW candidate feature sets against
    multiple regression models for the three forecasting horizons.

This script:
    - Uses the existing validated candidate datasets
    - Preserves chronological order
    - Uses an 80/20 time-ordered split
    - Does NOT shuffle
    - Does NOT modify source datasets
    - Evaluates MAE, RMSE and R2
    - Saves the benchmark results
    - Saves the best model for each horizon

Models:
    1. Linear Regression
    2. Random Forest Regressor
    3. HistGradientBoosting Regressor

Important:
    Feature sets are treated as candidates. No winner is assumed beforehand.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_DIR = (
    ROOT /
    "data/features/600mw/candidates"
)

MODEL_DIR = (
    ROOT /
    "models/forecasting/600mw"
)

RESULT_DIR = (
    ROOT /
    "data/processed/600mw/model_benchmark"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATES = {
    "A": "feature_set_A_power_history.csv",
    "B": "feature_set_B_process_core.csv",
    "C": "feature_set_C_thermal_process.csv",
    "D": "feature_set_D_full_candidate_pool.csv",
}

TARGETS = [
    "target_power_2min",
    "target_power_10min",
    "target_power_30min",
]

TRAIN_RATIO = 0.80


# ============================================================
# MODEL FACTORIES
# ============================================================

def create_models():

    return {
        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt"
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42
        ),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# MAIN
# ============================================================

print("")
print("=" * 80)
print("600 MW FORECASTING MODEL BENCHMARK")
print("=" * 80)
print("")

all_results = []
best_models = []


for candidate_name, filename in CANDIDATES.items():

    filepath = CANDIDATE_DIR / filename

    print("")
    print("-" * 80)
    print(f"FEATURE SET {candidate_name}")
    print("-" * 80)

    if not filepath.exists():

        print(
            f"FILE NOT FOUND: {filepath}"
        )

        continue

    print(
        f"Loading: {filename}"
    )

    df = pd.read_csv(
        filepath
    )

    print(
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Target availability
    # --------------------------------------------------------

    available_targets = [
        target
        for target in TARGETS
        if target in df.columns
    ]

    print(
        f"Targets found: {available_targets}"
    )

    if not available_targets:

        print(
            "No forecasting targets found. Skipping."
        )

        continue

    # --------------------------------------------------------
    # Candidate features
    # --------------------------------------------------------

    excluded_columns = set(
        available_targets
    )

    if timestamp_column:
        excluded_columns.add(
            timestamp_column
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    # Keep numeric model inputs only
    feature_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    print(
        f"Numeric feature count: {len(feature_columns)}"
    )

    if not feature_columns:

        print(
            "No numeric features available. Skipping."
        )

        continue

    # --------------------------------------------------------
    # Validate numeric feature matrix
    # --------------------------------------------------------

    X_all = df[
        feature_columns
    ].copy()

    # No imputation of source data.
    # Rows with unusable model values are removed only from
    # the specific training/evaluation target table.
    X_all = X_all.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Each target gets its own chronological split
    # --------------------------------------------------------

    for target in available_targets:

        print("")
        print(
            f"Target: {target}"
        )

        target_df = pd.concat(
            [
                X_all,
                df[target]
            ],
            axis=1
        ).dropna()

        X = target_df[
            feature_columns
        ]

        y = target_df[
            target
        ]

        total_rows = len(
            target_df
        )

        split_index = int(
            total_rows * TRAIN_RATIO
        )

        if split_index <= 0 or split_index >= total_rows:

            print(
                "Invalid chronological split. Skipping."
            )

            continue

        X_train = X.iloc[
            :split_index
        ]

        X_test = X.iloc[
            split_index:
        ]

        y_train = y.iloc[
            :split_index
        ]

        y_test = y.iloc[
            split_index:
        ]

        print(
            f"Train rows: {len(X_train)}"
        )

        print(
            f"Test rows : {len(X_test)}"
        )

        models = create_models()

        target_results = []

        for model_name, model in models.items():

            print(
                f"  Training {model_name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mae, rmse, r2 = calculate_metrics(
                y_test,
                predictions
            )

            result = {
                "feature_set": candidate_name,
                "target": target,
                "model": model_name,
                "total_rows": total_rows,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": len(feature_columns),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }

            all_results.append(
                result
            )

            target_results.append(
                result
            )

            print(
                f"    MAE : {mae:.6f}"
            )

            print(
                f"    RMSE: {rmse:.6f}"
            )

            print(
                f"    R2  : {r2:.6f}"
            )

        # ----------------------------------------------------
        # Select best model for this candidate + target
        # ----------------------------------------------------
        #
        # Primary metric: RMSE
        # Secondary metric: MAE
        #
        # Lower RMSE is better.
        # ----------------------------------------------------

        target_results.sort(
            key=lambda row: (
                row["RMSE"],
                row["MAE"]
            )
        )

        best = target_results[0]

        best_models.append(
            best
        )

        print("")
        print(
            "Best for this feature set/target:"
        )

        print(
            f"  {best['model']}"
        )

        print(
            f"  RMSE = {best['RMSE']:.6f}"
        )


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_path = (
    RESULT_DIR /
    "600mw_model_benchmark_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# BEST RESULT PER TARGET
# ============================================================

if not results_df.empty:

    best_per_target = (
        results_df
        .sort_values(
            ["target", "RMSE", "MAE"]
        )
        .groupby(
            "target",
            as_index=False
        )
        .first()
    )

else:

    best_per_target = pd.DataFrame()


best_results_path = (
    RESULT_DIR /
    "600mw_best_model_per_target.csv"
)

best_per_target.to_csv(
    best_results_path,
    index=False
)


# ============================================================
# TRAIN FINAL MODELS
# ============================================================
#
# For each horizon:
#   choose the best feature-set/model combination based on
#   chronological holdout RMSE.
#
# Then retrain that exact combination on ALL valid rows.
#
# The original datasets remain untouched.
# ============================================================

manifest = {
    "project": "AI_SCADA",
    "dataset": "600 MW",
    "selection_metric": "RMSE",
    "secondary_metric": "MAE",
    "split": "80/20 chronological holdout",
    "shuffle": False,
    "targets": {},
}


for _, best in best_per_target.iterrows():

    candidate_name = best["feature_set"]
    target = best["target"]
    model_name = best["model"]

    filename = CANDIDATES[
        candidate_name
    ]

    filepath = CANDIDATE_DIR / filename

    df = pd.read_csv(
        filepath
    )

    if "Time" in df.columns:

        timestamp_column = "Time"

        df[timestamp_column] = pd.to_datetime(
            df[timestamp_column],
            errors="coerce"
        )

        df = df.sort_values(
            timestamp_column
        ).reset_index(drop=True)

    elif "timestamp" in df.columns:

        timestamp_column = "timestamp"

        df[timestamp_column] = pd.to_datetime(
            df[timestamp_column],
            errors="coerce"
        )

        df = df.sort_values(
            timestamp_column
        ).reset_index(drop=True)

    else:

        timestamp_column = None

    excluded = set(
        TARGETS
    )

    if timestamp_column:
        excluded.add(
            timestamp_column
        )

    features = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    model_df = df[
        features + [target]
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    X = model_df[
        features
    ]

    y = model_df[
        target
    ]

    models = create_models()

    model = models[
        model_name
    ]

    print("")
    print(
        f"Training final {target} model:"
    )

    print(
        f"  Feature set: {candidate_name}"
    )

    print(
        f"  Model: {model_name}"
    )

    print(
        f"  Features: {len(features)}"
    )

    model.fit(
        X,
        y
    )

    safe_target = (
        target
        .replace(
            "target_power_",
            ""
        )
        .replace(
            "min",
            "min"
        )
    )

    model_path = (
        MODEL_DIR /
        f"600mw_{safe_target}_{model_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    manifest["targets"][target] = {
        "feature_set": candidate_name,
        "model": model_name,
        "feature_count": len(features),
        "features": features,
        "training_rows": len(model_df),
        "model_path": str(
            model_path.relative_to(ROOT)
        ),
    }


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_path = (
    MODEL_DIR /
    "600mw_forecasting_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("")
print("=" * 80)
print("600 MW FORECASTING BENCHMARK COMPLETE")
print("=" * 80)

print("")
print(
    f"Benchmark results:"
)

print(
    results_path
)

print("")
print(
    "Best model per target:"
)

print(
    best_results_path
)

print("")
print(
    "Model directory:"
)

print(
    MODEL_DIR
)

print("")
print(
    "Manifest:"
)

print(
    manifest_path
)

if not best_per_target.empty:

    print("")
    print(
        "SELECTED MODELS"
    )

    print(
        "-" * 80
    )

    for _, row in best_per_target.iterrows():

        print(
            f"{row['target']}: "
            f"Feature Set {row['feature_set']} + "
            f"{row['model']} | "
            f"RMSE={row['RMSE']:.6f} | "
            f"MAE={row['MAE']:.6f} | "
            f"R2={row['R2']:.6f}"
        )

print("")
print(
    "Next: review the benchmark results before treating the"
)

print(
    "selected models as the production forecasting models."
)

print("=" * 80)