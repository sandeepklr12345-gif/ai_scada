"""
600 MW FINAL LEAKAGE-SAFE MODEL VALIDATION

Purpose:
    Re-test the 600 MW forecasting branch after excluding the
    previously flagged potentially derived / leakage-prone variables.

This is a validation experiment only.
It does not overwrite the existing benchmark or model artifacts.

Flagged variables from the earlier leakage review:
    - Plant auxiliary power
    - Net power output
    - Boiler efficiency
    - Standard coal consumption rate for power generation
    - Turbine heat rate
    - Unit efficiency

The script:
    1. Loads Feature Set D.
    2. Removes only columns whose names match the flagged leakage
       concepts.
    3. Reports exactly which columns were removed.
    4. Trains Linear Regression, Random Forest and
       HistGradientBoosting.
    5. Uses the same 80/20 chronological split.
    6. Evaluates 2/10/30 minute targets.
    7. Compares leakage-safe results with the existing benchmark.
    8. Saves a separate leakage-safe report.

IMPORTANT:
    The matching is intentionally conservative. If a flagged concept
    cannot be identified from the column name, it is NOT silently
    removed. The report will show the unmatched concepts.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT /
    "data/features/600mw/candidates/"
    "feature_set_D_full_candidate_pool.csv"
)

BENCHMARK_FILE = (
    ROOT /
    "data/processed/600mw/model_benchmark/"
    "600mw_best_model_per_target.csv"
)

OUT_DIR = (
    ROOT /
    "data/processed/600mw/model_benchmark/"
    "leakage_safe"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

TARGETS = [
    "target_power_2min",
    "target_power_10min",
    "target_power_30min"
]

TRAIN_RATIO = 0.80

# Conservative name fragments.
# We inspect actual matches before training.
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


# ============================================================
# HELPERS
# ============================================================

def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(name).lower()
    ).strip()


def find_flagged_columns(columns):

    matched = {}
    matched_columns = set()

    normalized = {
        column: normalize_name(column)
        for column in columns
    }

    for concept, patterns in FLAGGED_PATTERNS.items():

        matches = []

        for column, clean_name in normalized.items():

            for pattern in patterns:

                if re.search(pattern, clean_name):

                    matches.append(column)
                    matched_columns.add(column)
                    break

        matched[concept] = sorted(
            set(matches)
        )

    return matched, matched_columns


def make_models():

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


def metrics(y_true, y_pred):

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
# LOAD
# ============================================================

print("")
print("=" * 80)
print("600 MW FINAL LEAKAGE-SAFE MODEL VALIDATION")
print("=" * 80)
print("")

if not FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Feature Set D not found:\n{FEATURE_FILE}"
    )

print(
    f"Loading Feature Set D:\n{FEATURE_FILE}"
)

df = pd.read_csv(
    FEATURE_FILE
)

print(
    f"Original shape: {df.shape[0]} rows x {df.shape[1]} columns"
)


# ============================================================
# IDENTIFY TIMESTAMP
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
# FIND FLAGGED COLUMNS
# ============================================================

matched, matched_columns = find_flagged_columns(
    df.columns
)

print("")
print("LEAKAGE REVIEW MATCHES")
print("-" * 80)

for concept, columns in matched.items():

    print("")
    print(concept + ":")

    if columns:

        for column in columns:
            print(
                f"  REMOVE: {column}"
            )

    else:

        print(
            "  No direct column-name match found."
        )


print("")
print(
    f"Columns flagged for exclusion: "
    f"{len(matched_columns)}"
)


# ============================================================
# BUILD SAFE FEATURE LIST
# ============================================================

excluded_columns = set(
    TARGETS
)

if timestamp_column:

    excluded_columns.add(
        timestamp_column
    )

excluded_columns.update(
    matched_columns
)

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
    and pd.api.types.is_numeric_dtype(
        df[column]
    )
]

print("")
print(
    f"Leakage-safe numeric feature count: "
    f"{len(feature_columns)}"
)


# ============================================================
# SAVE FEATURE MANIFEST
# ============================================================

manifest_path = (
    OUT_DIR /
    "600mw_leakage_safe_feature_manifest.csv"
)

pd.DataFrame({
    "feature": feature_columns,
    "status": "KEEP"
}).to_csv(
    manifest_path,
    index=False
)

excluded_path = (
    OUT_DIR /
    "600mw_leakage_review_excluded_columns.csv"
)

pd.DataFrame({
    "excluded_column": sorted(
        matched_columns
    )
}).to_csv(
    excluded_path,
    index=False
)


# ============================================================
# TRAIN/EVALUATE
# ============================================================

results = []

for target in TARGETS:

    print("")
    print("-" * 80)
    print(
        f"TARGET: {target}"
    )
    print("-" * 80)

    target_df = pd.concat(
        [
            df[feature_columns],
            df[target]
        ],
        axis=1
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    X = target_df[
        feature_columns
    ]

    y = target_df[
        target
    ]

    split = int(
        len(target_df) *
        TRAIN_RATIO
    )

    X_train = X.iloc[
        :split
    ]

    X_test = X.iloc[
        split:
    ]

    y_train = y.iloc[
        :split
    ]

    y_test = y.iloc[
        split:
    ]

    print(
        f"Train rows: {len(X_train)}"
    )

    print(
        f"Test rows : {len(X_test)}"
    )

    for model_name, model in make_models().items():

        print(
            f"Training {model_name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        mae, rmse, r2 = metrics(
            y_test,
            prediction
        )

        print(
            f"  MAE={mae:.6f} "
            f"RMSE={rmse:.6f} "
            f"R2={r2:.6f}"
        )

        results.append({
            "target": target,
            "model": model_name,
            "feature_count": len(feature_columns),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })


results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST LEAKAGE-SAFE MODEL PER TARGET
# ============================================================

best_df = (
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


# ============================================================
# LOAD ORIGINAL BENCHMARK FOR COMPARISON
# ============================================================

comparison_rows = []

if BENCHMARK_FILE.exists():

    original = pd.read_csv(
        BENCHMARK_FILE
    )

    for _, safe_row in best_df.iterrows():

        original_rows = original[
            original["target"] ==
            safe_row["target"]
        ]

        if not original_rows.empty:

            original_row = original_rows.iloc[0]

            comparison_rows.append({
                "target": safe_row["target"],

                "original_feature_set":
                    original_row["feature_set"],

                "original_model":
                    original_row["model"],

                "original_RMSE":
                    original_row["RMSE"],

                "original_MAE":
                    original_row["MAE"],

                "original_R2":
                    original_row["R2"],

                "leakage_safe_model":
                    safe_row["model"],

                "leakage_safe_RMSE":
                    safe_row["RMSE"],

                "leakage_safe_MAE":
                    safe_row["MAE"],

                "leakage_safe_R2":
                    safe_row["R2"],

                "RMSE_change":
                    safe_row["RMSE"] -
                    original_row["RMSE"],

                "R2_change":
                    safe_row["R2"] -
                    original_row["R2"]
            })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUT_DIR /
    "600mw_leakage_safe_benchmark_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

best_path = (
    OUT_DIR /
    "600mw_leakage_safe_best_models.csv"
)

best_df.to_csv(
    best_path,
    index=False
)

comparison_path = (
    OUT_DIR /
    "600mw_original_vs_leakage_safe.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# REPORT
# ============================================================

report_lines = []

report_lines += [
    "=" * 80,
    "600 MW LEAKAGE-SAFE VALIDATION REPORT",
    "=" * 80,
    "",
    f"Feature Set D source: {FEATURE_FILE}",
    f"Original columns: {df.shape[1]}",
    f"Leakage-safe features: {len(feature_columns)}",
    "",
    "EXCLUDED COLUMNS",
    "-" * 80,
]

for concept, columns in matched.items():

    report_lines.append(
        concept + ":"
    )

    if columns:

        for column in columns:

            report_lines.append(
                f"  {column}"
            )

    else:

        report_lines.append(
            "  No direct name match"
        )

report_lines += [
    "",
    "LEAKAGE-SAFE BEST MODELS",
    "-" * 80,
]

for _, row in best_df.iterrows():

    report_lines.append(
        f"{row['target']}: "
        f"{row['model']} | "
        f"RMSE={row['RMSE']:.6f} | "
        f"MAE={row['MAE']:.6f} | "
        f"R2={row['R2']:.6f}"
    )

if not comparison_df.empty:

    report_lines += [
        "",
        "ORIGINAL VS LEAKAGE-SAFE",
        "-" * 80,
    ]

    for _, row in comparison_df.iterrows():

        report_lines.append(
            f"{row['target']}: "
            f"RMSE change={row['RMSE_change']:.6f}, "
            f"R2 change={row['R2_change']:.6f}"
        )

report_lines += [
    "",
    "INTERPRETATION RULE",
    "-" * 80,
    "The leakage-safe model is NOT automatically declared superior.",
    "The original benchmark and leakage-safe benchmark must be compared.",
    "The final production model should use only features that are",
    "available legitimately at prediction time.",
    "",
    "STATUS: VALIDATION COMPLETE",
]

report_path = (
    OUT_DIR /
    "600mw_leakage_safe_validation_report.txt"
)

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("")
print("=" * 80)
print("600 MW LEAKAGE-SAFE VALIDATION COMPLETE")
print("=" * 80)

print("")
print(
    f"Leakage-safe features: {len(feature_columns)}"
)

print("")
print(
    "Results:"
)

print(
    results_path
)

print("")
print(
    "Best models:"
)

print(
    best_path
)

print("")
print(
    "Comparison:"
)

print(
    comparison_path
)

print("")
print(
    "Report:"
)

print(
    report_path
)

print("")
print(
    "IMPORTANT: Do not replace the existing production artifacts yet."
)

print(
    "Review the comparison before final model selection."
)

print("=" * 80)