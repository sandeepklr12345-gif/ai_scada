"""
600 MW TEMPORAL ROBUSTNESS VALIDATION

Final validation gate before freezing the 600 MW forecasting models.

Uses:
    Feature Set D
    119 leakage-safe features
    Linear Regression

Evaluation:
    Three chronological walk-forward-style holdout windows.

No shuffling.
No random train/test split.
No modification of existing datasets or model artifacts.

The goal is to check whether the strong leakage-safe results remain
stable across different chronological portions of the one-week dataset.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
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

# The same leakage concepts used in the previous validation.
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


# Three chronological evaluation windows.
#
# For each fold:
#   train_end = fraction of valid chronological rows
#   test_start = train_end
#   test_end = next fraction
#
# Fold 1: train 60%, test next 20%
# Fold 2: train 70%, test next 20%
# Fold 3: train 80%, test final 20%
#
FOLDS = [
    ("Fold_1", 0.60, 0.80),
    ("Fold_2", 0.70, 0.90),
    ("Fold_3", 0.80, 1.00),
]


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

    matched_columns = set()

    for column in columns:

        clean = normalize_name(column)

        for patterns in FLAGGED_PATTERNS.values():

            for pattern in patterns:

                if re.search(pattern, clean):

                    matched_columns.add(column)
                    break

    return matched_columns


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
# LOAD
# ============================================================

print("")
print("=" * 80)
print("600 MW TEMPORAL ROBUSTNESS VALIDATION")
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
# SORT CHRONOLOGICALLY
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

    print(
        f"Timestamp column: {timestamp_column}"
    )

else:

    print(
        "WARNING: timestamp column not found."
    )


# ============================================================
# IDENTIFY LEAKAGE-SAFE FEATURES
# ============================================================

flagged_columns = find_flagged_columns(
    df.columns
)

excluded = set(
    TARGETS
)

if timestamp_column:
    excluded.add(
        timestamp_column
    )

excluded.update(
    flagged_columns
)

feature_columns = [
    column
    for column in df.columns
    if column not in excluded
    and pd.api.types.is_numeric_dtype(
        df[column]
    )
]


print(
    f"Leakage-safe feature count: {len(feature_columns)}"
)

print(
    f"Excluded leakage-review columns: {len(flagged_columns)}"
)


# ============================================================
# RUN TEMPORAL FOLDS
# ============================================================

results = []

for target in TARGETS:

    print("")
    print("-" * 80)
    print(
        f"TARGET: {target}"
    )
    print("-" * 80)

    working = pd.concat(
        [
            df[feature_columns],
            df[target]
        ],
        axis=1
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    X = working[
        feature_columns
    ]

    y = working[
        target
    ]

    total = len(
        working
    )

    for fold_name, train_fraction, test_end_fraction in FOLDS:

        train_end = int(
            total * train_fraction
        )

        test_end = int(
            total * test_end_fraction
        )

        X_train = X.iloc[
            :train_end
        ]

        y_train = y.iloc[
            :train_end
        ]

        X_test = X.iloc[
            train_end:test_end
        ]

        y_test = y.iloc[
            train_end:test_end
        ]

        model = LinearRegression()

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        mae, rmse, r2 = calculate_metrics(
            y_test,
            prediction
        )

        results.append({
            "target": target,
            "fold": fold_name,
            "train_fraction": train_fraction,
            "test_end_fraction": test_end_fraction,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_count": len(feature_columns),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(
            f"{fold_name}: "
            f"train={len(X_train)}, "
            f"test={len(X_test)}, "
            f"MAE={mae:.6f}, "
            f"RMSE={rmse:.6f}, "
            f"R2={r2:.6f}"
        )


results_df = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary_rows = []

for target in TARGETS:

    subset = results_df[
        results_df["target"] ==
        target
    ]

    summary_rows.append({
        "target": target,

        "folds": len(subset),

        "mean_MAE":
            subset["MAE"].mean(),

        "std_MAE":
            subset["MAE"].std(ddof=0),

        "mean_RMSE":
            subset["RMSE"].mean(),

        "std_RMSE":
            subset["RMSE"].std(ddof=0),

        "mean_R2":
            subset["R2"].mean(),

        "std_R2":
            subset["R2"].std(ddof=0),

        "min_R2":
            subset["R2"].min(),

        "max_R2":
            subset["R2"].max(),

        "min_RMSE":
            subset["RMSE"].min(),

        "max_RMSE":
            subset["RMSE"].max()
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE RESULTS
# ============================================================

fold_path = (
    OUT_DIR /
    "600mw_temporal_robustness_folds.csv"
)

summary_path = (
    OUT_DIR /
    "600mw_temporal_robustness_summary.csv"
)

results_df.to_csv(
    fold_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# CREATE REPORT
# ============================================================

report = []

report += [
    "=" * 80,
    "600 MW TEMPORAL ROBUSTNESS VALIDATION REPORT",
    "=" * 80,
    "",
    "Model: Linear Regression",
    "Feature source: Feature Set D",
    f"Leakage-safe feature count: {len(feature_columns)}",
    "Evaluation: chronological expanding training windows",
    "Shuffle: False",
    "",
    "FOLDS",
    "-" * 80,
    "Fold 1: first 60% train -> next 20% test",
    "Fold 2: first 70% train -> next 20% test",
    "Fold 3: first 80% train -> final 20% test",
    "",
    "RESULTS",
    "-" * 80,
]


for _, row in results_df.iterrows():

    report.append(
        f"{row['target']} | "
        f"{row['fold']} | "
        f"MAE={row['MAE']:.6f} | "
        f"RMSE={row['RMSE']:.6f} | "
        f"R2={row['R2']:.6f}"
    )


report += [
    "",
    "SUMMARY",
    "-" * 80,
]


for _, row in summary_df.iterrows():

    report.append(
        f"{row['target']}: "
        f"mean RMSE={row['mean_RMSE']:.6f}, "
        f"std RMSE={row['std_RMSE']:.6f}, "
        f"mean R2={row['mean_R2']:.6f}, "
        f"min R2={row['min_R2']:.6f}"
    )


report += [
    "",
    "INTERPRETATION",
    "-" * 80,
    "This test checks whether performance remains consistent across",
    "different chronological portions of the available operating data.",
    "It does not use random shuffling.",
    "Existing model artifacts were not overwritten.",
    "",
    "STATUS: TEMPORAL ROBUSTNESS VALIDATION COMPLETE",
]


report_path = (
    OUT_DIR /
    "600mw_temporal_robustness_report.txt"
)

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("")
print("=" * 80)
print("TEMPORAL ROBUSTNESS VALIDATION COMPLETE")
print("=" * 80)

print("")
print(
    "Fold results:"
)

print(
    fold_path
)

print("")
print(
    "Summary:"
)

print(
    summary_path
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
    "No existing model artifacts were overwritten."
)

print("")
print(
    "Review the fold results before freezing the final 600 MW models."
)

print("=" * 80)