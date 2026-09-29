from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STAGE 22C
# HAI 23.05 - Temporal Feature Quality Validation
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"
    r"\data\features\hai\hai-23.05"
    r"\temporal_representation"
    r"\hai_2305_training_temporal_features.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"
    r"\data\features\hai\hai-23.05"
    r"\temporal_representation"
)

QUALITY_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_feature_quality.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_feature_quality_summary.csv"
)


print("=" * 70)
print("STAGE 22C: TEMPORAL FEATURE QUALITY VALIDATION")
print("=" * 70)


# ============================================================
# 1. LOAD
# ============================================================

print("\nLoading temporal feature dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. INPUT VALIDATION
# ============================================================

assert "timestamp" in df.columns

feature_columns = [
    col for col in df.columns
    if col != "timestamp"
]

assert len(feature_columns) == 174

print(f"Temporal features: {len(feature_columns)}")


# ============================================================
# 3. BASIC QUALITY ANALYSIS
# ============================================================

print("\nAnalyzing temporal feature quality...")

records = []

for i, feature in enumerate(
    feature_columns,
    start=1
):

    s = df[feature]

    total_rows = len(s)

    missing_count = int(s.isna().sum())

    finite_mask = np.isfinite(
        s.dropna().to_numpy(dtype=float)
    )

    infinite_count = int(
        (~finite_mask).sum()
    )

    valid = s.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    valid_count = len(valid)

    if valid_count > 0:

        mean_value = float(valid.mean())
        std_value = float(valid.std())

        min_value = float(valid.min())
        max_value = float(valid.max())

        q01 = float(valid.quantile(0.01))
        q05 = float(valid.quantile(0.05))
        q25 = float(valid.quantile(0.25))
        median = float(valid.quantile(0.50))
        q75 = float(valid.quantile(0.75))
        q95 = float(valid.quantile(0.95))
        q99 = float(valid.quantile(0.99))

        unique_count = int(
            valid.nunique()
        )

        abs_max = float(
            valid.abs().max()
        )

    else:

        mean_value = np.nan
        std_value = np.nan

        min_value = np.nan
        max_value = np.nan

        q01 = np.nan
        q05 = np.nan
        q25 = np.nan
        median = np.nan
        q75 = np.nan
        q95 = np.nan
        q99 = np.nan

        unique_count = 0
        abs_max = np.nan


    # --------------------------------------------------------
    # Missing percentage
    # --------------------------------------------------------

    missing_pct = (
        missing_count / total_rows
    ) * 100


    # --------------------------------------------------------
    # Constant / near-constant classification
    # --------------------------------------------------------

    if valid_count == 0:

        classification = "NO_VALID_DATA"

    elif unique_count <= 1:

        classification = "CONSTANT"

    elif unique_count <= 10:

        classification = "VERY_LOW_CARDINALITY"

    elif std_value == 0:

        classification = "CONSTANT"

    else:

        classification = "VARIABLE"


    records.append({
        "feature": feature,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "infinite_count": infinite_count,
        "valid_count": valid_count,
        "unique_count": unique_count,
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "q01": q01,
        "q05": q05,
        "q25": q25,
        "median": median,
        "q75": q75,
        "q95": q95,
        "q99": q99,
        "max": max_value,
        "absolute_max": abs_max,
        "classification": classification
    })


quality_df = pd.DataFrame(records)


# ============================================================
# 4. GLOBAL VALIDATION
# ============================================================

print("\nGlobal validation...")

total_missing = int(
    quality_df["missing_count"].sum()
)

total_infinite = int(
    quality_df["infinite_count"].sum()
)

constant_count = int(
    (
        quality_df["classification"]
        == "CONSTANT"
    ).sum()
)

very_low_cardinality_count = int(
    (
        quality_df["classification"]
        == "VERY_LOW_CARDINALITY"
    ).sum()
)

variable_count = int(
    (
        quality_df["classification"]
        == "VARIABLE"
    ).sum()
)


print(
    f"Total temporal NaNs       : "
    f"{total_missing:,}"
)

print(
    f"Total infinite values     : "
    f"{total_infinite:,}"
)

print(
    f"Constant features         : "
    f"{constant_count}"
)

print(
    f"Very low cardinality      : "
    f"{very_low_cardinality_count}"
)

print(
    f"Variable features         : "
    f"{variable_count}"
)


# ============================================================
# 5. EXPECTED NaN VALIDATION
# ============================================================

print("\nChecking expected boundary NaNs...")

diff_features = [
    f for f in feature_columns
    if "__diff_1s" in f
]

abs_diff_features = [
    f for f in feature_columns
    if "__abs_diff_1s" in f
]

rolling_features = [
    f for f in feature_columns
    if "__rolling_std_5s" in f
]

assert len(diff_features) == 58
assert len(abs_diff_features) == 58
assert len(rolling_features) == 58

for feature in diff_features:

    assert (
        df[feature].isna().sum()
        == 4
    )

for feature in abs_diff_features:

    assert (
        df[feature].isna().sum()
        == 4
    )

for feature in rolling_features:

    assert (
        df[feature].isna().sum()
        == 16
    )


print(
    "PASS: Expected temporal boundary "
    "NaN structure confirmed"
)


# ============================================================
# 6. INFINITE VALUE CHECK
# ============================================================

assert total_infinite == 0

print(
    "PASS: No infinite values"
)


# ============================================================
# 7. CONSTANT FEATURE CHECK
# ============================================================

print("\nConstant feature check...")

if constant_count == 0:

    print(
        "PASS: No constant temporal features"
    )

else:

    print(
        "WARNING: Constant temporal features found:"
    )

    print(
        quality_df[
            quality_df["classification"]
            == "CONSTANT"
        ]["feature"].to_string(
            index=False
        )
    )


# ============================================================
# 8. TOP VARIABLE FEATURES
# ============================================================

print("\nTop temporal features by standard deviation:")

top_std = (
    quality_df
    .sort_values(
        "std",
        ascending=False
    )
    .head(20)
)

print(
    top_std[
        [
            "feature",
            "std",
            "unique_count",
            "missing_count"
        ]
    ].to_string(index=False)
)


# ============================================================
# 9. LOW VARIABILITY FEATURES
# ============================================================

print(
    "\nLowest non-zero standard deviation "
    "temporal features:"
)

low_std = (
    quality_df[
        quality_df["std"] > 0
    ]
    .sort_values(
        "std",
        ascending=True
    )
    .head(20)
)

print(
    low_std[
        [
            "feature",
            "std",
            "unique_count",
            "missing_count"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. EXTREME VALUE REVIEW
# ============================================================

print(
    "\nTop features by absolute maximum:"
)

top_abs = (
    quality_df
    .sort_values(
        "absolute_max",
        ascending=False
    )
    .head(20)
)

print(
    top_abs[
        [
            "feature",
            "absolute_max",
            "q99",
            "max"
        ]
    ].to_string(index=False)
)


# ============================================================
# 11. SAVE QUALITY REPORT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

quality_df.to_csv(
    QUALITY_FILE,
    index=False
)


# ============================================================
# 12. SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame([
    {
        "metric": "training_rows",
        "value": len(df)
    },
    {
        "metric": "temporal_features",
        "value": len(feature_columns)
    },
    {
        "metric": "total_missing_values",
        "value": total_missing
    },
    {
        "metric": "total_infinite_values",
        "value": total_infinite
    },
    {
        "metric": "constant_features",
        "value": constant_count
    },
    {
        "metric": "very_low_cardinality_features",
        "value": very_low_cardinality_count
    },
    {
        "metric": "variable_features",
        "value": variable_count
    },
    {
        "metric": "diff_features",
        "value": len(diff_features)
    },
    {
        "metric": "absolute_diff_features",
        "value": len(abs_diff_features)
    },
    {
        "metric": "rolling_std_features",
        "value": len(rolling_features)
    }
])

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 13. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(quality_df) == 174

assert (
    quality_df["feature"].nunique()
    == 174
)

assert total_infinite == 0

assert total_missing == (
    (58 * 4) +
    (58 * 4) +
    (58 * 16)
)

assert (
    variable_count
    + constant_count
    + very_low_cardinality_count
    == 174
)

print(
    "PASS: 174 temporal features analyzed"
)

print(
    "PASS: Expected NaN count confirmed"
)

print(
    "PASS: No infinite values"
)

print(
    "PASS: Feature classifications complete"
)

print(
    "PASS: Quality report created"
)

print(
    "PASS: Training data only"
)

print(
    "\nQuality report:"
)

print(QUALITY_FILE)

print(
    "\nSummary report:"
)

print(SUMMARY_FILE)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 22C: COMPLETE"
)

print(
    "=" * 70
)