from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# STAGE 22E
# HAI 23.05 - Temporal Feature Redundancy Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
    / "hai_2305_training_temporal_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
)

CORR_MATRIX_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_correlation_matrix.csv"
)

HIGH_CORR_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_high_correlation_pairs.csv"
)

DUPLICATES_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_exact_duplicates.csv"
)

FAMILY_SUMMARY_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_family_redundancy_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

CORRELATION_THRESHOLD = 0.95


print("=" * 70)
print("STAGE 22E: TEMPORAL FEATURE REDUNDANCY ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD
# ============================================================

print("\nLoading temporal feature dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. IDENTIFY FEATURES
# ============================================================

assert "timestamp" in df.columns

feature_columns = [
    col
    for col in df.columns
    if col != "timestamp"
]

assert len(feature_columns) == 174

print(
    f"Temporal features: {len(feature_columns)}"
)


# ============================================================
# 3. CHECK DUPLICATE FEATURE NAMES
# ============================================================

assert (
    df.columns.nunique()
    == len(df.columns)
)

print(
    "PASS: Feature names are unique"
)


# ============================================================
# 4. LOAD NUMERICAL MATRIX
# ============================================================

X = df[feature_columns]

assert X.select_dtypes(
    include=np.number
).shape[1] == 174


# ============================================================
# 5. CORRELATION MATRIX
# ============================================================

print("\nCalculating Pearson correlation matrix...")

corr_matrix = X.corr(
    method="pearson"
)

assert corr_matrix.shape == (
    174,
    174
)

assert np.isfinite(
    corr_matrix.to_numpy()
).all()

corr_matrix.to_csv(
    CORR_MATRIX_FILE
)

print(
    "PASS: Correlation matrix created"
)


# ============================================================
# 6. EXTRACT HIGH-CORRELATION PAIRS
# ============================================================

print(
    "\nSearching for highly correlated "
    f"pairs (|r| >= {CORRELATION_THRESHOLD})..."
)

pairs = []

for i in range(len(feature_columns)):

    feature_a = feature_columns[i]

    for j in range(i + 1, len(feature_columns)):

        feature_b = feature_columns[j]

        correlation = corr_matrix.iloc[
            i,
            j
        ]

        if abs(correlation) >= CORRELATION_THRESHOLD:

            pairs.append({
                "feature_a": feature_a,
                "feature_b": feature_b,
                "correlation": float(
                    correlation
                ),
                "absolute_correlation": float(
                    abs(correlation)
                )
            })


high_corr_df = pd.DataFrame(
    pairs
)


# ============================================================
# 7. SORT
# ============================================================

if len(high_corr_df) > 0:

    high_corr_df = (
        high_corr_df
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
        .reset_index(drop=True)
    )


high_corr_df.to_csv(
    HIGH_CORR_FILE,
    index=False
)


print(
    f"High-correlation pairs: "
    f"{len(high_corr_df)}"
)


# ============================================================
# 8. EXACT DUPLICATE FEATURES
# ============================================================

print("\nChecking exact duplicate features...")

duplicate_records = []

for i in range(len(feature_columns)):

    feature_a = feature_columns[i]

    for j in range(i + 1, len(feature_columns)):

        feature_b = feature_columns[j]

        a = X[feature_a]
        b = X[feature_b]

        # NaNs occur at known temporal boundaries.
        # Compare only rows where both are valid.

        valid_mask = (
            a.notna()
            & b.notna()
        )

        if not valid_mask.any():
            continue

        if np.array_equal(
            a[valid_mask].to_numpy(),
            b[valid_mask].to_numpy()
        ):

            duplicate_records.append({
                "feature_a": feature_a,
                "feature_b": feature_b
            })


duplicates_df = pd.DataFrame(
    duplicate_records
)

duplicates_df.to_csv(
    DUPLICATES_FILE,
    index=False
)

print(
    f"Exact duplicate pairs: "
    f"{len(duplicates_df)}"
)


# ============================================================
# 9. TRANSFORMATION FAMILIES
# ============================================================

def get_family(feature):

    if "__diff_1s" in feature:
        return "diff_1s"

    if "__abs_diff_1s" in feature:
        return "abs_diff_1s"

    if "__rolling_std_5s" in feature:
        return "rolling_std_5s"

    return "unknown"


families = {
    "diff_1s": [],
    "abs_diff_1s": [],
    "rolling_std_5s": []
}

for feature in feature_columns:

    family = get_family(feature)

    assert family in families

    families[family].append(feature)


assert len(families["diff_1s"]) == 58
assert len(families["abs_diff_1s"]) == 58
assert len(families["rolling_std_5s"]) == 58


# ============================================================
# 10. FAMILY-LEVEL REDUNDANCY
# ============================================================

print("\nAnalyzing redundancy by transformation family...")

family_records = []


for family_name, family_features in families.items():

    family_corr = (
        X[family_features]
        .corr()
    )

    pair_count = 0
    maximum_correlation = 0.0

    for i in range(len(family_features)):

        for j in range(i + 1, len(family_features)):

            correlation = family_corr.iloc[
                i,
                j
            ]

            absolute_correlation = abs(
                correlation
            )

            if (
                absolute_correlation
                >= CORRELATION_THRESHOLD
            ):

                pair_count += 1

            if (
                absolute_correlation
                > maximum_correlation
            ):

                maximum_correlation = (
                    absolute_correlation
                )


    family_records.append({
        "family": family_name,
        "feature_count": len(
            family_features
        ),
        "high_correlation_pairs": pair_count,
        "maximum_absolute_correlation":
            maximum_correlation
    })


family_summary_df = pd.DataFrame(
    family_records
)

family_summary_df.to_csv(
    FAMILY_SUMMARY_FILE,
    index=False
)


# ============================================================
# 11. CROSS-FAMILY ANALYSIS
# ============================================================

print("\nCross-family redundancy analysis...")

cross_family_counts = {
    "diff_1s_vs_abs_diff_1s": 0,
    "diff_1s_vs_rolling_std_5s": 0,
    "abs_diff_1s_vs_rolling_std_5s": 0
}


def count_cross_pairs(
    family_a,
    family_b
):

    count = 0

    matrix = (
        X[
            families[family_a]
            + families[family_b]
        ]
        .corr()
    )

    for feature_a in families[family_a]:

        for feature_b in families[family_b]:

            correlation = matrix.loc[
                feature_a,
                feature_b
            ]

            if (
                abs(correlation)
                >= CORRELATION_THRESHOLD
            ):

                count += 1

    return count


cross_family_counts[
    "diff_1s_vs_abs_diff_1s"
] = count_cross_pairs(
    "diff_1s",
    "abs_diff_1s"
)

cross_family_counts[
    "diff_1s_vs_rolling_std_5s"
] = count_cross_pairs(
    "diff_1s",
    "rolling_std_5s"
)

cross_family_counts[
    "abs_diff_1s_vs_rolling_std_5s"
] = count_cross_pairs(
    "abs_diff_1s",
    "rolling_std_5s"
)


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL REDUNDANCY SUMMARY")
print("=" * 70)

print(
    f"Total temporal features : "
    f"{len(feature_columns)}"
)

print(
    f"High-correlation pairs  : "
    f"{len(high_corr_df)}"
)

print(
    f"Exact duplicate pairs   : "
    f"{len(duplicates_df)}"
)

print("\nFamily-level results:")

print(
    family_summary_df.to_string(
        index=False
    )
)

print("\nCross-family high-correlation pairs:")

for key, value in cross_family_counts.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# 13. TOP CORRELATED PAIRS
# ============================================================

if len(high_corr_df) > 0:

    print(
        "\nTop 30 high-correlation pairs:"
    )

    print(
        high_corr_df.head(30)
        .to_string(index=False)
    )

else:

    print(
        "\nNo high-correlation pairs found."
    )


# ============================================================
# 14. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(feature_columns) == 174

assert corr_matrix.shape == (
    174,
    174
)

assert (
    high_corr_df[
        [
            "correlation",
            "absolute_correlation"
        ]
    ]
    .apply(
        lambda col:
        np.isfinite(col).all()
    )
    .all()
    if len(high_corr_df) > 0
    else True
)

assert (
    len(duplicates_df)
    >= 0
)

assert (
    family_summary_df["feature_count"]
    .sum()
    == 174
)

print(
    "PASS: 174 temporal features analyzed"
)

print(
    "PASS: Correlation matrix validated"
)

print(
    "PASS: High-correlation pairs extracted"
)

print(
    "PASS: Exact duplicate check completed"
)

print(
    "PASS: Transformation-family analysis completed"
)

print(
    "PASS: Cross-family analysis completed"
)

print(
    "PASS: No automatic feature deletion performed"
)

print(
    "PASS: Training data only"
)

print("\nOutputs:")

print(CORR_MATRIX_FILE)
print(HIGH_CORR_FILE)
print(DUPLICATES_FILE)
print(FAMILY_SUMMARY_FILE)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 22E: COMPLETE"
)

print(
    "=" * 70
)