from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAINING_CANDIDATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_ml_candidates"
    / "hai_2305_training_ml_candidates.csv"
)

ROLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "hai_2305_feature_roles.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_redundancy"
)

CORRELATION_MATRIX_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_training_candidate_correlation_matrix.csv"
)

HIGH_CORRELATION_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_training_high_correlation_pairs.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CORRELATION_THRESHOLD = 0.95


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 18: TRAINING-ONLY REDUNDANCY ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training candidate dataset...")

training = pd.read_csv(
    TRAINING_CANDIDATE_FILE
)

print("Loading feature roles...")

roles = pd.read_csv(
    ROLE_FILE
)

print("Files loaded successfully.")


# ============================================================
# IDENTIFY FEATURES
# ============================================================

assert "timestamp" in training.columns

feature_columns = [
    column
    for column in training.columns
    if column != "timestamp"
]

print("\nTraining candidate statistics:")
print(
    f"Rows              : {len(training)}"
)

print(
    f"Candidate features: {len(feature_columns)}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

assert len(feature_columns) == 66

assert len(set(feature_columns)) == 66

assert all(
    feature in roles["feature"].values
    for feature in feature_columns
)

assert all(
    pd.api.types.is_numeric_dtype(
        training[feature]
    )
    for feature in feature_columns
)

assert training[feature_columns].isna().sum().sum() == 0

assert np.isfinite(
    training[feature_columns].to_numpy()
).all()

print("PASS: 66 numeric training features")
print("PASS: No duplicate feature names")
print("PASS: All features have role mappings")
print("PASS: No missing values")
print("PASS: No infinite values")


# ============================================================
# CORRELATION MATRIX
# ============================================================

print("\nCalculating Pearson correlation matrix...")

correlation_matrix = training[
    feature_columns
].corr(
    method="pearson"
)

print("Correlation matrix calculated.")


# ============================================================
# SAVE CORRELATION MATRIX
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

correlation_matrix.to_csv(
    CORRELATION_MATRIX_OUTPUT
)


# ============================================================
# FEATURE ROLE LOOKUP
# ============================================================

role_lookup = dict(
    zip(
        roles["feature"],
        roles["role"]
    )
)


# ============================================================
# FIND HIGH-CORRELATION PAIRS
# ============================================================

pairs = []

for i in range(len(feature_columns)):

    feature_a = feature_columns[i]

    for j in range(i + 1, len(feature_columns)):

        feature_b = feature_columns[j]

        correlation = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        if (
            pd.notna(correlation)
            and abs(correlation)
            >= CORRELATION_THRESHOLD
        ):

            pairs.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": correlation,
                    "absolute_correlation":
                        abs(correlation),
                    "role_a":
                        role_lookup.get(
                            feature_a,
                            "Unknown"
                        ),
                    "role_b":
                        role_lookup.get(
                            feature_b,
                            "Unknown"
                        ),
                }
            )


# ============================================================
# CREATE PAIR DATAFRAME
# ============================================================

high_corr = pd.DataFrame(
    pairs
)

if not high_corr.empty:

    high_corr = high_corr.sort_values(
        by="absolute_correlation",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# SAVE HIGH-CORRELATION PAIRS
# ============================================================

high_corr.to_csv(
    HIGH_CORRELATION_OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 18 RESULTS")
print("=" * 70)

print(
    f"\nCorrelation threshold : "
    f"|r| >= {CORRELATION_THRESHOLD}"
)

print(
    f"Candidate features    : "
    f"{len(feature_columns)}"
)

print(
    f"High-correlation pairs: "
    f"{len(high_corr)}"
)


# ============================================================
# DISPLAY PAIRS
# ============================================================

if high_corr.empty:

    print(
        "\nNo feature pairs exceeded "
        f"|r| >= {CORRELATION_THRESHOLD}."
    )

else:

    print("\n" + "-" * 70)
    print("HIGH-CORRELATION FEATURE PAIRS")
    print("-" * 70)

    print(
        high_corr.to_string(
            index=False
        )
    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

assert correlation_matrix.shape == (
    66,
    66
)

assert list(
    correlation_matrix.columns
) == feature_columns

assert list(
    correlation_matrix.index
) == feature_columns

assert np.allclose(
    correlation_matrix.values,
    correlation_matrix.values.T,
    equal_nan=True
)

assert all(
    abs(correlation) >= CORRELATION_THRESHOLD
    for correlation in high_corr[
        "correlation"
    ]
)

assert high_corr[
    ["feature_a", "feature_b"]
].apply(
    lambda row:
        row["feature_a"]
        != row["feature_b"],
    axis=1
).all()


print(
    "PASS: 66 × 66 correlation matrix"
)

print(
    "PASS: Correlation matrix is symmetric"
)

print(
    "PASS: All reported pairs meet "
    f"|r| >= {CORRELATION_THRESHOLD}"
)

print(
    "PASS: No self-pairs"
)

print(
    "PASS: Analysis uses training data only"
)

print("\nOutputs saved to:")

print(
    CORRELATION_MATRIX_OUTPUT
)

print(
    HIGH_CORRELATION_OUTPUT
)

print("\n" + "=" * 70)
print("STAGE 18: COMPLETE")
print("=" * 70)