from pathlib import Path
import pandas as pd


# ============================================================
# STAGE 19: TRAINING FEATURE SELECTION REVIEW
# ============================================================

print("=" * 70)
print("STAGE 19: TRAINING FEATURE SELECTION REVIEW")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAINING_CANDIDATES = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_ml_candidates"
    / "hai_2305_training_ml_candidates.csv"
)

FEATURE_ROLES = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "hai_2305_feature_roles.csv"
)

CORRELATION_PAIRS = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_redundancy"
    / "hai_2305_training_high_correlation_pairs.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_feature_selection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_2305_training_feature_selection_review.csv"
)


# ------------------------------------------------------------
# LOAD FILES
# ------------------------------------------------------------

print("\nLoading training candidates...")
training = pd.read_csv(TRAINING_CANDIDATES)

print("Loading feature roles...")
roles = pd.read_csv(FEATURE_ROLES)

print("Loading Stage 18 correlation pairs...")
correlations = pd.read_csv(CORRELATION_PAIRS)

print("Files loaded successfully.")


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

assert "timestamp" in training.columns, \
    "Missing timestamp column in training candidates"

assert "feature" in roles.columns, \
    "Feature-role file must contain 'feature' column"

assert "feature_a" in correlations.columns, \
    "Correlation file missing feature_a"

assert "feature_b" in correlations.columns, \
    "Correlation file missing feature_b"

assert "correlation" in correlations.columns, \
    "Correlation file missing correlation"

assert "absolute_correlation" in correlations.columns, \
    "Correlation file missing absolute_correlation"


candidate_features = [
    col for col in training.columns
    if col != "timestamp"
]

print("\nCandidate feature count:", len(candidate_features))

assert len(candidate_features) == 66, \
    f"Expected 66 candidate features, found {len(candidate_features)}"


# ------------------------------------------------------------
# ROLE MAPPING
# ------------------------------------------------------------

role_columns = [
    col for col in roles.columns
    if col != "feature"
]

print("\nRole columns found:")
for col in role_columns:
    print("  -", col)


roles_indexed = roles.set_index("feature")


missing_roles = [
    feature
    for feature in candidate_features
    if feature not in roles_indexed.index
]

assert not missing_roles, \
    f"Missing role mappings: {missing_roles}"


# ------------------------------------------------------------
# BUILD CORRELATION SUMMARY
# ------------------------------------------------------------

correlation_info = {
    feature: []
    for feature in candidate_features
}


for _, row in correlations.iterrows():

    feature_a = row["feature_a"]
    feature_b = row["feature_b"]
    corr = row["correlation"]
    abs_corr = row["absolute_correlation"]

    if feature_a in correlation_info:
        correlation_info[feature_a].append(
            (feature_b, corr, abs_corr)
        )

    if feature_b in correlation_info:
        correlation_info[feature_b].append(
            (feature_a, corr, abs_corr)
        )


# ------------------------------------------------------------
# CREATE REVIEW TABLE
# ------------------------------------------------------------

review_rows = []

for feature in candidate_features:

    row = {
        "feature": feature,
    }

    # Add all available role information
    for column in role_columns:
        row[column] = roles_indexed.loc[feature, column]

    related = correlation_info[feature]

    row["high_correlation_count"] = len(related)

    if related:
        max_pair = max(
            related,
            key=lambda x: x[2]
        )

        row["max_absolute_correlation"] = max_pair[2]
        row["most_correlated_feature"] = max_pair[0]
        row["most_correlated_correlation"] = max_pair[1]

        related_features = [
            item[0]
            for item in sorted(
                related,
                key=lambda x: x[2],
                reverse=True
            )
        ]

        row["high_correlation_features"] = "; ".join(
            related_features
        )

    else:
        row["max_absolute_correlation"] = 0.0
        row["most_correlated_feature"] = ""
        row["most_correlated_correlation"] = 0.0
        row["high_correlation_features"] = ""

    # Selection status is intentionally NOT decided here.
    row["selection_status"] = "REVIEW_REQUIRED"

    row["selection_reason"] = ""

    review_rows.append(row)


review = pd.DataFrame(review_rows)


# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

review = review.sort_values(
    by=[
        "high_correlation_count",
        "max_absolute_correlation",
        "feature"
    ],
    ascending=[
        False,
        False,
        True
    ]
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

review.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 19 REVIEW SUMMARY")
print("=" * 70)

print(f"Training candidates       : {len(candidate_features)}")
print(
    "Features with |r| >= 0.95:",
    (review["high_correlation_count"] > 0).sum()
)
print(
    "Features without high correlation:",
    (review["high_correlation_count"] == 0).sum()
)

print("\nTop features requiring review:")

print(
    review[
        [
            "feature",
            "high_correlation_count",
            "max_absolute_correlation",
            "most_correlated_feature"
        ]
    ].head(20).to_string(index=False)
)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

assert len(review) == 66
assert review["feature"].is_unique
assert review["high_correlation_count"].ge(0).all()
assert review["max_absolute_correlation"].le(1.0 + 1e-10).all()

print("PASS: 66 features represented")
print("PASS: No duplicate feature names")
print("PASS: Correlation summary generated")
print("PASS: No feature was automatically removed")
print("PASS: Selection status remains REVIEW_REQUIRED")

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STAGE 19 REVIEW TABLE: COMPLETE")
print("=" * 70)