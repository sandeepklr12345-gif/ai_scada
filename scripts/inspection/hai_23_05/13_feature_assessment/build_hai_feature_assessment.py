from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
)

ROLE_FILE = FEATURE_DIR / "hai_2305_feature_roles.csv"
REDUNDANCY_FILE = FEATURE_DIR / "hai_2305_redundancy_analysis.csv"
LABEL_FILE = FEATURE_DIR / "hai_2305_label_relationship_analysis.csv"

OUTPUT_FILE = FEATURE_DIR / "hai_2305_feature_assessment.csv"


# ============================================================
# LOAD
# ============================================================

roles = pd.read_csv(ROLE_FILE)
redundancy = pd.read_csv(REDUNDANCY_FILE)
labels = pd.read_csv(LABEL_FILE)


# ============================================================
# BASE FEATURE TABLE
# ============================================================

assessment = roles[
    [
        "feature",
        "role",
        "variability",
        "constant_test1",
        "constant_test2",
        "constant_both",
    ]
].copy()


# ============================================================
# LABEL RELATIONSHIP INFORMATION
# ============================================================

label_columns = [
    "feature",
    "Test1",
    "Test2",
    "max_abs_correlation",
    "mean_abs_correlation",
    "sign_consistency",
    "relationship_stability",
    "relationship_strength",
]

label_info = labels[label_columns].copy()

assessment = assessment.merge(
    label_info,
    on="feature",
    how="left"
)


# ============================================================
# REDUNDANCY INFORMATION
# ============================================================

# Count how many high-correlation pairs involve each feature.

redundancy_counts = {}

for _, row in redundancy.iterrows():

    f1 = row["feature_1"]
    f2 = row["feature_2"]

    redundancy_counts[f1] = redundancy_counts.get(f1, 0) + 1
    redundancy_counts[f2] = redundancy_counts.get(f2, 0) + 1


assessment["high_correlation_pair_count"] = (
    assessment["feature"]
    .map(redundancy_counts)
    .fillna(0)
    .astype(int)
)


# ============================================================
# CHECK WHETHER FEATURE HAS ANY CONFIRMED REDUNDANCY
# ============================================================

review_pairs = set()

for _, row in redundancy.iterrows():

    if row["decision"] == "REVIEW":

        review_pairs.add(row["feature_1"])
        review_pairs.add(row["feature_2"])


assessment["requires_redundancy_review"] = (
    assessment["feature"].isin(review_pairs)
)


# ============================================================
# CONSTANT STATUS
# ============================================================

def constant_status(row):

    if row["constant_both"]:
        return "Constant in both"

    if row["constant_test1"] and not row["constant_test2"]:
        return "Constant only in Test1"

    if row["constant_test2"] and not row["constant_test1"]:
        return "Constant only in Test2"

    return "Variable"


assessment["constant_status"] = assessment.apply(
    constant_status,
    axis=1
)


# ============================================================
# LABEL RELATIONSHIP STABILITY
# ============================================================

def assess_label_evidence(row):

    test1 = row["Test1"]
    test2 = row["Test2"]

    if pd.isna(test1) and pd.isna(test2):
        return "No label correlation available"

    if pd.isna(test1) or pd.isna(test2):
        return "Single-dataset evidence"

    abs1 = abs(test1)
    abs2 = abs(test2)

    difference = abs(abs1 - abs2)

    if difference <= 0.05:
        return "Consistent magnitude"

    if difference <= 0.10:
        return "Moderately consistent magnitude"

    return "Magnitude varies"


assessment["label_evidence"] = assessment.apply(
    assess_label_evidence,
    axis=1
)


# ============================================================
# DESCRIPTIVE FEATURE ASSESSMENT
# ============================================================

def assessment_category(row):

    # Constants are kept for showcase but are not informative
    # as varying ML inputs.
    if row["constant_both"]:
        return "Showcase / Configuration"

    # Strong and moderate observed label relationship.
    if row["relationship_strength"] == "Strong":
        return "High-priority candidate"

    if row["relationship_strength"] == "Moderate":
        return "Candidate for further evaluation"

    # Stable weak relationship.
    if (
        row["relationship_strength"] == "Weak"
        and row["sign_consistency"] == "Consistent sign"
    ):
        return "Potential supporting feature"

    # Very weak or unstable evidence.
    if row["relationship_strength"] == "Very Weak":
        return "Low observed label association"

    return "Requires further evaluation"


assessment["assessment"] = assessment.apply(
    assessment_category,
    axis=1
)


# ============================================================
# DO NOT MAKE FINAL KEEP/REMOVE DECISIONS
# ============================================================

assessment["ml_decision"] = "Not yet decided"


# ============================================================
# SORT
# ============================================================

assessment = assessment.sort_values(
    by=[
        "assessment",
        "max_abs_correlation"
    ],
    ascending=[
        True,
        False
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

assessment.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("HAI 23.05 FEATURE ASSESSMENT MATRIX")
print("=" * 70)

print()
print(f"Total features : {len(assessment)}")

print()
print("ASSESSMENT SUMMARY")
print("-" * 70)

print(
    assessment["assessment"]
    .value_counts()
    .to_string()
)

print()
print("CONSTANT STATUS")
print("-" * 70)

print(
    assessment["constant_status"]
    .value_counts()
    .to_string()
)

print()
print("TOP LABEL-ASSOCIATED FEATURES")
print("-" * 70)

top_features = assessment[
    [
        "feature",
        "role",
        "Test1",
        "Test2",
        "max_abs_correlation",
        "sign_consistency",
        "assessment",
    ]
].sort_values(
    "max_abs_correlation",
    ascending=False
)

print(
    top_features
    .head(15)
    .to_string(index=False)
)

print()
print("OUTPUT")
print("-" * 70)

print(OUTPUT_FILE)

print()
print("=" * 70)
print("FEATURE ASSESSMENT COMPLETE")
print("=" * 70)