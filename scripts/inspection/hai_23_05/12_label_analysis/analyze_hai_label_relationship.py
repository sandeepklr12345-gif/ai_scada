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

TEST1_FILE = FEATURE_DIR / "hai_test1_label_correlations.csv"
TEST2_FILE = FEATURE_DIR / "hai_test2_label_correlations.csv"

ROLE_FILE = FEATURE_DIR / "hai_2305_feature_roles.csv"

OUTPUT_FILE = FEATURE_DIR / "hai_2305_label_relationship_analysis.csv"


# ============================================================
# LOAD
# ============================================================

test1 = pd.read_csv(TEST1_FILE)
test2 = pd.read_csv(TEST2_FILE)
roles = pd.read_csv(ROLE_FILE)


# ============================================================
# NORMALIZE CORRELATION FILES
# ============================================================

def normalize(df, dataset_name):

    feature_col = None
    corr_col = None

    for col in df.columns:
        name = col.lower()

        if "feature" in name or "variable" in name:
            feature_col = col

        if "corr" in name or name in ["r", "pearson_r"]:
            corr_col = col

    if feature_col is None:
        raise ValueError(
            f"Could not identify feature column: {df.columns.tolist()}"
        )

    if corr_col is None:
        raise ValueError(
            f"Could not identify correlation column: {df.columns.tolist()}"
        )

    result = df[
        [feature_col, corr_col]
    ].copy()

    result.columns = [
        "feature",
        dataset_name
    ]

    return result


test1 = normalize(test1, "Test1")
test2 = normalize(test2, "Test2")


# ============================================================
# COMBINE
# ============================================================

combined = pd.merge(
    test1,
    test2,
    on="feature",
    how="outer"
)


# ============================================================
# ROLE INFORMATION
# ============================================================

role_lookup = dict(
    zip(
        roles["feature"],
        roles["role"]
    )
)

combined["role"] = combined["feature"].map(
    lambda x: role_lookup.get(x, "Unknown")
)


# ============================================================
# ABSOLUTE CORRELATION
# ============================================================

combined["abs_Test1"] = combined["Test1"].abs()
combined["abs_Test2"] = combined["Test2"].abs()


# ============================================================
# MAXIMUM OBSERVED RELATIONSHIP
# ============================================================

combined["max_abs_correlation"] = combined[
    ["abs_Test1", "abs_Test2"]
].max(axis=1)


# ============================================================
# AVERAGE ABSOLUTE CORRELATION
# ============================================================

combined["mean_abs_correlation"] = combined[
    ["abs_Test1", "abs_Test2"]
].mean(axis=1)


# ============================================================
# SIGN CONSISTENCY
# ============================================================

def sign_consistency(row):

    a = row["Test1"]
    b = row["Test2"]

    if pd.isna(a) or pd.isna(b):
        return "One dataset unavailable"

    if a == 0 or b == 0:
        return "Weak/zero relationship"

    if (a > 0 and b > 0) or (a < 0 and b < 0):
        return "Consistent sign"

    return "Opposite signs"


combined["sign_consistency"] = combined.apply(
    sign_consistency,
    axis=1
)


# ============================================================
# INFORMATION STABILITY
# ============================================================

def stability(row):

    a = row["abs_Test1"]
    b = row["abs_Test2"]

    if pd.isna(a) or pd.isna(b):
        return "Single-dataset evidence"

    difference = abs(a - b)

    if difference <= 0.05:
        return "Stable"

    if difference <= 0.10:
        return "Moderately stable"

    return "Variable"


combined["relationship_stability"] = combined.apply(
    stability,
    axis=1
)


# ============================================================
# DESCRIPTIVE STRENGTH CATEGORY
# ============================================================

def strength(value):

    if pd.isna(value):
        return "Unavailable"

    value = abs(value)

    if value >= 0.30:
        return "Strong"
    elif value >= 0.10:
        return "Moderate"
    elif value >= 0.05:
        return "Weak"
    else:
        return "Very Weak"


combined["relationship_strength"] = combined[
    "max_abs_correlation"
].map(strength)


# ============================================================
# SORT
# ============================================================

combined = combined.sort_values(
    by="max_abs_correlation",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

output = combined[
    [
        "feature",
        "role",
        "Test1",
        "Test2",
        "abs_Test1",
        "abs_Test2",
        "max_abs_correlation",
        "mean_abs_correlation",
        "sign_consistency",
        "relationship_stability",
        "relationship_strength",
    ]
]

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("HAI 23.05 LABEL RELATIONSHIP ANALYSIS")
print("=" * 70)

print()
print(f"Total features analysed : {len(output)}")

print()
print("TEST 1")
print("-" * 70)

print(
    output[
        [
            "feature",
            "role",
            "Test1"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print()
print("TEST 2")
print("-" * 70)

print(
    output[
        [
            "feature",
            "role",
            "Test2"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print()
print("RELATIONSHIP STRENGTH")
print("-" * 70)

print(
    output["relationship_strength"]
    .value_counts()
    .to_string()
)

print()
print("SIGN CONSISTENCY")
print("-" * 70)

print(
    output["sign_consistency"]
    .value_counts()
    .to_string()
)

print()
print("OUTPUT")
print("-" * 70)

print(OUTPUT_FILE)

print()
print("=" * 70)
print("LABEL RELATIONSHIP ANALYSIS COMPLETE")
print("=" * 70)