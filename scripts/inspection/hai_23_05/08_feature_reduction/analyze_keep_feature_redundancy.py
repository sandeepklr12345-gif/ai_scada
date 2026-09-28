from pathlib import Path
import pandas as pd
import numpy as np


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

SELECTION_FILE = (
    FEATURE_DIR
    / "hai_2305_feature_selection_decision.csv"
)

TEST1_CORR_FILE = (
    FEATURE_DIR
    / "hai_test1_correlation_matrix.csv"
)

TEST2_CORR_FILE = (
    FEATURE_DIR
    / "hai_test2_correlation_matrix.csv"
)

OUTPUT_PAIRS = (
    FEATURE_DIR
    / "hai_2305_keep_high_correlation_pairs.csv"
)

OUTPUT_GROUPS = (
    FEATURE_DIR
    / "hai_2305_keep_correlation_groups.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CORRELATION_THRESHOLD = 0.95


# ============================================================
# LOAD FEATURE DECISIONS
# ============================================================

selection = pd.read_csv(SELECTION_FILE)

keep_features = (
    selection.loc[
        selection["decision"] == "KEEP",
        "feature"
    ]
    .tolist()
)

print("=" * 70)
print("HAI 23.05 KEEP-FEATURE REDUNDANCY ANALYSIS")
print("=" * 70)

print(f"\nTotal original features : {len(selection)}")
print(f"KEEP features           : {len(keep_features)}")
print(
    f"Correlation threshold   : "
    f"|r| >= {CORRELATION_THRESHOLD}"
)


# ============================================================
# LOAD CORRELATION MATRICES
# ============================================================

corr1 = pd.read_csv(
    TEST1_CORR_FILE,
    index_col=0
)

corr2 = pd.read_csv(
    TEST2_CORR_FILE,
    index_col=0
)

# Keep only features that are actually present
# in both correlation matrices.
available_features = [
    f for f in keep_features
    if f in corr1.index and f in corr2.index
]

print(
    f"Features available in both correlation matrices: "
    f"{len(available_features)}"
)


# ============================================================
# FIND HIGH-CORRELATION PAIRS
# ============================================================

pairs = []

for i in range(len(available_features)):

    feature_a = available_features[i]

    for j in range(i + 1, len(available_features)):

        feature_b = available_features[j]

        r1 = corr1.loc[feature_a, feature_b]
        r2 = corr2.loc[feature_a, feature_b]

        if pd.isna(r1) or pd.isna(r2):
            continue

        if (
            abs(r1) >= CORRELATION_THRESHOLD
            or
            abs(r2) >= CORRELATION_THRESHOLD
        ):

            # Average absolute correlation
            mean_abs_corr = (
                abs(r1) + abs(r2)
            ) / 2

            pairs.append({
                "feature_1": feature_a,
                "feature_2": feature_b,
                "test1_correlation": r1,
                "test2_correlation": r2,
                "mean_absolute_correlation":
                    mean_abs_corr
            })


pairs_df = pd.DataFrame(pairs)


# ============================================================
# SORT RESULTS
# ============================================================

if not pairs_df.empty:

    pairs_df = pairs_df.sort_values(
        "mean_absolute_correlation",
        ascending=False
    )

    pairs_df.to_csv(
        OUTPUT_PAIRS,
        index=False
    )

else:

    pairs_df = pd.DataFrame(
        columns=[
            "feature_1",
            "feature_2",
            "test1_correlation",
            "test2_correlation",
            "mean_absolute_correlation"
        ]
    )

    pairs_df.to_csv(
        OUTPUT_PAIRS,
        index=False
    )


# ============================================================
# BUILD CORRELATION GROUPS
# ============================================================

# Treat highly correlated features as connected components.
#
# Example:
#
# A -- B
# B -- C
#
# becomes:
#
# {A, B, C}
#
# This helps reveal larger redundancy groups rather
# than looking only at individual pairs.

parent = {
    feature: feature
    for feature in available_features
}


def find(x):

    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]

    return x


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


for _, row in pairs_df.iterrows():

    union(
        row["feature_1"],
        row["feature_2"]
    )


groups = {}

for feature in available_features:

    root = find(feature)

    if root not in groups:
        groups[root] = []

    groups[root].append(feature)


# Only show groups containing more than one feature
correlation_groups = [
    sorted(features)
    for features in groups.values()
    if len(features) > 1
]

correlation_groups.sort(
    key=lambda x: (-len(x), x)
)


# ============================================================
# SAVE GROUPS
# ============================================================

group_rows = []

for group_id, group in enumerate(
    correlation_groups,
    start=1
):

    for feature in group:

        group_rows.append({
            "correlation_group":
                f"Group_{group_id:02d}",
            "group_size":
                len(group),
            "feature":
                feature
        })


groups_df = pd.DataFrame(group_rows)

groups_df.to_csv(
    OUTPUT_GROUPS,
    index=False
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CORRELATION PAIRS")
print("=" * 70)

print(
    f"\nTotal high-correlation pairs: "
    f"{len(pairs_df)}"
)

if not pairs_df.empty:

    print(
        pairs_df.head(30).to_string(
            index=False
        )
    )

else:

    print("No high-correlation pairs found.")


print("\n" + "=" * 70)
print("CORRELATION GROUPS")
print("=" * 70)

print(
    f"\nTotal correlation groups: "
    f"{len(correlation_groups)}"
)

for group_id, group in enumerate(
    correlation_groups,
    start=1
):

    print(
        f"\nGroup {group_id} "
        f"({len(group)} features):"
    )

    for feature in group:

        print(f"  - {feature}")


# ============================================================
# FEATURES NOT IN ANY REDUNDANCY GROUP
# ============================================================

grouped_features = set(
    feature
    for group in correlation_groups
    for feature in group
)

independent_features = [
    feature
    for feature in available_features
    if feature not in grouped_features
]


print("\n" + "=" * 70)
print("FEATURES WITHOUT HIGH-CORRELATION PARTNERS")
print("=" * 70)

print(
    f"\nIndependent KEEP features: "
    f"{len(independent_features)}"
)

for feature in independent_features:

    print(f"  - {feature}")


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(OUTPUT_PAIRS)
print(OUTPUT_GROUPS)

print("\n" + "=" * 70)
print("HAI 23.05 KEEP-FEATURE REDUNDANCY ANALYSIS COMPLETE")
print("=" * 70)