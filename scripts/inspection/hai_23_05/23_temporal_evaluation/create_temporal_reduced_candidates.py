from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# STAGE 23F
# HAI 23.05 - Training-Only Temporal Feature Reduction
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TEMPORAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
)

ASSOCIATION_FILE = (
    TEMPORAL_DIR
    / "isolation_forest"
    / "hai_2305_temporal_feature_score_association.csv"
)

SCHEMA_FILE = (
    TEMPORAL_DIR
    / "hai_2305_temporal_model_ready_schema.csv"
)

OUTPUT_DIR = (
    TEMPORAL_DIR
    / "reduced_candidates"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_reduced_candidate_manifest.csv"
)

FEATURE_LIST_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_reduced_candidate_features.txt"
)


print("=" * 70)
print("STAGE 23F: TRAINING-ONLY TEMPORAL FEATURE REDUCTION")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINING-ONLY ASSOCIATION RESULTS
# ============================================================

print(
    "\nLoading Stage 23D training-only associations..."
)

association_df = pd.read_csv(
    ASSOCIATION_FILE
)

print(
    f"Rows: {len(association_df)}"
)

assert len(
    association_df
) == 232


# ============================================================
# 2. LOAD MODEL-READY SCHEMA
# ============================================================

print(
    "\nLoading temporal model-ready schema..."
)

schema_df = pd.read_csv(
    SCHEMA_FILE
)

schema_columns = (
    schema_df["feature"]
    .tolist()
    if "feature" in schema_df.columns
    else schema_df.iloc[:, 0].tolist()
)


# The schema should contain timestamp + 232 features.
schema_features = [
    col
    for col in schema_columns
    if col != "timestamp"
]

assert len(
    schema_features
) == 232


# ============================================================
# 3. IDENTIFY FEATURE FAMILIES
# ============================================================

original_features = [
    f
    for f in schema_features
    if "__" not in f
]

diff_features = [
    f
    for f in schema_features
    if f.endswith("__diff_1s")
]

abs_diff_features = [
    f
    for f in schema_features
    if f.endswith("__abs_diff_1s")
]

rolling_features = [
    f
    for f in schema_features
    if f.endswith("__rolling_std_5s")
]


assert len(original_features) == 58
assert len(diff_features) == 58
assert len(abs_diff_features) == 58
assert len(rolling_features) == 58


print(
    "\nFeature families:"
)

print(
    f"Original          : {len(original_features)}"
)

print(
    f"diff_1s           : {len(diff_features)}"
)

print(
    f"abs_diff_1s       : {len(abs_diff_features)}"
)

print(
    f"rolling_std_5s    : {len(rolling_features)}"
)


# ============================================================
# 4. RANK TEMPORAL FEATURES
# ============================================================

print(
    "\nRanking temporal features using training-only "
    "Spearman association..."
)

# Only temporal features are ranked.
temporal_assoc = association_df[
    association_df[
        "feature_family"
    ].isin(
        [
            "diff_1s",
            "abs_diff_1s",
            "rolling_std_5s"
        ]
    )
].copy()


# Rank independently within each family.
temporal_assoc[
    "family_rank"
] = (
    temporal_assoc
    .groupby(
        "feature_family"
    )[
        "abs_spearman_r"
    ]
    .rank(
        method="first",
        ascending=False
    )
)


# ============================================================
# 5. GET TOP-K FEATURES
# ============================================================

def top_k(
    family,
    k
):

    subset = (
        temporal_assoc[
            temporal_assoc[
                "feature_family"
            ] == family
        ]
        .sort_values(
            [
                "abs_spearman_r",
                "abs_pearson_r"
            ],
            ascending=False
        )
    )

    return (
        subset[
            "feature"
        ]
        .head(k)
        .tolist()
    )


# ============================================================
# 6. CREATE CANDIDATES
# ============================================================

candidate_features = {}

candidate_features[
    "candidate_A"
] = (
    original_features
    + top_k(
        "abs_diff_1s",
        10
    )
    + top_k(
        "rolling_std_5s",
        10
    )
)


candidate_features[
    "candidate_B"
] = (
    original_features
    + top_k(
        "abs_diff_1s",
        20
    )
    + top_k(
        "rolling_std_5s",
        20
    )
)


candidate_features[
    "candidate_C"
] = (
    original_features
    + top_k(
        "abs_diff_1s",
        30
    )
    + top_k(
        "rolling_std_5s",
        30
    )
)


candidate_features[
    "candidate_D"
] = (
    original_features
    + top_k(
        "diff_1s",
        10
    )
    + top_k(
        "abs_diff_1s",
        10
    )
    + top_k(
        "rolling_std_5s",
        10
    )
)


candidate_features[
    "candidate_E"
] = (
    original_features
    + top_k(
        "diff_1s",
        20
    )
    + top_k(
        "abs_diff_1s",
        20
    )
    + top_k(
        "rolling_std_5s",
        20
    )
)


# ============================================================
# 7. REMOVE ACCIDENTAL DUPLICATES
# ============================================================

for name in candidate_features:

    candidate_features[name] = list(
        dict.fromkeys(
            candidate_features[name]
        )
    )


# ============================================================
# 8. VALIDATE CANDIDATES
# ============================================================

print(
    "\nCandidate feature sets:"
)

manifest_rows = []

for name, features in candidate_features.items():

    assert (
        len(features)
        == len(set(features))
    )

    assert all(
        f in schema_features
        for f in features
    )

    assert (
        original_features
        == [
            f
            for f in features
            if "__" not in f
        ]
    )

    family_counts = {
        "original": sum(
            "__" not in f
            for f in features
        ),
        "diff_1s": sum(
            f.endswith(
                "__diff_1s"
            )
            for f in features
        ),
        "abs_diff_1s": sum(
            f.endswith(
                "__abs_diff_1s"
            )
            for f in features
        ),
        "rolling_std_5s": sum(
            f.endswith(
                "__rolling_std_5s"
            )
            for f in features
        )
    }

    print(
        f"\n{name}:"
    )

    print(
        f"Total features : {len(features)}"
    )

    print(
        f"Original       : "
        f"{family_counts['original']}"
    )

    print(
        f"diff_1s        : "
        f"{family_counts['diff_1s']}"
    )

    print(
        f"abs_diff_1s    : "
        f"{family_counts['abs_diff_1s']}"
    )

    print(
        f"rolling_std_5s : "
        f"{family_counts['rolling_std_5s']}"
    )

    manifest_rows.append(
        {
            "candidate": name,
            "total_features": len(features),
            "original_features":
                family_counts[
                    "original"
                ],
            "diff_1s_features":
                family_counts[
                    "diff_1s"
                ],
            "abs_diff_1s_features":
                family_counts[
                    "abs_diff_1s"
                ],
            "rolling_std_5s_features":
                family_counts[
                    "rolling_std_5s"
                ],
            "selection_source":
                "Stage 23D training-only "
                "score association",
            "test_labels_used":
                False
        }
    )


# ============================================================
# 9. CREATE FULL REFERENCE SET
# ============================================================

candidate_features[
    "full_temporal_reference"
] = schema_features.copy()

assert len(
    candidate_features[
        "full_temporal_reference"
    ]
) == 232

manifest_rows.append(
    {
        "candidate":
            "full_temporal_reference",
        "total_features":
            232,
        "original_features":
            58,
        "diff_1s_features":
            58,
        "abs_diff_1s_features":
            58,
        "rolling_std_5s_features":
            58,
        "selection_source":
            "Full Stage 22H representation",
        "test_labels_used":
            False
    }
)


# ============================================================
# 10. SAVE MANIFEST
# ============================================================

manifest_df = pd.DataFrame(
    manifest_rows
)

manifest_df.to_csv(
    MANIFEST_FILE,
    index=False
)


# ============================================================
# 11. SAVE FEATURE LISTS
# ============================================================

with open(
    FEATURE_LIST_FILE,
    "w",
    encoding="utf-8"
) as f:

    for name, features in (
        candidate_features.items()
    ):

        f.write(
            f"\n{'=' * 70}\n"
        )

        f.write(
            f"{name}\n"
        )

        f.write(
            f"{'=' * 70}\n"
        )

        f.write(
            f"Feature count: "
            f"{len(features)}\n\n"
        )

        for feature in features:

            f.write(
                feature
                + "\n"
            )


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23F FINAL VALIDATION"
)

print(
    "=" * 70
)

assert len(
    candidate_features
) == 6

assert len(
    manifest_df
) == 6

expected_counts = {
    "candidate_A": 78,
    "candidate_B": 98,
    "candidate_C": 118,
    "candidate_D": 88,
    "candidate_E": 118,
    "full_temporal_reference": 232
}

for name, expected in (
    expected_counts.items()
):

    assert (
        len(
            candidate_features[name]
        )
        == expected
    )


assert Path(
    MANIFEST_FILE
).exists()

assert Path(
    FEATURE_LIST_FILE
).exists()

print(
    "PASS: Six candidate sets created"
)

print(
    "PASS: All candidates retain 58 original features"
)

print(
    "PASS: Temporal features selected from training-only evidence"
)

print(
    "PASS: No Test 1 labels used"
)

print(
    "PASS: No Test 2 labels used"
)

print(
    "PASS: Full 232-feature representation preserved"
)

print(
    "PASS: No duplicate features"
)

print(
    "PASS: All selected features exist in Stage 22H schema"
)

print(
    "\nCandidate manifest:"
)

print(
    manifest_df.to_string(
        index=False
    )
)

print(
    "\nManifest:"
)

print(
    MANIFEST_FILE
)

print(
    "\nFeature lists:"
)

print(
    FEATURE_LIST_FILE
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23F: COMPLETE"
)

print(
    "=" * 70
)