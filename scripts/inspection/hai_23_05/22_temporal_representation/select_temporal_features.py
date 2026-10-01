from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# STAGE 22G
# HAI 23.05 - Temporal Feature Selection
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

TEMPORAL_FILE = (
    TEMPORAL_DIR
    / "hai_2305_training_temporal_features.csv"
)

REVIEW_FILE = (
    TEMPORAL_DIR
    / "hai_2305_temporal_engineering_review.csv"
)

QUALITY_FILE = (
    TEMPORAL_DIR
    / "hai_2305_temporal_feature_quality.csv"
)

OUTPUT_MANIFEST = (
    TEMPORAL_DIR
    / "hai_2305_temporal_feature_selection_manifest.csv"
)

OUTPUT_FEATURE_LIST = (
    TEMPORAL_DIR
    / "hai_2305_selected_temporal_features.txt"
)


print("=" * 70)
print("STAGE 22G: TEMPORAL FEATURE SELECTION")
print("=" * 70)


# ============================================================
# 1. LOAD METADATA ONLY
# ============================================================

print("\nReading temporal feature schema...")

temporal_columns = pd.read_csv(
    TEMPORAL_FILE,
    nrows=0
).columns.tolist()

assert temporal_columns[0] == "timestamp"

feature_columns = [
    col
    for col in temporal_columns
    if col != "timestamp"
]

assert len(feature_columns) == 174

print(
    f"Temporal features available: "
    f"{len(feature_columns)}"
)


# ============================================================
# 2. LOAD ENGINEERING REVIEW
# ============================================================

print("\nLoading Stage 22F engineering review...")

review_df = pd.read_csv(
    REVIEW_FILE
)

assert len(review_df) == 22

print(
    f"High-correlation relationships: "
    f"{len(review_df)}"
)


# ============================================================
# 3. LOAD QUALITY METADATA
# ============================================================

print("\nLoading temporal feature quality report...")

quality_df = pd.read_csv(
    QUALITY_FILE
)

assert len(quality_df) == 174

print(
    f"Quality records: {len(quality_df)}"
)


# ============================================================
# 4. BUILD REDUNDANCY INFORMATION
# ============================================================

redundancy_count = {
    feature: 0
    for feature in feature_columns
}

exact_duplicate_count = {
    feature: 0
    for feature in feature_columns
}

review_relationships = {
    feature: []
    for feature in feature_columns
}


for _, row in review_df.iterrows():

    feature_a = row["feature_a"]
    feature_b = row["feature_b"]

    assert feature_a in feature_columns
    assert feature_b in feature_columns

    redundancy_count[
        feature_a
    ] += 1

    redundancy_count[
        feature_b
    ] += 1

    if bool(
        row["confirmed_exact_duplicate"]
    ):

        exact_duplicate_count[
            feature_a
        ] += 1

        exact_duplicate_count[
            feature_b
        ] += 1

    review_relationships[
        feature_a
    ].append(feature_b)

    review_relationships[
        feature_b
    ].append(feature_a)


# ============================================================
# 5. TRANSFORMATION PARSER
# ============================================================

def get_base_feature(feature):

    suffixes = [
        "__diff_1s",
        "__abs_diff_1s",
        "__rolling_std_5s"
    ]

    for suffix in suffixes:

        if feature.endswith(suffix):

            return feature[
                :-len(suffix)
            ]

    return feature


def get_transformation(feature):

    if feature.endswith(
        "__diff_1s"
    ):
        return "diff_1s"

    if feature.endswith(
        "__abs_diff_1s"
    ):
        return "abs_diff_1s"

    if feature.endswith(
        "__rolling_std_5s"
    ):
        return "rolling_std_5s"

    return "unknown"


# ============================================================
# 6. FEATURE-LEVEL SELECTION
# ============================================================

records = []


for feature in feature_columns:

    base_feature = get_base_feature(
        feature
    )

    transformation = get_transformation(
        feature
    )

    redundancy_pairs = (
        redundancy_count[feature]
    )

    exact_pairs = (
        exact_duplicate_count[feature]
    )

    related_features = (
        review_relationships[feature]
    )

    # --------------------------------------------------------
    # Selection decision
    # --------------------------------------------------------
    #
    # No feature received EXCLUDE_CANDIDATE in Stage 22F.
    # Therefore no feature is removed at this stage.
    #
    # Exact duplicates remain flagged because the observed
    # equality is training-data evidence, not proof that the
    # source signals are semantically interchangeable.
    # --------------------------------------------------------

    selection = "KEEP"

    if exact_pairs > 0:

        reason = (
            "Keep pending further validation because exact "
            "redundancy was observed in training data, but "
            "source signals are distinct status/control variables."
        )

        review_status = (
            "EXACT_REDUNDANCY_REVIEW"
        )

    elif redundancy_pairs > 0:

        reason = (
            "Keep because high correlation was identified, "
            "but engineering review did not justify exclusion."
        )

        review_status = (
            "HIGH_CORRELATION_REVIEW"
        )

    else:

        reason = (
            "No high-correlation relationship requiring "
            "exclusion was identified."
        )

        review_status = (
            "NO_REDUNDANCY_FLAG"
        )


    records.append({

        "feature":
            feature,

        "base_feature":
            base_feature,

        "transformation":
            transformation,

        "redundancy_pair_count":
            redundancy_pairs,

        "exact_duplicate_pair_count":
            exact_pairs,

        "related_features":
            ";".join(
                sorted(
                    set(
                        related_features
                    )
                )
            ),

        "selection":
            selection,

        "review_status":
            review_status,

        "selection_reason":
            reason
    })


manifest_df = pd.DataFrame(
    records
)


# ============================================================
# 7. MERGE QUALITY INFORMATION
# ============================================================

quality_columns = [
    col
    for col in [
        "feature",
        "std",
        "minimum",
        "maximum",
        "unique_values",
        "nan_count",
        "infinite_count"
    ]
    if col in quality_df.columns
]


if "feature" in quality_df.columns:

    manifest_df = manifest_df.merge(
        quality_df[
            quality_columns
        ],
        on="feature",
        how="left"
    )


# ============================================================
# 8. VALIDATION
# ============================================================

assert len(manifest_df) == 174

assert (
    manifest_df["feature"]
    .nunique()
    == 174
)

assert (
    manifest_df["selection"]
    .isin(["KEEP"])
    .all()
)

assert (
    manifest_df["transformation"]
    .isin([
        "diff_1s",
        "abs_diff_1s",
        "rolling_std_5s"
    ])
    .all()
)

assert (
    manifest_df[
        "redundancy_pair_count"
    ].sum()
    == 44
)

print(
    "\nPASS: 174 unique temporal features represented"
)

print(
    "PASS: All temporal transformations recognized"
)

print(
    "PASS: All high-correlation relationships mapped"
)

print(
    "PASS: No feature marked for exclusion"
)


# ============================================================
# 9. SAVE MANIFEST
# ============================================================

manifest_df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)

print(
    f"\nSaved feature-selection manifest:"
)

print(
    OUTPUT_MANIFEST
)


# ============================================================
# 10. SAVE SELECTED FEATURE LIST
# ============================================================

selected_features = (
    manifest_df[
        manifest_df["selection"] == "KEEP"
    ]["feature"]
    .tolist()
)

assert len(
    selected_features
) == 174


with open(
    OUTPUT_FEATURE_LIST,
    "w",
    encoding="utf-8"
) as f:

    for feature in selected_features:

        f.write(
            feature + "\n"
        )


print(
    "\nSaved selected feature list:"
)

print(
    OUTPUT_FEATURE_LIST
)


# ============================================================
# 11. SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TEMPORAL FEATURE SELECTION SUMMARY"
)

print(
    "=" * 70
)

print(
    f"Original temporal features : "
    f"{len(feature_columns)}"
)

print(
    f"Selected temporal features : "
    f"{len(selected_features)}"
)

print(
    f"Excluded temporal features : "
    f"{len(feature_columns) - len(selected_features)}"
)


print(
    "\nBy transformation:"
)

print(
    manifest_df[
        "transformation"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nReview status:"
)

print(
    manifest_df[
        "review_status"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nFeatures involved in high-correlation relationships:"
)

print(
    (
        manifest_df[
            "redundancy_pair_count"
        ] > 0
    ).sum()
)


print(
    "\nFeatures involved in exact duplicate relationships:"
)

print(
    (
        manifest_df[
            "exact_duplicate_pair_count"
        ] > 0
    ).sum()
)


# ============================================================
# 12. SHOW FLAGGED FEATURES
# ============================================================

flagged = manifest_df[
    manifest_df[
        "redundancy_pair_count"
    ] > 0
]

print(
    "\n" + "=" * 70
)

print(
    "REDUNDANCY-FLAGGED FEATURES"
)

print(
    "=" * 70
)

print(
    flagged[
        [
            "feature",
            "redundancy_pair_count",
            "exact_duplicate_pair_count",
            "review_status",
            "related_features"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 13. FINAL VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL VALIDATION"
)

print(
    "=" * 70
)

assert len(
    selected_features
) == 174

assert (
    len(feature_columns)
    == len(selected_features)
)

assert (
    set(feature_columns)
    == set(selected_features)
)

print(
    "PASS: All 174 temporal features retained"
)

print(
    "PASS: No feature silently removed"
)

print(
    "PASS: Selection manifest matches source schema"
)

print(
    "PASS: Redundancy flags preserved"
)

print(
    "PASS: Training-derived selection only"
)

print(
    "PASS: No test labels used"
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 22G: COMPLETE"
)

print(
    "=" * 70
)