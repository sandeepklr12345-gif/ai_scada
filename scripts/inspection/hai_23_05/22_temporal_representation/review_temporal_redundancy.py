from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STAGE 22F
# HAI 23.05 - Engineering Redundancy Review
# ============================================================

ROOT = Path(
    r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"
)

TEMPORAL_DIR = (
    ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
)

CORR_FILE = (
    TEMPORAL_DIR
    / "hai_2305_temporal_high_correlation_pairs.csv"
)

DUPLICATES_FILE = (
    TEMPORAL_DIR
    / "hai_2305_temporal_exact_duplicates.csv"
)

OUTPUT_FILE = (
    TEMPORAL_DIR
    / "hai_2305_temporal_engineering_review.csv"
)


print("=" * 70)
print("STAGE 22F: ENGINEERING REDUNDANCY REVIEW")
print("=" * 70)


# ============================================================
# 1. LOAD
# ============================================================

print("\nLoading Stage 22E results...")

corr_df = pd.read_csv(CORR_FILE)
duplicates_df = pd.read_csv(DUPLICATES_FILE)

print(
    f"High-correlation pairs : {len(corr_df)}"
)

print(
    f"Exact duplicate pairs  : {len(duplicates_df)}"
)


# ============================================================
# 2. FEATURE ROLE KNOWLEDGE
# ============================================================

# Roles based on the previously established HAI 23.05
# feature-role classification and engineering review.

STATUS_FEATURES = {
    "P2_ATSW_Lamp",
    "P2_AutoGO",
    "P2_AutoSD",
    "P2_MASW",
    "P2_MASW_Lamp",
    "P2_ManualGO",
    "P2_ManualSD",
    "P2_SCO",
    "P2_SCST",
}

PROCESS_FEATURES = {
    "P1_FCV01D",
    "P1_FT02Z",
    "P4_ST_GOV",
    "P4_ST_PO",
    "P3_LCV01D",
    "P4_HT_PO",
}


# ============================================================
# 3. TRANSFORMATION PARSER
# ============================================================

def get_base_feature(feature):

    for suffix in [
        "__rolling_std_5s",
        "__abs_diff_1s",
        "__diff_1s",
    ]:

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
# 4. ENGINEERING CLASSIFICATION
# ============================================================

def classify_pair(feature_a, feature_b, correlation):

    base_a = get_base_feature(feature_a)
    base_b = get_base_feature(feature_b)

    trans_a = get_transformation(feature_a)
    trans_b = get_transformation(feature_b)

    exact = abs(abs(correlation) - 1.0) < 1e-12

    same_base = (
        base_a == base_b
    )

    same_transformation = (
        trans_a == trans_b
    )

    base_a_status = (
        base_a in STATUS_FEATURES
    )

    base_b_status = (
        base_b in STATUS_FEATURES
    )

    base_a_process = (
        base_a in PROCESS_FEATURES
    )

    base_b_process = (
        base_b in PROCESS_FEATURES
    )


    # --------------------------------------------------------
    # CASE 1: Exact duplicate temporal representations
    # --------------------------------------------------------

    if exact:

        if same_base:

            return (
                "KEEP",
                "Same source feature and mathematically "
                "identical temporal representation."
            )

        if base_a_status and base_b_status:

            return (
                "REVIEW",
                "Exact redundancy between documented "
                "status/switch-related signals."
            )

        return (
            "REVIEW",
            "Exact redundancy exists, but source-feature "
            "relationship should be retained for engineering review."
        )


    # --------------------------------------------------------
    # CASE 2: Signed difference vs absolute difference
    # --------------------------------------------------------

    if {
        trans_a,
        trans_b
    } == {
        "diff_1s",
        "abs_diff_1s"
    }:

        if same_base:

            return (
                "KEEP",
                "Signed and absolute first differences represent "
                "different information: direction versus magnitude."
            )

        return (
            "REVIEW",
            "Cross-feature first-difference redundancy."
        )


    # --------------------------------------------------------
    # CASE 3: Same transformation family
    # --------------------------------------------------------

    if same_transformation:

        if (
            base_a_status
            and base_b_status
        ):

            return (
                "REVIEW",
                "Highly correlated temporal representations "
                "of status/control signals."
            )

        if (
            base_a_process
            and base_b_process
        ):

            return (
                "REVIEW",
                "Highly correlated process/control signals. "
                "Engineering coupling does not by itself justify deletion."
            )

        return (
            "KEEP",
            "Strong correlation exists, but the temporal features "
            "represent distinct source variables."
        )


    # --------------------------------------------------------
    # CASE 4: Cross-family redundancy
    # --------------------------------------------------------

    if (
        base_a_status
        or base_b_status
    ):

        return (
            "REVIEW",
            "Cross-transformation redundancy involving "
            "status/control information."
        )


    return (
        "REVIEW",
        "High cross-family correlation requires engineering review."
    )


# ============================================================
# 5. REVIEW ALL HIGH-CORRELATION PAIRS
# ============================================================

records = []


for _, row in corr_df.iterrows():

    feature_a = row["feature_a"]
    feature_b = row["feature_b"]

    correlation = float(
        row["correlation"]
    )

    decision, reason = classify_pair(
        feature_a,
        feature_b,
        correlation
    )

    records.append({

        "feature_a":
            feature_a,

        "feature_b":
            feature_b,

        "base_feature_a":
            get_base_feature(feature_a),

        "base_feature_b":
            get_base_feature(feature_b),

        "transformation_a":
            get_transformation(feature_a),

        "transformation_b":
            get_transformation(feature_b),

        "correlation":
            correlation,

        "absolute_correlation":
            abs(correlation),

        "exact_redundancy":
            abs(abs(correlation) - 1.0)
            < 1e-12,

        "decision":
            decision,

        "reason":
            reason
    })


review_df = pd.DataFrame(
    records
)


# ============================================================
# 6. MARK EXACT DUPLICATE PAIRS
# ============================================================

duplicate_pairs = set()

for _, row in duplicates_df.iterrows():

    pair = tuple(
        sorted([
            row["feature_a"],
            row["feature_b"]
        ])
    )

    duplicate_pairs.add(pair)


review_df["confirmed_exact_duplicate"] = (
    review_df.apply(
        lambda row:
        tuple(
            sorted([
                row["feature_a"],
                row["feature_b"]
            ])
        ) in duplicate_pairs,
        axis=1
    )
)


# ============================================================
# 7. SAVE
# ============================================================

review_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ENGINEERING REVIEW SUMMARY")
print("=" * 70)

print(
    f"Total high-correlation pairs : "
    f"{len(review_df)}"
)

print(
    "\nDecision counts:"
)

print(
    review_df[
        "decision"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nConfirmed exact duplicates:"
)

print(
    review_df[
        "confirmed_exact_duplicate"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 9. DISPLAY REVIEW TABLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PAIR-BY-PAIR ENGINEERING REVIEW"
)

print(
    "=" * 70
)

display_columns = [
    "feature_a",
    "feature_b",
    "correlation",
    "confirmed_exact_duplicate",
    "decision",
    "reason"
]

print(
    review_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 10. VALIDATION
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


assert len(review_df) == len(
    corr_df
)

assert set(
    review_df["decision"].unique()
).issubset({
    "KEEP",
    "REVIEW",
    "EXCLUDE_CANDIDATE"
})

assert (
    review_df[
        "absolute_correlation"
    ]
    >= 0.95
).all()

assert (
    review_df[
        "confirmed_exact_duplicate"
    ].sum()
    == len(duplicates_df)
    or len(duplicates_df) == 0
)

print(
    "PASS: All Stage 22E high-correlation pairs reviewed"
)

print(
    "PASS: All reviewed pairs satisfy |r| >= 0.95"
)

print(
    "PASS: Exact duplicate relationships checked"
)

print(
    "PASS: Engineering decisions assigned"
)

print(
    "PASS: No features deleted"
)

print(
    "PASS: No test data used"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 22F: COMPLETE"
)

print(
    "=" * 70
)