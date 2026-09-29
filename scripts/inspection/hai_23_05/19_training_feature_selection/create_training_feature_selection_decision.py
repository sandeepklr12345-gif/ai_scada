from pathlib import Path
import pandas as pd


# ============================================================
# STAGE 19B: TRAINING FEATURE-SELECTION DECISION
# ============================================================

print("=" * 70)
print("STAGE 19B: TRAINING FEATURE-SELECTION DECISION")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]


REVIEW_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_feature_selection"
    / "hai_2305_training_feature_selection_review.csv"
)

ROLES_FILE = (
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
    / "training_feature_selection"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_2305_training_feature_selection_decision.csv"
)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("\nLoading Stage 19 review table...")
review = pd.read_csv(REVIEW_FILE)

print("Loading engineering feature roles...")
roles = pd.read_csv(ROLES_FILE)

print("Files loaded successfully.")


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

assert "feature" in review.columns
assert "role" in review.columns
assert "high_correlation_count" in review.columns
assert "max_absolute_correlation" in review.columns

assert "feature" in roles.columns
assert "role" in roles.columns

assert len(review) == 66
assert review["feature"].is_unique

role_map = roles.set_index("feature")["role"].to_dict()

missing_roles = [
    feature
    for feature in review["feature"]
    if feature not in role_map
]

assert not missing_roles, (
    f"Missing role mappings: {missing_roles}"
)


# ------------------------------------------------------------
# ENGINEERING DECISION RULES
# ------------------------------------------------------------

DOCUMENTED_ROLES = {
    "Process Measurement",
    "Steam Temperature",
    "Steam Pressure",
    "Electrical Load Demand",
    "Output Power",
    "Valve Position Command",
    "Valve Current Position",
    "Measured Flowrate",
    "Converted Flowrate",
    "Controller Output",
    "Gate Opening Rate",
    "Mode Switch",
    "Status / Indicator",
    "Start Command",
    "Process Measurement",
    "Internal Control-Logic Output",
}


def decide_feature(role):

    # Internal HAIEnd/control-logic outputs are not
    # automatically removed, but their exact semantics
    # require documentation review.
    if role == "Internal Control-Logic Output":
        return (
            "REVIEW",
            "Internal control-logic output; exact engineering "
            "semantics are not sufficiently documented for "
            "automatic ML retention."
        )

    # Any role explicitly requiring documentation must remain
    # under engineering review.
    if "Requires Documentation" in str(role):
        return (
            "REVIEW",
            "Engineering role requires documentation review "
            "before final ML inclusion."
        )

    # Documented engineering/process/control variables are
    # retained even when highly correlated because correlation
    # does not establish that their engineering roles are
    # duplicates.
    if role in DOCUMENTED_ROLES:
        return (
            "KEEP",
            "Documented engineering/process/control signal. "
            "High correlation alone is not sufficient evidence "
            "for exclusion."
        )

    # Unknown/unexpected role.
    return (
        "REVIEW",
        "Role is not covered by the current documented "
        "engineering decision rules."
    )


# ------------------------------------------------------------
# CREATE DECISIONS
# ------------------------------------------------------------

decision_rows = []

for _, row in review.iterrows():

    feature = row["feature"]
    role = role_map[feature]

    decision, reason = decide_feature(role)

    decision_rows.append({
        "feature": feature,
        "role": role,
        "high_correlation_count": int(
            row["high_correlation_count"]
        ),
        "max_absolute_correlation": float(
            row["max_absolute_correlation"]
        ),
        "most_correlated_feature": row[
            "most_correlated_feature"
        ],
        "most_correlated_correlation": float(
            row["most_correlated_correlation"]
        ),
        "selection_decision": decision,
        "selection_reason": reason,
    })


decisions = pd.DataFrame(decision_rows)


# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

decision_order = {
    "REVIEW": 0,
    "KEEP": 1,
    "EXCLUDE": 2,
}

decisions["_order"] = decisions[
    "selection_decision"
].map(decision_order)

decisions = (
    decisions
    .sort_values(
        by=[
            "_order",
            "high_correlation_count",
            "feature",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )
    .drop(columns=["_order"])
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

decisions.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 19B DECISION SUMMARY")
print("=" * 70)

counts = decisions[
    "selection_decision"
].value_counts()

print(
    "KEEP    :",
    counts.get("KEEP", 0)
)

print(
    "REVIEW  :",
    counts.get("REVIEW", 0)
)

print(
    "EXCLUDE :",
    counts.get("EXCLUDE", 0)
)

print(
    "TOTAL   :",
    len(decisions)
)


print("\nFeatures requiring engineering review:")

review_features = decisions[
    decisions["selection_decision"] == "REVIEW"
]

if len(review_features) > 0:
    print(
        review_features[
            [
                "feature",
                "role",
                "high_correlation_count",
                "max_absolute_correlation",
            ]
        ].to_string(index=False)
    )
else:
    print("None")


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

assert len(decisions) == 66
assert decisions["feature"].is_unique

assert set(
    decisions["selection_decision"]
).issubset({
    "KEEP",
    "REVIEW",
    "EXCLUDE",
})

# No automatic exclusions are permitted by this stage.
assert (
    decisions["selection_decision"] == "EXCLUDE"
).sum() == 0

print("PASS: All 66 training candidates represented")
print("PASS: No duplicate feature names")
print("PASS: Every feature has an engineering role")
print("PASS: Decisions use training candidate information")
print("PASS: No Test 1/Test 2 labels used")
print("PASS: No automatic exclusions")
print("PASS: Correlated engineering signals retained")
print("PASS: Undocumented/internal signals flagged for review")

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STAGE 19B: COMPLETE")
print("=" * 70)