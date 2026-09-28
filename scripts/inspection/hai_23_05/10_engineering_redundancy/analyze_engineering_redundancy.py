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

STABILITY_FILE = (
    FEATURE_DIR
    / "hai_2305_correlation_stability.csv"
)

ROLE_FILE = (
    FEATURE_DIR
    / "hai_2305_feature_roles.csv"
)

SELECTION_FILE = (
    FEATURE_DIR
    / "hai_2305_feature_selection_decision.csv"
)

OUTPUT_FILE = (
    FEATURE_DIR
    / "hai_2305_engineering_redundancy_review.csv"
)


# ============================================================
# LOAD
# ============================================================

stability = pd.read_csv(STABILITY_FILE)
roles = pd.read_csv(ROLE_FILE)
selection = pd.read_csv(SELECTION_FILE)

# Only stable high correlations
stable = stability[
    stability["stability_class"]
    == "STABLE_HIGH_CORRELATION"
].copy()


# ============================================================
# ADD FEATURE INFORMATION
# ============================================================

role_map = roles.set_index("feature")["role"].to_dict()
variability_map = (
    roles.set_index("feature")["variability"].to_dict()
)

decision_map = (
    selection.set_index("feature")["decision"].to_dict()
)


# ============================================================
# ENGINEERING DECISION RULE
# ============================================================

def assess_pair(row):

    f1 = row["feature_1"]
    f2 = row["feature_2"]

    role1 = role_map.get(f1, "Unknown")
    role2 = role_map.get(f2, "Unknown")

    var1 = variability_map.get(f1, "Unknown")
    var2 = variability_map.get(f2, "Unknown")

    # --------------------------------------------------------
    # SAME ROLE
    # --------------------------------------------------------

    if role1 == role2:

        if role1 == "Process Measurement":

            return (
                "REDUNDANT_CANDIDATE",
                f"Both features are process measurements with "
                f"stable high correlation. Review which signal "
                f"provides the more direct or useful process "
                f"representation before retaining both."
            )

        if role1 == "Equipment / Pump Signal":

            return (
                "REDUNDANT_CANDIDATE",
                "Both are equipment/pump signals with stable "
                "high correlation. Review engineering meaning "
                "before retaining both."
            )

        if role1 == "Equipment Condition":

            return (
                "REDUNDANT_CANDIDATE",
                "Both represent equipment condition and are "
                "strongly correlated. Review whether both "
                "provide independent diagnostic information."
            )

        if role1 == "Status / Switch":

            return (
                "REVIEW",
                "Both are status/switch variables. High "
                "correlation may indicate duplicate operating "
                "state information."
            )

    # --------------------------------------------------------
    # PROCESS MEASUREMENT + SETPOINT
    # --------------------------------------------------------

    if (
        "Process Measurement" in [role1, role2]
        and
        "Setpoint" in [role1, role2]
    ):

        return (
            "RETAIN_BOTH_REVIEW",
            "A process measurement and a setpoint are "
            "engineering-distinct even when strongly correlated. "
            "Do not remove solely because of correlation."
        )

    # --------------------------------------------------------
    # PROCESS MEASUREMENT + CONTROL
    # --------------------------------------------------------

    if (
        "Process Measurement" in [role1, role2]
        and
        "Control / Assignment" in [role1, role2]
    ):

        return (
            "RETAIN_BOTH_REVIEW",
            "Process measurement and control/assignment signals "
            "have different operational roles. Correlation alone "
            "is insufficient for removal."
        )

    # --------------------------------------------------------
    # OTHER DIFFERENT ENGINEERING ROLES
    # --------------------------------------------------------

    if role1 != role2:

        return (
            "RETAIN_BOTH_REVIEW",
            "Features have different engineering roles despite "
            "stable high correlation. Further engineering "
            "validation is required before removal."
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return (
        "REVIEW",
        "Requires engineering review."
    )


# ============================================================
# APPLY
# ============================================================

assessments = stable.apply(
    assess_pair,
    axis=1
)

stable["engineering_decision"] = [
    item[0]
    for item in assessments
]

stable["engineering_reason"] = [
    item[1]
    for item in assessments
]


# ============================================================
# ADD ROLE INFORMATION
# ============================================================

stable["feature_1_role"] = (
    stable["feature_1"]
    .map(role_map)
)

stable["feature_2_role"] = (
    stable["feature_2"]
    .map(role_map)
)

stable["feature_1_variability"] = (
    stable["feature_1"]
    .map(variability_map)
)

stable["feature_2_variability"] = (
    stable["feature_2"]
    .map(variability_map)
)


# ============================================================
# COLUMN ORDER
# ============================================================

columns = [
    "feature_1",
    "feature_1_role",
    "feature_1_variability",
    "feature_2",
    "feature_2_role",
    "feature_2_variability",
    "test1_correlation",
    "test2_correlation",
    "mean_absolute_correlation",
    "engineering_decision",
    "engineering_reason"
]

stable = stable[columns]


# ============================================================
# SAVE
# ============================================================

stable.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("HAI 23.05 ENGINEERING-AWARE REDUNDANCY REVIEW")
print("=" * 70)

print(
    f"\nStable high-correlation pairs analysed: "
    f"{len(stable)}"
)

print("\n" + "=" * 70)
print("ENGINEERING DECISION SUMMARY")
print("=" * 70)

print(
    stable["engineering_decision"]
    .value_counts()
    .to_string()
)


# ============================================================
# FULL REVIEW
# ============================================================

print("\n" + "=" * 70)
print("ENGINEERING REDUNDANCY REVIEW")
print("=" * 70)

for _, row in stable.iterrows():

    print("\n" + "-" * 70)

    print(
        f"{row['feature_1']} "
        f"({row['feature_1_role']})"
    )

    print(
        f"    ↔ "
        f"{row['feature_2']} "
        f"({row['feature_2_role']})"
    )

    print(
        f"    Test 1 correlation: "
        f"{row['test1_correlation']:.6f}"
    )

    print(
        f"    Test 2 correlation: "
        f"{row['test2_correlation']:.6f}"
    )

    print(
        f"    Decision: "
        f"{row['engineering_decision']}"
    )

    print(
        f"    Reason: "
        f"{row['engineering_reason']}"
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("ENGINEERING-AWARE REDUNDANCY REVIEW COMPLETE")
print("=" * 70)