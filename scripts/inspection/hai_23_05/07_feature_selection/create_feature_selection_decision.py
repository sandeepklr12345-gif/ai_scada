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
STABILITY_FILE = FEATURE_DIR / "hai_2305_feature_stability.csv"

OUTPUT_FILE = FEATURE_DIR / "hai_2305_feature_selection_decision.csv"


# ============================================================
# LOAD DATA
# ============================================================

roles = pd.read_csv(ROLE_FILE)
stability = pd.read_csv(STABILITY_FILE)

# Normalize column names where needed
roles.columns = roles.columns.str.strip()
stability.columns = stability.columns.str.strip()


# ============================================================
# MERGE ROLE + STABILITY INFORMATION
# ============================================================

df = roles.merge(
    stability,
    on="feature",
    how="left",
    suffixes=("", "_stability")
)


# ============================================================
# DECISION LOGIC
# ============================================================

def decide_feature(row):

    feature = row["feature"]
    role = row["role"]

    constant_test1 = bool(row["constant_test1"])
    constant_test2 = bool(row["constant_test2"])
    constant_both = bool(row["constant_both"])

    variability = row["variability"]

    # --------------------------------------------------------
    # 1. Constant in BOTH test datasets
    # --------------------------------------------------------

    if constant_both:
        return (
            "EXCLUDE",
            "Constant in both Test 1 and Test 2; "
            "provides no variation for anomaly detection."
        )

    # --------------------------------------------------------
    # 2. Constant in only one dataset
    # --------------------------------------------------------

    if constant_test1 and not constant_test2:
        return (
            "REVIEW",
            "Constant in Test 1 but variable in Test 2; "
            "retain for cross-test comparison."
        )

    if constant_test2 and not constant_test1:
        return (
            "REVIEW",
            "Constant in Test 2 but variable in Test 1; "
            "retain for cross-test comparison."
        )

    # --------------------------------------------------------
    # 3. Unknown engineering meaning
    # --------------------------------------------------------

    if role == "Requires Documentation":
        return (
            "DOCUMENT",
            "Engineering meaning is not sufficiently established "
            "from the available feature information."
        )

    # --------------------------------------------------------
    # 4. Equipment condition
    # --------------------------------------------------------

    if role == "Vibration / Equipment Condition":
        return (
            "KEEP",
            "Directly relevant to equipment-condition and "
            "abnormal-condition analysis."
        )

    if role == "Equipment Condition":
        return (
            "KEEP",
            "Potentially relevant to equipment-condition analysis."
        )

    # --------------------------------------------------------
    # 5. Process measurements
    # --------------------------------------------------------

    if role == "Process Measurement":
        return (
            "KEEP",
            "Variable process measurement suitable as an "
            "anomaly-analysis candidate."
        )

    # --------------------------------------------------------
    # 6. Status / switch
    # --------------------------------------------------------

    if role == "Status / Switch":
        return (
            "KEEP",
            "Operational state information may help identify "
            "abnormal operating conditions."
        )

    # --------------------------------------------------------
    # 7. Setpoint
    # --------------------------------------------------------

    if role == "Setpoint":
        return (
            "KEEP",
            "Setpoint information can help distinguish expected "
            "operation from abnormal deviation."
        )

    # --------------------------------------------------------
    # 8. Control / assignment
    # --------------------------------------------------------

    if role == "Control / Assignment":
        return (
            "KEEP",
            "Control/assignment information may provide context "
            "for operating-state changes."
        )

    # --------------------------------------------------------
    # 9. Derived / calculated
    # --------------------------------------------------------

    if role == "Derived / Calculated":
        return (
            "REVIEW",
            "Derived variable may contain useful information but "
            "requires leakage/redundancy review."
        )

    # --------------------------------------------------------
    # 10. Equipment / pump signal
    # --------------------------------------------------------

    if role == "Equipment / Pump Signal":

        if variability == "Variable":
            return (
                "KEEP",
                "Variable equipment/pump signal is a candidate "
                "for abnormal-condition analysis."
            )

        if variability == "Two-State":
            return (
                "KEEP",
                "Two-state equipment signal can represent "
                "operating-state changes."
            )

        return (
            "EXCLUDE",
            "Equipment/pump signal is constant and therefore "
            "does not provide variation."
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return (
        "REVIEW",
        "Requires further engineering review."
    )


# ============================================================
# APPLY DECISIONS
# ============================================================

decisions = df.apply(decide_feature, axis=1)

df["decision"] = [item[0] for item in decisions]
df["decision_reason"] = [item[1] for item in decisions]


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

output_columns = [
    "feature",
    "role",
    "variability",
    "unique_values_test1",
    "constant_test1",
    "constant_test2",
    "constant_both",
    "decision",
    "decision_reason"
]

df = df[output_columns]


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("HAI 23.05 FEATURE SELECTION DECISION ANALYSIS")
print("=" * 70)

print("\nDECISION SUMMARY")
print("-" * 70)

print(
    df["decision"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 70)
print("FEATURES TO EXCLUDE")
print("=" * 70)

excluded = df[df["decision"] == "EXCLUDE"]

if len(excluded) > 0:
    print(
        excluded[
            ["feature", "role", "decision_reason"]
        ].to_string(index=False)
    )
else:
    print("None")


print("\n" + "=" * 70)
print("FEATURES REQUIRING REVIEW")
print("=" * 70)

review = df[df["decision"] == "REVIEW"]

if len(review) > 0:
    print(
        review[
            ["feature", "role", "variability", "decision_reason"]
        ].to_string(index=False)
    )
else:
    print("None")


print("\n" + "=" * 70)
print("FEATURES REQUIRING DOCUMENTATION")
print("=" * 70)

document = df[df["decision"] == "DOCUMENT"]

if len(document) > 0:
    print(
        document[
            ["feature", "role", "decision_reason"]
        ].to_string(index=False)
    )
else:
    print("None")


print("\n" + "=" * 70)
print("FEATURES TO KEEP")
print("=" * 70)

keep = df[df["decision"] == "KEEP"]

print(
    keep[
        ["feature", "role", "variability"]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("HAI 23.05 FEATURE SELECTION DECISION COMPLETE")
print("=" * 70)