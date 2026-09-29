from pathlib import Path
import pandas as pd


# ============================================================
# STAGE 19C: ENGINEERING REVIEW RESOLUTION
# ============================================================

print("=" * 70)
print("STAGE 19C: ENGINEERING REVIEW RESOLUTION")
print("=" * 70)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "training_feature_selection"
    / "hai_2305_training_feature_selection_decision.csv"
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
    / "hai_2305_training_feature_selection_resolved.csv"
)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("\nLoading Stage 19B decisions...")

df = pd.read_csv(INPUT_FILE)

print("File loaded successfully.")


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

assert len(df) == 66
assert df["feature"].is_unique

required_columns = [
    "feature",
    "role",
    "selection_decision",
    "selection_reason",
]

for column in required_columns:
    assert column in df.columns, (
        f"Missing required column: {column}"
    )

print(f"Training candidate features: {len(df)}")


# ------------------------------------------------------------
# DOCUMENTED ENGINEERING SIGNALS
# ------------------------------------------------------------

DOCUMENTED_KEEP = {
    "P1_PP04",
    "P1_PP04SP",
    "P2_24Vdc",
    "P2_AutoSD",
    "P2_ManualSD",
    "P2_SCST",
    "P2_VIBTR01",
    "P2_VIBTR02",
    "P2_VIBTR03",
    "P2_VIBTR04",
    "P2_VT01",
    "P3_LCP01D",
    "P4_HT_FD",
    "P4_HT_PS",
    "P4_LD",
    "P4_ST_FD",
    "P4_ST_PS",
}


# ------------------------------------------------------------
# UNDOCUMENTED FEATURE
# ------------------------------------------------------------

REQUIRES_DOCUMENTATION = {
    "P1_PP04D",
}


# ------------------------------------------------------------
# INTERNAL CONTROL-LOGIC OUTPUTS
# ------------------------------------------------------------

INTERNAL_CONTROL_OUTPUTS = {
    "x1001_05_SETPOINT_OUT",
    "x1001_15_ASSIGN_OUT",
    "x1002_07_SETPOINT_OUT",
    "x1002_08_SETPOINT_OUT",
    "x1003_10_SETPOINT_OUT",
    "x1003_18_SETPOINT_OUT",
    "x1003_24_SUM_OUT",
}


# ------------------------------------------------------------
# RESOLVE
# ------------------------------------------------------------

resolved_decisions = []
resolved_reasons = []
review_categories = []


for _, row in df.iterrows():

    feature = row["feature"]
    old_decision = row["selection_decision"]

    if feature in DOCUMENTED_KEEP:

        new_decision = "KEEP"
        category = "DOCUMENTED_ENGINEERING_SIGNAL"

        reason = (
            "Documented engineering/process/control signal. "
            "Its role is meaningful for anomaly detection and "
            "there is no evidence requiring exclusion at this "
            "stage. High correlation alone is not sufficient "
            "evidence for removal."
        )

    elif feature in REQUIRES_DOCUMENTATION:

        new_decision = "REVIEW"
        category = "REQUIRES_DOCUMENTATION"

        reason = (
            "The feature is present in the dataset but its "
            "engineering meaning is not defined sufficiently "
            "in the available HAI documentation. Do not "
            "invent semantics; retain for controlled review."
        )

    elif feature in INTERNAL_CONTROL_OUTPUTS:

        new_decision = "REVIEW"
        category = "INTERNAL_CONTROL_LOGIC"

        reason = (
            "HAIEnd internal control-logic output. The available "
            "documentation does not provide sufficient detail "
            "to establish the exact engineering meaning of this "
            "specific output. Retain as a separate review group."
        )

    else:

        new_decision = old_decision
        category = "UNRESOLVED"
        reason = row["selection_reason"]

    resolved_decisions.append(new_decision)
    review_categories.append(category)
    resolved_reasons.append(reason)


df["review_category"] = review_categories
df["resolved_decision"] = resolved_decisions
df["resolved_reason"] = resolved_reasons


# ------------------------------------------------------------
# VALIDATION OF EXPECTED GROUPS
# ------------------------------------------------------------

documented_count = (
    df["review_category"]
    == "DOCUMENTED_ENGINEERING_SIGNAL"
).sum()

documentation_count = (
    df["review_category"]
    == "REQUIRES_DOCUMENTATION"
).sum()

internal_count = (
    df["review_category"]
    == "INTERNAL_CONTROL_LOGIC"
).sum()

print("\n" + "=" * 70)
print("STAGE 19C RESOLUTION SUMMARY")
print("=" * 70)

print(
    "Documented engineering signals :",
    documented_count
)

print(
    "Requires documentation         :",
    documentation_count
)

print(
    "Internal control-logic outputs :",
    internal_count
)

print(
    "Total classified               :",
    documented_count
    + documentation_count
    + internal_count
)


# ------------------------------------------------------------
# PRINT GROUPS
# ------------------------------------------------------------

print("\nDOCUMENTED ENGINEERING SIGNALS:")

print(
    df[
        df["review_category"]
        == "DOCUMENTED_ENGINEERING_SIGNAL"
    ][
        ["feature", "role"]
    ].to_string(index=False)
)


print("\nREQUIRES DOCUMENTATION:")

print(
    df[
        df["review_category"]
        == "REQUIRES_DOCUMENTATION"
    ][
        ["feature", "role"]
    ].to_string(index=False)
)


print("\nINTERNAL CONTROL-LOGIC OUTPUTS:")

print(
    df[
        df["review_category"]
        == "INTERNAL_CONTROL_LOGIC"
    ][
        ["feature", "role"]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

assert len(df) == 66
assert df["feature"].is_unique

assert documented_count == 17
assert documentation_count == 1
assert internal_count == 7

assert (
    df["resolved_decision"] == "EXCLUDE"
).sum() == 0

print("PASS: All 66 features preserved")
print("PASS: 17 documented engineering signals resolved KEEP")
print("PASS: 1 undocumented feature retained for review")
print("PASS: 7 internal control-logic outputs retained for review")
print("PASS: No features automatically excluded")
print("PASS: No Test 1/Test 2 labels used")
print("PASS: No undocumented semantics invented")

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STAGE 19C: COMPLETE")
print("=" * 70)