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

OUTPUT_FILE = FEATURE_DIR / "hai_2305_redundancy_analysis.csv"

CORR_TEST1 = FEATURE_DIR / "hai_test1_high_correlation_pairs.csv"
CORR_TEST2 = FEATURE_DIR / "hai_test2_high_correlation_pairs.csv"

ROLE_FILE = FEATURE_DIR / "hai_2305_feature_roles.csv"


# ============================================================
# LOAD DATA
# ============================================================

test1 = pd.read_csv(CORR_TEST1)
test2 = pd.read_csv(CORR_TEST2)
roles = pd.read_csv(ROLE_FILE)


# ============================================================
# NORMALIZE CORRELATION FILES
# ============================================================

def normalize_correlation_file(df):
    """
    Convert the correlation-pair file into:
        feature_1
        feature_2
        correlation
    regardless of the exact column order.
    """

    columns = {c.lower(): c for c in df.columns}

    feature_columns = [
        c for c in df.columns
        if "feature" in c.lower() or "variable" in c.lower()
    ]

    correlation_columns = [
        c for c in df.columns
        if "corr" in c.lower() or c.lower() in ["r", "pearson_r"]
    ]

    if len(feature_columns) < 2:
        raise ValueError(
            f"Could not identify two feature columns in: {df.columns.tolist()}"
        )

    if len(correlation_columns) < 1:
        raise ValueError(
            f"Could not identify correlation column in: {df.columns.tolist()}"
        )

    result = df[
        [feature_columns[0], feature_columns[1], correlation_columns[0]]
    ].copy()

    result.columns = [
        "feature_1",
        "feature_2",
        "correlation"
    ]

    return result


test1 = normalize_correlation_file(test1)
test2 = normalize_correlation_file(test2)


# ============================================================
# ADD DATASET SOURCE
# ============================================================

test1["dataset"] = "Test1"
test2["dataset"] = "Test2"


# ============================================================
# COMBINE CORRELATION PAIRS
# ============================================================

correlations = pd.concat(
    [test1, test2],
    ignore_index=True
)


# ============================================================
# CREATE ORDER-INDEPENDENT PAIR KEY
# ============================================================

def pair_key(row):
    return " | ".join(
        sorted(
            [
                str(row["feature_1"]),
                str(row["feature_2"])
            ]
        )
    )


correlations["pair_key"] = correlations.apply(
    pair_key,
    axis=1
)


# ============================================================
# PIVOT TEST 1 / TEST 2 CORRELATIONS
# ============================================================

correlations_pivot = (
    correlations
    .pivot_table(
        index="pair_key",
        columns="dataset",
        values="correlation",
        aggfunc="first"
    )
    .reset_index()
)


# ============================================================
# RECOVER FEATURE NAMES
# ============================================================

def split_pair(pair):
    parts = pair.split(" | ")

    if len(parts) != 2:
        raise ValueError(f"Invalid pair: {pair}")

    return parts[0], parts[1]


correlations_pivot[
    ["feature_1", "feature_2"]
] = correlations_pivot["pair_key"].apply(
    lambda x: pd.Series(split_pair(x))
)


# ============================================================
# ROLE LOOKUP
# ============================================================

role_lookup = dict(
    zip(
        roles["feature"],
        roles["role"]
    )
)


def get_role(feature):
    return role_lookup.get(
        feature,
        "Unknown"
    )


correlations_pivot["role_1"] = correlations_pivot[
    "feature_1"
].map(get_role)

correlations_pivot["role_2"] = correlations_pivot[
    "feature_2"
].map(get_role)


# ============================================================
# ENGINEERING RELATIONSHIP CLASSIFICATION
# ============================================================

def classify_relationship(role1, role2):
    roles_pair = {role1, role2}

    # Command vs actual/current position
    if (
        "Valve Position Command" in roles_pair
        and "Valve Current Position" in roles_pair
    ):
        return "Command vs Actual Position"

    # Measured vs converted flow
    if (
        "Measured Flowrate" in roles_pair
        and "Converted Flowrate" in roles_pair
    ):
        return "Measured vs Converted Flowrate"

    # Setpoint vs process measurement
    if (
        "Setpoint" in roles_pair
        and "Process Measurement" in roles_pair
    ):
        return "Setpoint vs Process Measurement"

    # Threshold vs process measurement
    if (
        "Process Threshold" in roles_pair
        and "Process Measurement" in roles_pair
    ):
        return "Threshold vs Process Measurement"

    # Controller output vs process measurement
    if (
        "Controller Output" in roles_pair
        and "Process Measurement" in roles_pair
    ):
        return "Controller Output vs Process Measurement"

    # Output power vs load demand
    if (
        "Output Power" in roles_pair
        and (
            "Electrical Load Demand" in roles_pair
            or "Total Electrical Load Demand" in roles_pair
        )
    ):
        return "Power Output vs Load Demand"

    # Scheduled demand vs actual output
    if (
        "Scheduled Power Demand" in roles_pair
        and (
            "Output Power" in roles_pair
            or "Electrical Load Demand" in roles_pair
        )
    ):
        return "Scheduled Demand vs Actual/Load"

    # Frequency deviation vs power/load
    if (
        "Frequency Deviation" in roles_pair
        and (
            "Output Power" in roles_pair
            or "Electrical Load Demand" in roles_pair
            or "Total Electrical Load Demand" in roles_pair
        )
    ):
        return "Frequency vs Power/Load"

    # Internal HAIEnd control outputs
    if (
        "Internal Control-Logic Output" in roles_pair
        and len(roles_pair) > 1
    ):
        return "Internal Control Logic vs Process Signal"

    # Same engineering role
    if role1 == role2:
        return "Same Role"

    return "Different Engineering Roles"


correlations_pivot["engineering_relationship"] = (
    correlations_pivot.apply(
        lambda row: classify_relationship(
            row["role_1"],
            row["role_2"]
        ),
        axis=1
    )
)


# ============================================================
# DECISION
# ============================================================

def decide(row):
    relationship = row["engineering_relationship"]

    # These relationships represent distinct engineering signals.
    keep_relationships = {
        "Command vs Actual Position",
        "Measured vs Converted Flowrate",
        "Setpoint vs Process Measurement",
        "Threshold vs Process Measurement",
        "Controller Output vs Process Measurement",
        "Power Output vs Load Demand",
        "Scheduled Demand vs Actual/Load",
        "Frequency vs Power/Load",
        "Internal Control Logic vs Process Signal",
    }

    if relationship in keep_relationships:
        return "KEEP"

    if relationship == "Same Role":
        return "REVIEW"

    return "REVIEW"


correlations_pivot["decision"] = correlations_pivot.apply(
    decide,
    axis=1
)


# ============================================================
# REASON
# ============================================================

def reason(row):
    decision = row["decision"]
    relationship = row["engineering_relationship"]

    if decision == "KEEP":
        return (
            "High correlation exists, but the documented engineering "
            "roles represent distinct signals in the control/process system."
        )

    if relationship == "Same Role":
        return (
            "Both features have the same documented engineering role. "
            "High correlation may indicate redundancy, but engineering "
            "meaning and data representation should be reviewed before removal."
        )

    return (
        "High correlation requires engineering review before any "
        "feature-reduction decision."
    )


correlations_pivot["reason"] = correlations_pivot.apply(
    reason,
    axis=1
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

output = correlations_pivot[
    [
        "feature_1",
        "feature_2",
        "Test1",
        "Test2",
        "role_1",
        "role_2",
        "engineering_relationship",
        "decision",
        "reason",
    ]
].copy()


# ============================================================
# SORT
# ============================================================

output = output.sort_values(
    by=[
        "decision",
        "feature_1",
        "feature_2"
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("HAI 23.05 ENGINEERING-BASED REDUNDANCY ANALYSIS")
print("=" * 70)

print()
print(f"Test 1 high-correlation pairs : {len(test1)}")
print(f"Test 2 high-correlation pairs : {len(test2)}")
print(f"Unique feature pairs          : {len(output)}")

print()
print("DECISION SUMMARY")
print("-" * 70)

print(
    output["decision"]
    .value_counts()
    .to_string()
)

print()
print("ENGINEERING RELATIONSHIPS")
print("-" * 70)

print(
    output["engineering_relationship"]
    .value_counts()
    .to_string()
)

print()
print("OUTPUT")
print("-" * 70)

print(OUTPUT_FILE)

print()
print("=" * 70)
print("REDUNDANCY ANALYSIS COMPLETE")
print("=" * 70)