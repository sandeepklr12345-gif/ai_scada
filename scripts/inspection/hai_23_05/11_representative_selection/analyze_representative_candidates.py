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

GROUP_FILE = (
    FEATURE_DIR
    / "hai_2305_keep_correlation_groups.csv"
)

ROLE_FILE = (
    FEATURE_DIR
    / "hai_2305_feature_roles.csv"
)

STABILITY_FILE = (
    FEATURE_DIR
    / "hai_2305_correlation_stability.csv"
)

OUTPUT_FILE = (
    FEATURE_DIR
    / "hai_2305_representative_candidates.csv"
)


# ============================================================
# LOAD
# ============================================================

groups = pd.read_csv(GROUP_FILE)
roles = pd.read_csv(ROLE_FILE)
stability = pd.read_csv(STABILITY_FILE)


# ============================================================
# ROLE / VARIABILITY MAPS
# ============================================================

role_map = (
    roles
    .set_index("feature")["role"]
    .to_dict()
)

variability_map = (
    roles
    .set_index("feature")["variability"]
    .to_dict()
)


# ============================================================
# BUILD GROUPS
# ============================================================

grouped = (
    groups
    .groupby("correlation_group")["feature"]
    .apply(list)
    .to_dict()
)


# ============================================================
# COUNT STABLE RELATIONSHIPS PER FEATURE
# ============================================================

stable = stability[
    stability["stability_class"]
    == "STABLE_HIGH_CORRELATION"
].copy()

stable_count = {}

for _, row in stable.iterrows():

    f1 = row["feature_1"]
    f2 = row["feature_2"]

    stable_count[f1] = (
        stable_count.get(f1, 0) + 1
    )

    stable_count[f2] = (
        stable_count.get(f2, 0) + 1
    )


# ============================================================
# BUILD REPRESENTATIVE CANDIDATE TABLE
# ============================================================

rows = []

for group_name, features in grouped.items():

    group_size = len(features)

    for feature in features:

        role = role_map.get(
            feature,
            "Unknown"
        )

        variability = variability_map.get(
            feature,
            "Unknown"
        )

        stable_connections = stable_count.get(
            feature,
            0
        )

        # ----------------------------------------------------
        # INITIAL STATUS
        # ----------------------------------------------------

        if role == "Setpoint":

            candidate_status = (
                "RETAIN_FOR_CONTEXT"
            )

            reason = (
                "Setpoint provides operational context "
                "and should not be removed solely because "
                "of high correlation."
            )

        elif role == "Control / Assignment":

            candidate_status = (
                "RETAIN_FOR_CONTEXT"
            )

            reason = (
                "Control/assignment information has a "
                "distinct operational role."
            )

        elif role == "Process Measurement":

            candidate_status = (
                "REPRESENTATIVE_CANDIDATE"
            )

            reason = (
                "Process measurement is eligible to "
                "represent the correlated process group."
            )

        elif role == "Equipment Condition":

            candidate_status = (
                "REPRESENTATIVE_CANDIDATE"
            )

            reason = (
                "Equipment-condition signal may provide "
                "diagnostic information."
            )

        elif role == "Vibration / Equipment Condition":

            candidate_status = (
                "RETAIN_FOR_DIAGNOSTICS"
            )

            reason = (
                "Vibration/condition information is "
                "diagnostically important and should not "
                "be removed based on correlation alone."
            )

        elif role == "Status / Switch":

            candidate_status = (
                "RETAIN_FOR_CONTEXT"
            )

            reason = (
                "Operational-state information may be "
                "useful for anomaly interpretation."
            )

        else:

            candidate_status = "REVIEW"

            reason = (
                "Engineering role requires further review."
            )

        rows.append({
            "correlation_group": group_name,
            "group_size": group_size,
            "feature": feature,
            "role": role,
            "variability": variability,
            "stable_high_correlation_connections":
                stable_connections,
            "candidate_status":
                candidate_status,
            "reason":
                reason
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

result = pd.DataFrame(rows)


# ============================================================
# SORT
# ============================================================

result = result.sort_values(
    [
        "correlation_group",
        "candidate_status",
        "feature"
    ]
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("HAI 23.05 REPRESENTATIVE SELECTION ANALYSIS")
print("=" * 70)

print(
    f"\nCorrelation groups analysed: "
    f"{len(grouped)}"
)

print(
    f"Features across groups: "
    f"{len(result)}"
)


print("\n" + "=" * 70)
print("CANDIDATE STATUS SUMMARY")
print("=" * 70)

print(
    result["candidate_status"]
    .value_counts()
    .to_string()
)


# ============================================================
# GROUP-BY-GROUP DISPLAY
# ============================================================

for group_name in sorted(grouped.keys()):

    group = result[
        result["correlation_group"]
        == group_name
    ]

    print("\n" + "=" * 70)
    print(
        f"{group_name} "
        f"({len(group)} features)"
    )
    print("=" * 70)

    print(
        group[
            [
                "feature",
                "role",
                "variability",
                "stable_high_correlation_connections",
                "candidate_status"
            ]
        ].to_string(index=False)
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("REPRESENTATIVE SELECTION ANALYSIS COMPLETE")
print("=" * 70)