import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_DIR = PROJECT_ROOT / "data/raw/hai/hai-23.05"

SCADA_FILE = INPUT_DIR / "hai-test2.csv"
LABEL_FILE = INPUT_DIR / "label-test2.csv"


print("=" * 80)
print("HAI 23.05 TEST2 LABEL ALIGNMENT")
print("=" * 80)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

print("\nLoading SCADA data...")

scada = pd.read_csv(SCADA_FILE)
scada["timestamp"] = pd.to_datetime(scada["timestamp"])

print("SCADA rows:", len(scada))
print("SCADA unique timestamps:", scada["timestamp"].nunique())


print("\nLoading label data...")

labels = pd.read_csv(LABEL_FILE)
labels["timestamp"] = pd.to_datetime(labels["timestamp"])

print("Label rows:", len(labels))
print("Unique label timestamps:", labels["timestamp"].nunique())


# ---------------------------------------------------------
# 2. RECONSTRUCT LABEL TIMESTAMPS
# ---------------------------------------------------------

print("\nReconstructing label timestamps...")

expanded_rows = []

first_label_minute = labels["timestamp"].min()
last_label_minute = labels["timestamp"].max()

for minute, group in labels.groupby("timestamp", sort=True):

    group = group.reset_index(drop=True)

    for second_offset, label_value in enumerate(group["label"]):

        reconstructed_time = (
            minute
            + pd.Timedelta(seconds=second_offset)
        )

        expanded_rows.append(
            {
                "timestamp": reconstructed_time,
                "label": label_value
            }
        )


expanded_labels = pd.DataFrame(expanded_rows)

print("Expanded label rows:", len(expanded_labels))
print(
    "Expanded unique timestamps:",
    expanded_labels["timestamp"].nunique()
)

print(
    "Duplicate timestamps:",
    expanded_labels["timestamp"].duplicated().sum()
)


# ---------------------------------------------------------
# 3. HANDLE FIRST PARTIAL MINUTE
# ---------------------------------------------------------

print("\nHandling first partial label minute...")

first_group_mask = (
    expanded_labels["timestamp"] < first_label_minute
    + pd.Timedelta(minutes=1)
)

expanded_labels.loc[
    first_group_mask,
    "timestamp"
] = (
    expanded_labels.loc[
        first_group_mask,
        "timestamp"
    ]
    + pd.Timedelta(seconds=1)
)


print(
    "First label minute:",
    first_label_minute
)

print(
    "First group rows:",
    first_group_mask.sum()
)


# ---------------------------------------------------------
# 4. REMOVE FINAL BOUNDARY LABEL
# ---------------------------------------------------------

print("\nHandling final boundary label...")

scada_start = scada["timestamp"].min()
scada_end = scada["timestamp"].max()

before_filter = len(expanded_labels)

expanded_labels = expanded_labels[
    (expanded_labels["timestamp"] >= scada_start)
    & (expanded_labels["timestamp"] <= scada_end)
].copy()

removed_rows = before_filter - len(expanded_labels)

print("SCADA start:", scada_start)
print("SCADA end:", scada_end)
print("Boundary labels removed:", removed_rows)


# ---------------------------------------------------------
# 5. ALIGN WITH SCADA
# ---------------------------------------------------------

print("\nAligning labels with SCADA timestamps...")

aligned = scada.merge(
    expanded_labels,
    on="timestamp",
    how="left",
    validate="one_to_one"
)


# ---------------------------------------------------------
# 6. VALIDATION
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ALIGNMENT VALIDATION")
print("=" * 80)

print("SCADA rows:", len(scada))
print("Aligned rows:", len(aligned))

print(
    "SCADA unique timestamps:",
    scada["timestamp"].nunique()
)

print(
    "Aligned unique timestamps:",
    aligned["timestamp"].nunique()
)

print(
    "Missing labels:",
    aligned["label"].isna().sum()
)

print(
    "Duplicate aligned timestamps:",
    aligned["timestamp"].duplicated().sum()
)

outside_count = (
    ~expanded_labels["timestamp"].isin(
        scada["timestamp"]
    )
).sum()

print(
    "Labels outside SCADA range:",
    outside_count
)


# ---------------------------------------------------------
# 7. LABEL DISTRIBUTION
# ---------------------------------------------------------

print("\nAligned label distribution:")

print(
    aligned["label"]
    .value_counts(dropna=False)
    .sort_index()
)


# ---------------------------------------------------------
# 8. CHECK FIRST TRANSITION
# ---------------------------------------------------------

print("\nFirst 10 aligned rows:")

print(
    aligned[
        ["timestamp", "label"]
    ]
    .head(10)
    .to_string(index=False)
)


print("\nRows around first-minute transition:")

transition_start = (
    first_label_minute
    + pd.Timedelta(seconds=55)
)

transition_end = (
    first_label_minute
    + pd.Timedelta(minutes=1, seconds=5)
)

transition = aligned[
    (aligned["timestamp"] >= transition_start)
    & (aligned["timestamp"] <= transition_end)
][["timestamp", "label"]]

print(
    transition.to_string(index=False)
)


print("\nLast 10 aligned rows:")

print(
    aligned[
        ["timestamp", "label"]
    ]
    .tail(10)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 9. FINAL CHECKS
# ---------------------------------------------------------

checks = {

    "row_count":
        len(aligned) == len(scada),

    "unique_timestamps":
        aligned["timestamp"].nunique()
        == len(aligned),

    "missing_labels":
        aligned["label"].isna().sum() == 0,

    "duplicate_timestamps":
        aligned["timestamp"].duplicated().sum() == 0,

    "outside_labels":
        outside_count == 0,

    "label_count":
        aligned["label"].isin([0, 1]).all(),

}


print("\n" + "=" * 80)
print("FINAL ALIGNMENT CHECKS")
print("=" * 80)

all_passed = True

for check_name, passed in checks.items():

    status = "PASS" if passed else "FAIL"

    print(
        f"{check_name:25s}: {status}"
    )

    if not passed:
        all_passed = False


print("\n" + "=" * 80)

if all_passed:
    print("HAI TEST2 LABEL ALIGNMENT: PASS")
else:
    print("HAI TEST2 LABEL ALIGNMENT: REVIEW REQUIRED")

print("=" * 80)