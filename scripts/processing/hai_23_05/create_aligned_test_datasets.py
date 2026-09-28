import pandas as pd
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hai"
    / "hai-23.05"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
)


# =========================================================
# FILE PATHS
# =========================================================

TEST1_SCADA = RAW_DIR / "hai-test1.csv"
TEST1_LABEL = RAW_DIR / "label-test1.csv"

TEST2_SCADA = RAW_DIR / "hai-test2.csv"
TEST2_LABEL = RAW_DIR / "label-test2.csv"


OUTPUT_TEST1 = (
    PROCESSED_DIR
    / "hai-test1_aligned.csv"
)

OUTPUT_TEST2 = (
    PROCESSED_DIR
    / "hai-test2_aligned.csv"
)


# =========================================================
# SETUP
# =========================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("HAI 23.05 PROCESSED TEST DATASET CREATION")
print("=" * 80)


# =========================================================
# TEST 1
# =========================================================

print("\n" + "=" * 80)
print("PROCESSING TEST1")
print("=" * 80)

print("\nLoading SCADA data...")

test1_scada = pd.read_csv(TEST1_SCADA)

test1_scada["timestamp"] = pd.to_datetime(
    test1_scada["timestamp"]
)

print(
    "SCADA rows:",
    len(test1_scada)
)

print(
    "SCADA columns:",
    len(test1_scada.columns)
)


print("\nLoading labels...")

test1_labels = pd.read_csv(TEST1_LABEL)

test1_labels["timestamp"] = pd.to_datetime(
    test1_labels["timestamp"]
)

print(
    "Label rows:",
    len(test1_labels)
)


# ---------------------------------------------------------
# TEST1 ALIGNMENT VALIDATION
# ---------------------------------------------------------

print("\nValidating TEST1 timestamp alignment...")

test1_scada_timestamps = set(
    test1_scada["timestamp"]
)

test1_label_timestamps = set(
    test1_labels["timestamp"]
)

test1_missing_labels = (
    test1_scada_timestamps
    - test1_label_timestamps
)

test1_extra_labels = (
    test1_label_timestamps
    - test1_scada_timestamps
)

print(
    "SCADA timestamps without labels:",
    len(test1_missing_labels)
)

print(
    "Label timestamps without SCADA:",
    len(test1_extra_labels)
)

if len(test1_missing_labels) != 0:
    raise ValueError(
        "TEST1 contains SCADA timestamps without labels."
    )

if len(test1_extra_labels) != 0:
    raise ValueError(
        "TEST1 contains labels without matching SCADA timestamps."
    )


# ---------------------------------------------------------
# MERGE TEST1
# ---------------------------------------------------------

test1_aligned = test1_scada.merge(
    test1_labels,
    on="timestamp",
    how="left",
    validate="one_to_one"
)


# ---------------------------------------------------------
# TEST1 VALIDATION
# ---------------------------------------------------------

if test1_aligned["label"].isna().any():
    raise ValueError(
        "TEST1 contains missing labels after alignment."
    )

if not test1_aligned["label"].isin([0, 1]).all():
    raise ValueError(
        "TEST1 contains invalid label values."
    )

if test1_aligned["timestamp"].duplicated().any():
    raise ValueError(
        "TEST1 contains duplicate timestamps."
    )


print("\nTEST1 validation: PASS")

print(
    "Final rows:",
    len(test1_aligned)
)

print(
    "Final columns:",
    len(test1_aligned.columns)
)

print(
    "Label distribution:"
)

print(
    test1_aligned["label"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# SAVE TEST1
# ---------------------------------------------------------

test1_aligned.to_csv(
    OUTPUT_TEST1,
    index=False
)

print(
    "\nSaved:",
    OUTPUT_TEST1
)


# =========================================================
# TEST 2
# =========================================================

print("\n" + "=" * 80)
print("PROCESSING TEST2")
print("=" * 80)

print("\nLoading SCADA data...")

test2_scada = pd.read_csv(TEST2_SCADA)

test2_scada["timestamp"] = pd.to_datetime(
    test2_scada["timestamp"]
)

print(
    "SCADA rows:",
    len(test2_scada)
)

print(
    "SCADA columns:",
    len(test2_scada.columns)
)


print("\nLoading labels...")

test2_labels = pd.read_csv(TEST2_LABEL)

test2_labels["timestamp"] = pd.to_datetime(
    test2_labels["timestamp"]
)

print(
    "Label rows:",
    len(test2_labels)
)


# ---------------------------------------------------------
# RECONSTRUCT TEST2 LABEL TIMESTAMPS
# ---------------------------------------------------------

print(
    "\nReconstructing TEST2 second-level labels..."
)

expanded_rows = []

first_label_minute = (
    test2_labels["timestamp"].min()
)

for minute, group in test2_labels.groupby(
    "timestamp",
    sort=True
):

    group = group.reset_index(drop=True)

    for second_offset, label_value in enumerate(
        group["label"]
    ):

        reconstructed_time = (
            minute
            + pd.Timedelta(
                seconds=second_offset
            )
        )

        expanded_rows.append(
            {
                "timestamp": reconstructed_time,
                "label": label_value
            }
        )


test2_expanded_labels = pd.DataFrame(
    expanded_rows
)

print(
    "Expanded label rows:",
    len(test2_expanded_labels)
)

print(
    "Expanded unique timestamps:",
    test2_expanded_labels[
        "timestamp"
    ].nunique()
)


# ---------------------------------------------------------
# APPLY VERIFIED FIRST-MINUTE CORRECTION
# ---------------------------------------------------------

print(
    "\nApplying verified first-minute correction..."
)

first_group_mask = (
    test2_expanded_labels["timestamp"]
    < first_label_minute
    + pd.Timedelta(minutes=1)
)

test2_expanded_labels.loc[
    first_group_mask,
    "timestamp"
] = (
    test2_expanded_labels.loc[
        first_group_mask,
        "timestamp"
    ]
    + pd.Timedelta(seconds=1)
)

print(
    "First-minute rows adjusted:",
    first_group_mask.sum()
)


# ---------------------------------------------------------
# REMOVE LABELS OUTSIDE SCADA RANGE
# ---------------------------------------------------------

scada_start = (
    test2_scada["timestamp"].min()
)

scada_end = (
    test2_scada["timestamp"].max()
)

before_filter = len(
    test2_expanded_labels
)

test2_expanded_labels = (
    test2_expanded_labels[
        (
            test2_expanded_labels["timestamp"]
            >= scada_start
        )
        &
        (
            test2_expanded_labels["timestamp"]
            <= scada_end
        )
    ]
    .copy()
)

removed_boundary_rows = (
    before_filter
    - len(test2_expanded_labels)
)

print(
    "Boundary labels removed:",
    removed_boundary_rows
)


# ---------------------------------------------------------
# TEST2 ALIGNMENT
# ---------------------------------------------------------

print("\nAligning TEST2 labels with SCADA...")

test2_aligned = test2_scada.merge(
    test2_expanded_labels,
    on="timestamp",
    how="left",
    validate="one_to_one"
)


# ---------------------------------------------------------
# TEST2 VALIDATION
# ---------------------------------------------------------

test2_missing_labels = (
    test2_aligned["label"].isna().sum()
)

test2_duplicate_timestamps = (
    test2_aligned["timestamp"]
    .duplicated()
    .sum()
)

test2_invalid_labels = (
    ~test2_aligned["label"].isin([0, 1])
).sum()


print("\nTEST2 validation:")

print(
    "SCADA rows:",
    len(test2_scada)
)

print(
    "Aligned rows:",
    len(test2_aligned)
)

print(
    "Missing labels:",
    test2_missing_labels
)

print(
    "Duplicate timestamps:",
    test2_duplicate_timestamps
)

print(
    "Invalid labels:",
    test2_invalid_labels
)


if len(test2_aligned) != len(test2_scada):
    raise ValueError(
        "TEST2 row count changed during alignment."
    )

if test2_missing_labels != 0:
    raise ValueError(
        "TEST2 contains missing labels."
    )

if test2_duplicate_timestamps != 0:
    raise ValueError(
        "TEST2 contains duplicate timestamps."
    )

if test2_invalid_labels != 0:
    raise ValueError(
        "TEST2 contains invalid labels."
    )


print("\nTEST2 validation: PASS")

print(
    "Final rows:",
    len(test2_aligned)
)

print(
    "Final columns:",
    len(test2_aligned.columns)
)

print(
    "Label distribution:"
)

print(
    test2_aligned["label"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# SAVE TEST2
# ---------------------------------------------------------

test2_aligned.to_csv(
    OUTPUT_TEST2,
    index=False
)

print(
    "\nSaved:",
    OUTPUT_TEST2
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("HAI 23.05 PROCESSED DATASET CREATION COMPLETE")
print("=" * 80)

print("\nTEST1:")
print(
    "Rows:",
    len(test1_aligned)
)

print(
    "Columns:",
    len(test1_aligned.columns)
)

print(
    "Output:",
    OUTPUT_TEST1
)


print("\nTEST2:")
print(
    "Rows:",
    len(test2_aligned)
)

print(
    "Columns:",
    len(test2_aligned.columns)
)

print(
    "Output:",
    OUTPUT_TEST2
)

print("\nALL PROCESSING VALIDATIONS PASSED")
print("=" * 80)