from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]

BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
)

MODEL_READY_DIR = (
    BASE_DIR
    / "model_ready"
)

TEMPORAL_DIR = (
    BASE_DIR
    / "temporal_representation"
)

INPUT_BASE = (
    MODEL_READY_DIR
    / "hai_2305_training_model_ready.csv"
)

INPUT_TEMPORAL = (
    TEMPORAL_DIR
    / "hai_2305_training_temporal_features.csv"
)

INPUT_SELECTION = (
    TEMPORAL_DIR
    / "hai_2305_selected_temporal_features.txt"
)

OUTPUT_FILE = (
    TEMPORAL_DIR
    / "hai_2305_training_temporal_model_ready.csv"
)

OUTPUT_SCHEMA = (
    TEMPORAL_DIR
    / "hai_2305_temporal_model_ready_schema.csv"
)

OUTPUT_SUMMARY = (
    TEMPORAL_DIR
    / "hai_2305_temporal_model_ready_summary.txt"
)


print("=" * 70)
print("STAGE 22H: FINAL TEMPORAL MODEL-READY DATASET")
print("=" * 70)


# ============================================================
# 1. LOAD BASE MODEL-READY DATA
# ============================================================

print("\nLoading original model-ready training data...")

base_df = pd.read_csv(
    INPUT_BASE
)

print(
    f"Base rows    : {len(base_df):,}"
)

print(
    f"Base columns : {len(base_df.columns)}"
)

assert len(base_df) == 896400
assert "timestamp" in base_df.columns


# ============================================================
# 2. LOAD TEMPORAL DATA
# ============================================================

print("\nLoading temporal feature data...")

temporal_df = pd.read_csv(
    INPUT_TEMPORAL
)

print(
    f"Temporal rows    : {len(temporal_df):,}"
)

print(
    f"Temporal columns : {len(temporal_df.columns)}"
)

assert len(temporal_df) == 896400
assert "timestamp" in temporal_df.columns

assert (
    len(temporal_df.columns) == 175
)


# ============================================================
# 3. LOAD SELECTED FEATURE LIST
# ============================================================

print("\nLoading selected temporal feature list...")

with open(
    INPUT_SELECTION,
    "r",
    encoding="utf-8"
) as f:

    selected_temporal_features = [
        line.strip()
        for line in f
        if line.strip()
    ]


assert len(
    selected_temporal_features
) == 174

assert len(
    set(selected_temporal_features)
) == 174


print(
    f"Selected temporal features: "
    f"{len(selected_temporal_features)}"
)


# ============================================================
# 4. VALIDATE TEMPORAL SCHEMA
# ============================================================

for feature in selected_temporal_features:

    assert feature in temporal_df.columns


print(
    "PASS: All selected temporal features exist"
)


# ============================================================
# 5. TIMESTAMP VALIDATION
# ============================================================

print("\nValidating timestamps...")

base_timestamps = pd.to_datetime(
    base_df["timestamp"]
)

temporal_timestamps = pd.to_datetime(
    temporal_df["timestamp"]
)

assert (
    base_timestamps.equals(
        temporal_timestamps
    )
)

print(
    "PASS: Base and temporal timestamps match exactly"
)


# ============================================================
# 6. ORIGINAL FEATURE EXTRACTION
# ============================================================

original_features = [
    col
    for col in base_df.columns
    if col != "timestamp"
]

assert len(
    original_features
) == 58

print(
    f"Original model-ready features: "
    f"{len(original_features)}"
)


# ============================================================
# 7. EXTRACT SELECTED TEMPORAL FEATURES
# ============================================================

temporal_selected_df = (
    temporal_df[
        selected_temporal_features
    ]
)


assert (
    temporal_selected_df.shape
    == (896400, 174)
)


# ============================================================
# 8. CHECK COLUMN COLLISIONS
# ============================================================

column_overlap = (
    set(original_features)
    &
    set(selected_temporal_features)
)

assert len(column_overlap) == 0

print(
    "PASS: No original/temporal column-name collisions"
)


# ============================================================
# 9. CONSTRUCT FINAL DATASET
# ============================================================

print("\nConstructing final temporal dataset...")

final_df = pd.concat(
    [
        base_df[
            [
                "timestamp"
            ]
            + original_features
        ],
        temporal_selected_df
    ],
    axis=1
)


# ============================================================
# 10. SHAPE VALIDATION
# ============================================================

expected_columns = (
    1
    + 58
    + 174
)

assert final_df.shape == (
    896400,
    expected_columns
)

assert expected_columns == 233

print(
    f"Final rows    : {len(final_df):,}"
)

print(
    f"Final columns : {len(final_df.columns)}"
)

print(
    "PASS: Expected 896,400 × 233 shape"
)


# ============================================================
# 11. TIMESTAMP POSITION
# ============================================================

assert (
    final_df.columns[0]
    == "timestamp"
)

print(
    "PASS: Timestamp is first column"
)


# ============================================================
# 12. ORIGINAL FEATURE PRESERVATION
# ============================================================

final_original_features = [
    col
    for col in final_df.columns
    if col in original_features
]

assert set(
    final_original_features
) == set(
    original_features
)

assert len(
    final_original_features
) == 58

print(
    "PASS: All 58 original features preserved"
)


# ============================================================
# 13. TEMPORAL FEATURE PRESERVATION
# ============================================================

final_temporal_features = [
    col
    for col in final_df.columns
    if col in selected_temporal_features
]

assert set(
    final_temporal_features
) == set(
    selected_temporal_features
)

assert len(
    final_temporal_features
) == 174

print(
    "PASS: All 174 temporal features preserved"
)


# ============================================================
# 14. DUPLICATE COLUMN CHECK
# ============================================================

assert (
    final_df.columns.nunique()
    == len(final_df.columns)
)

print(
    "PASS: All 233 column names are unique"
)


# ============================================================
# 15. TIMESTAMP QUALITY
# ============================================================

assert (
    final_df["timestamp"]
    .notna()
    .all()
)

print(
    "PASS: No missing timestamps"
)


# ============================================================
# 16. NUMERICAL FEATURE CHECK
# ============================================================

numeric_columns = [
    col
    for col in final_df.columns
    if col != "timestamp"
]

numeric_count = (
    final_df[
        numeric_columns
    ]
    .select_dtypes(
        include=np.number
    )
    .shape[1]
)

assert (
    numeric_count
    == 232
)

print(
    "PASS: All 232 non-timestamp columns are numeric"
)


# ============================================================
# 17. INFINITE VALUE CHECK
# ============================================================

print("\nChecking infinite values...")

infinite_count = np.isinf(
    final_df[
        numeric_columns
    ].to_numpy()
).sum()

assert infinite_count == 0

print(
    "PASS: No infinite values"
)


# ============================================================
# 18. TEMPORAL NaN CHECK
# ============================================================

print("\nChecking temporal boundary NaNs...")

temporal_nan_count = (
    final_df[
        selected_temporal_features
    ]
    .isna()
    .sum()
    .sum()
)

print(
    f"Temporal NaN count: "
    f"{temporal_nan_count:,}"
)

# Stage 22B established:
# diff      = 4 NaNs × 58
# abs diff  = 4 NaNs × 58
# rolling   = 16 NaNs × 58
#
# Expected total:
# 232 + 232 + 928 = 1392

expected_temporal_nan_count = 1392

assert (
    temporal_nan_count
    == expected_temporal_nan_count
)

print(
    "PASS: Expected boundary NaNs confirmed"
)


# ============================================================
# 19. ORIGINAL FEATURE NaN CHECK
# ============================================================

original_nan_count = (
    final_df[
        original_features
    ]
    .isna()
    .sum()
    .sum()
)

assert original_nan_count == 0

print(
    "PASS: Original model-ready features contain no NaNs"
)


# ============================================================
# 20. LABEL CHECK
# ============================================================

label_like_columns = [
    col
    for col in final_df.columns
    if "label" in col.lower()
    or "attack" in col.lower()
    or "target" in col.lower()
]

assert len(
    label_like_columns
) == 0

print(
    "PASS: No label/target columns present"
)


# ============================================================
# 21. TRANSFORMATION COUNTS
# ============================================================

diff_features = [
    col
    for col in selected_temporal_features
    if col.endswith(
        "__diff_1s"
    )
]

abs_diff_features = [
    col
    for col in selected_temporal_features
    if col.endswith(
        "__abs_diff_1s"
    )
]

rolling_features = [
    col
    for col in selected_temporal_features
    if col.endswith(
        "__rolling_std_5s"
    )
]

assert len(diff_features) == 58
assert len(abs_diff_features) == 58
assert len(rolling_features) == 58

print(
    "\nTemporal transformation counts:"
)

print(
    f"diff_1s          : {len(diff_features)}"
)

print(
    f"abs_diff_1s      : {len(abs_diff_features)}"
)

print(
    f"rolling_std_5s   : {len(rolling_features)}"
)


# ============================================================
# 22. SAVE FINAL DATASET
# ============================================================

print(
    "\nSaving final temporal model-ready dataset..."
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    OUTPUT_FILE
)


# ============================================================
# 23. CREATE SCHEMA REPORT
# ============================================================

schema_records = []

for index, column in enumerate(
    final_df.columns,
    start=1
):

    if column == "timestamp":

        category = "timestamp"
        transformation = "none"

    elif column in original_features:

        category = "original_model_ready"
        transformation = "original"

    elif column in diff_features:

        category = "temporal"
        transformation = "diff_1s"

    elif column in abs_diff_features:

        category = "temporal"
        transformation = "abs_diff_1s"

    elif column in rolling_features:

        category = "temporal"
        transformation = "rolling_std_5s"

    else:

        raise ValueError(
            f"Unexpected column: {column}"
        )

    schema_records.append({
        "column_index": index,
        "feature": column,
        "category": category,
        "transformation": transformation
    })


schema_df = pd.DataFrame(
    schema_records
)

schema_df.to_csv(
    OUTPUT_SCHEMA,
    index=False
)


# ============================================================
# 24. SUMMARY FILE
# ============================================================

summary_lines = [
    "HAI 23.05 TEMPORAL MODEL-READY DATASET",
    "",
    f"Rows: {len(final_df):,}",
    f"Columns: {len(final_df.columns)}",
    "",
    "Feature composition:",
    "Timestamp: 1",
    "Original model-ready features: 58",
    "Temporal features: 174",
    "Total numerical features: 232",
    "",
    "Temporal transformations:",
    "diff_1s: 58",
    "abs_diff_1s: 58",
    "rolling_std_5s: 58",
    "",
    f"Temporal NaN count: {temporal_nan_count:,}",
    f"Expected temporal NaN count: {expected_temporal_nan_count:,}",
    f"Infinite values: {infinite_count}",
    f"Original feature NaNs: {original_nan_count}",
    "",
    "Labels/targets included: No",
    "Test data included: No",
    "",
    "Selection status:",
    "174 / 174 temporal features retained",
    "0 temporal features excluded"
]

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary_lines)
    )


# ============================================================
# 25. FINAL VALIDATION
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

assert final_df.shape == (
    896400,
    233
)

assert (
    final_df.columns[0]
    == "timestamp"
)

assert (
    len(original_features)
    == 58
)

assert (
    len(selected_temporal_features)
    == 174
)

assert (
    temporal_nan_count
    == 1392
)

assert (
    infinite_count
    == 0
)

assert (
    original_nan_count
    == 0
)

assert (
    len(label_like_columns)
    == 0
)

assert (
    final_df.columns.nunique()
    == 233
)

print(
    "PASS: Final dataset is 896,400 × 233"
)

print(
    "PASS: 58 original model-ready features present"
)

print(
    "PASS: 174 selected temporal features present"
)

print(
    "PASS: 232 numerical features present"
)

print(
    "PASS: Expected 1,392 temporal boundary NaNs"
)

print(
    "PASS: No infinite values"
)

print(
    "PASS: No original-feature NaNs"
)

print(
    "PASS: No labels or targets"
)

print(
    "PASS: No duplicate column names"
)

print(
    "PASS: Training data only"
)

print(
    "\nOutputs:"
)

print(
    OUTPUT_FILE
)

print(
    OUTPUT_SCHEMA
)

print(
    OUTPUT_SUMMARY
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 22H: COMPLETE"
)

print(
    "=" * 70
)