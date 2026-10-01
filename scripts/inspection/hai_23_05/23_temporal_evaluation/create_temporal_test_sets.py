from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# STAGE 23A
# HAI 23.05 - Temporal Test Representation
# ============================================================

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

SELECTED_FEATURES_FILE = (
    TEMPORAL_DIR
    / "hai_2305_selected_temporal_features.txt"
)


TEST_FILES = {
    "test1": (
        MODEL_READY_DIR
        / "hai-test1_model_ready.csv"
    ),
    "test2": (
        MODEL_READY_DIR
        / "hai-test2_model_ready.csv"
    )
}


OUTPUT_FILES = {
    "test1": (
        TEMPORAL_DIR
        / "hai-test1_temporal_model_ready.csv"
    ),
    "test2": (
        TEMPORAL_DIR
        / "hai-test2_temporal_model_ready.csv"
    )
}


SCHEMA_FILE = (
    TEMPORAL_DIR
    / "hai_2305_temporal_test_schema.csv"
)

# ============================================================
# CONFIGURATION
# ============================================================

TEMPORAL_FEATURE_COUNT = 174
ORIGINAL_FEATURE_COUNT = 58


print("=" * 70)
print("STAGE 23A: TEMPORAL TEST REPRESENTATION")
print("=" * 70)


# ============================================================
# 1. LOAD SELECTED TEMPORAL FEATURE LIST
# ============================================================

print("\nLoading selected temporal feature list...")

with open(
    SELECTED_FEATURES_FILE,
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
) == TEMPORAL_FEATURE_COUNT

assert len(
    set(selected_temporal_features)
) == TEMPORAL_FEATURE_COUNT


print(
    f"Selected temporal features: "
    f"{len(selected_temporal_features)}"
)


# ============================================================
# 2. TEMPORAL FEATURE GENERATION
# ============================================================

def create_temporal_features(
    df,
    original_features
):

    timestamps = pd.to_datetime(
        df["timestamp"]
    )

    # --------------------------------------------------------
    # Detect continuous sequences
    # --------------------------------------------------------

    time_diff = (
        timestamps
        .diff()
        .dt.total_seconds()
    )

    sequence_id = (
        time_diff
        .ne(1)
        .cumsum()
    )

    sequence_count = (
        sequence_id.nunique()
    )

    print(
        f"Continuous sequences: "
        f"{sequence_count}"
    )

    print(
        f"Temporal gaps/boundaries: "
        f"{sequence_count - 1}"
    )

    # --------------------------------------------------------
    # Build temporal features
    # --------------------------------------------------------

    temporal_data = {}

    for feature in original_features:

        series = df[
            feature
        ]

        grouped = series.groupby(
            sequence_id,
            sort=False
        )

        # First difference
        diff = grouped.diff()

        temporal_data[
            f"{feature}__diff_1s"
        ] = diff

        # Absolute first difference
        temporal_data[
            f"{feature}__abs_diff_1s"
        ] = diff.abs()

        # Rolling 5-second standard deviation
        rolling_std = (
            grouped
            .rolling(
                window=5,
                min_periods=5
            )
            .std()
            .reset_index(
                level=0,
                drop=True
            )
        )

        temporal_data[
            f"{feature}__rolling_std_5s"
        ] = rolling_std


    temporal_df = pd.DataFrame(
        temporal_data,
        index=df.index
    )

    return (
        temporal_df,
        sequence_id,
        sequence_count
    )


# ============================================================
# 3. PROCESS EACH TEST SET
# ============================================================

schema_records = []


for dataset_name in [
    "test1",
    "test2"
]:

    print(
        "\n" + "=" * 70
    )

    print(
        f"PROCESSING {dataset_name.upper()}"
    )

    print(
        "=" * 70
    )

    input_file = TEST_FILES[
        dataset_name
    ]

    output_file = OUTPUT_FILES[
        dataset_name
    ]


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading model-ready test data..."
    )

    df = pd.read_csv(
        input_file
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )


    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    assert "timestamp" in df.columns
    assert "label" in df.columns

    original_features = [
        col
        for col in df.columns
        if col not in [
            "timestamp",
            "label"
        ]
    ]

    assert len(
        original_features
    ) == ORIGINAL_FEATURE_COUNT


    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        df["timestamp"]
    )

    assert timestamps.notna().all()

    print(
        "PASS: Valid timestamps"
    )


    # --------------------------------------------------------
    # Label isolation
    # --------------------------------------------------------

    labels = df["label"].copy()

    assert set(
        labels.dropna().unique()
    ).issubset({
        0,
        1
    })

    print(
        "PASS: Labels isolated from feature generation"
    )


    # --------------------------------------------------------
    # Generate temporal features
    # --------------------------------------------------------

    temporal_df, sequence_id, sequence_count = (
        create_temporal_features(
            df,
            original_features
        )
    )


    assert temporal_df.shape == (
        len(df),
        TEMPORAL_FEATURE_COUNT
    )


    # --------------------------------------------------------
    # Verify expected temporal schema
    # --------------------------------------------------------

    assert set(
        temporal_df.columns
    ) == set(
        selected_temporal_features
    )

    # Reorder exactly according to selected feature list
    temporal_df = temporal_df[
        selected_temporal_features
    ]


    # --------------------------------------------------------
    # Construct final evaluation dataset
    # --------------------------------------------------------

    final_df = pd.concat(
        [
            df[
                [
                    "timestamp"
                ]
                + original_features
            ],
            temporal_df,
            labels.rename("label")
        ],
        axis=1
    )


    expected_columns = (
        1
        + ORIGINAL_FEATURE_COUNT
        + TEMPORAL_FEATURE_COUNT
        + 1
    )

    assert final_df.shape == (
        len(df),
        expected_columns
    )

    assert expected_columns == 234


    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    assert final_df.columns[0] == (
        "timestamp"
    )

    assert final_df.columns[-1] == (
        "label"
    )

    assert (
        final_df.columns.nunique()
        == 234
    )


    # --------------------------------------------------------
    # Check temporal NaNs
    # --------------------------------------------------------

    temporal_nan_count = (
        temporal_df
        .isna()
        .sum()
        .sum()
    )

    print(
        f"\nTemporal NaN count: "
        f"{temporal_nan_count:,}"
    )


    # --------------------------------------------------------
    # Check infinite values
    # --------------------------------------------------------

    numerical_columns = [
        col
        for col in final_df.columns
        if col not in [
            "timestamp",
            "label"
        ]
    ]

    infinite_count = np.isinf(
        final_df[
            numerical_columns
        ].to_numpy()
    ).sum()

    assert infinite_count == 0

    print(
        "PASS: No infinite values"
    )


    # --------------------------------------------------------
    # Check original features
    # --------------------------------------------------------

    original_nan_count = (
        df[
            original_features
        ]
        .isna()
        .sum()
        .sum()
    )

    assert original_nan_count == 0

    print(
        "PASS: Original features contain no NaNs"
    )


    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    label_counts = (
        final_df["label"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nLabel distribution:"
    )

    print(
        label_counts.to_string()
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaving temporal test dataset..."
    )

    final_df.to_csv(
        output_file,
        index=False
    )

    print(
        output_file
    )


    # --------------------------------------------------------
    # Record schema information
    # --------------------------------------------------------

    for index, column in enumerate(
        final_df.columns,
        start=1
    ):

        if column == "timestamp":

            category = "timestamp"

        elif column == "label":

            category = "evaluation_label"

        elif column in original_features:

            category = "original_model_ready"

        else:

            category = "temporal"

        schema_records.append({
            "dataset": dataset_name,
            "column_index": index,
            "feature": column,
            "category": category
        })


    # --------------------------------------------------------
    # Final dataset validation
    # --------------------------------------------------------

    print(
        "\nVALIDATION"
    )

    assert len(final_df) == len(df)

    assert final_df.shape[1] == 234

    assert (
        final_df["timestamp"]
        .notna()
        .all()
    )

    assert (
        final_df["label"]
        .notna()
        .all()
    )

    assert (
        final_df.columns.nunique()
        == 234
    )

    print(
        f"PASS: {dataset_name} final shape "
        f"{len(final_df):,} × 234"
    )

    print(
        "PASS: 58 original features"
    )

    print(
        "PASS: 174 temporal features"
    )

    print(
        "PASS: Evaluation label preserved"
    )

    print(
        "PASS: No infinite values"
    )

    print(
        "PASS: Test labels were not used to create features"
    )


# ============================================================
# 4. SAVE COMMON SCHEMA
# ============================================================

schema_df = pd.DataFrame(
    schema_records
)

schema_df.to_csv(
    SCHEMA_FILE,
    index=False
)


# ============================================================
# 5. FINAL VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23A FINAL VALIDATION"
)

print(
    "=" * 70
)

for dataset_name in [
    "test1",
    "test2"
]:

    output_file = OUTPUT_FILES[
        dataset_name
    ]

    check_df = pd.read_csv(
        output_file,
        nrows=5
    )

    assert check_df.shape[1] == 234

    assert (
        check_df.columns[0]
        == "timestamp"
    )

    assert (
        check_df.columns[-1]
        == "label"
    )

    print(
        f"PASS: {dataset_name} output verified"
    )


print(
    "\nPASS: Test 1 temporal representation created"
)

print(
    "PASS: Test 2 temporal representation created"
)

print(
    "PASS: 174 selected temporal features applied"
)

print(
    "PASS: Test labels retained only for evaluation"
)

print(
    "PASS: No label-derived feature construction"
)

print(
    "PASS: Schema reports created"
)

print(
    "\nSchema:"
)

print(
    SCHEMA_FILE
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23A: COMPLETE"
)

print(
    "=" * 70
)