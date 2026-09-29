from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STAGE 22D
# HAI 23.05 - Temporal Extreme-Change Investigation
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"
    r"\data\features\hai\hai-23.05"
    r"\model_ready"
    r"\hai_2305_training_model_ready.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"
    r"\data\features\hai\hai-23.05"
    r"\temporal_representation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_extreme_change_events.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_FEATURES = [
    "P3_FIT01",
    "P3_PIT01",
    "P3_LCV01D",
    "P3_LCP01D",
    "P4_ST_GOV",
    "P3_LIT01",
]

TOP_N = 20


print("=" * 70)
print("STAGE 22D: TEMPORAL EXTREME-CHANGE INVESTIGATION")
print("=" * 70)


# ============================================================
# 1. LOAD
# ============================================================

print("\nLoading training model-ready data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. VALIDATE INPUT
# ============================================================

assert "timestamp" in df.columns

timestamps = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

assert timestamps.notna().all()

for feature in TARGET_FEATURES:
    assert feature in df.columns

print(
    f"Target features: {len(TARGET_FEATURES)}"
)


# ============================================================
# 3. DETECT SEQUENCE BOUNDARIES
# ============================================================

print("\nDetecting sequence boundaries...")

time_diff = timestamps.diff()

boundary_mask = (
    time_diff > pd.Timedelta(seconds=1)
)

sequence_id = (
    boundary_mask.astype(int)
    .cumsum()
)

df["_sequence_id"] = sequence_id


print(
    f"Continuous sequences : "
    f"{df['_sequence_id'].nunique()}"
)

print(
    f"Temporal boundaries  : "
    f"{int(boundary_mask.sum())}"
)


# ============================================================
# 4. INVESTIGATE EACH FEATURE
# ============================================================

all_events = []

for feature in TARGET_FEATURES:

    print("\n" + "-" * 70)

    print(
        f"Investigating: {feature}"
    )

    print("-" * 70)


    # --------------------------------------------------------
    # Calculate 1-second difference
    # --------------------------------------------------------

    previous_value = (
        df.groupby("_sequence_id")[feature]
        .shift(1)
    )

    current_value = df[feature]

    difference = (
        current_value
        - previous_value
    )

    absolute_difference = (
        difference.abs()
    )


    # --------------------------------------------------------
    # Build event table
    # --------------------------------------------------------

    events = pd.DataFrame({
        "timestamp": timestamps,
        "sequence_id": df["_sequence_id"],
        "feature": feature,
        "previous_value": previous_value,
        "current_value": current_value,
        "difference_1s": difference,
        "absolute_difference_1s":
            absolute_difference
    })


    # Remove sequence-boundary rows

    events = events[
        events["previous_value"].notna()
    ].copy()


    # Sort by magnitude

    events = (
        events
        .sort_values(
            "absolute_difference_1s",
            ascending=False
        )
        .head(TOP_N)
    )


    # --------------------------------------------------------
    # Print top events
    # --------------------------------------------------------

    display_columns = [
        "timestamp",
        "sequence_id",
        "previous_value",
        "current_value",
        "difference_1s",
        "absolute_difference_1s"
    ]

    print(
        events[
            display_columns
        ].to_string(
            index=False
        )
    )


    # Store

    all_events.append(events)


# ============================================================
# 5. COMBINE
# ============================================================

events_df = pd.concat(
    all_events,
    ignore_index=True
)


# ============================================================
# 6. SORT GLOBAL EVENTS
# ============================================================

events_df = (
    events_df
    .sort_values(
        "absolute_difference_1s",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 7. SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

events_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL EXTREME EVENTS")
print("=" * 70)

print(
    events_df[
        [
            "feature",
            "timestamp",
            "sequence_id",
            "previous_value",
            "current_value",
            "difference_1s",
            "absolute_difference_1s"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 9. RANGE / JUMP SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTREME SUMMARY")
print("=" * 70)

summary_records = []

for feature in TARGET_FEATURES:

    feature_events = events_df[
        events_df["feature"] == feature
    ]

    max_event = feature_events.iloc[0]

    summary_records.append({
        "feature": feature,
        "largest_abs_change":
            float(
                max_event[
                    "absolute_difference_1s"
                ]
            ),
        "previous_value":
            float(
                max_event[
                    "previous_value"
                ]
            ),
        "current_value":
            float(
                max_event[
                    "current_value"
                ]
            ),
        "timestamp":
            max_event["timestamp"],
        "sequence_id":
            int(
                max_event["sequence_id"]
            )
    })


summary_df = pd.DataFrame(
    summary_records
)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 10. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

assert len(events_df) == (
    len(TARGET_FEATURES) * TOP_N
)

assert (
    events_df["absolute_difference_1s"]
    .notna()
    .all()
)

assert np.isfinite(
    events_df[
        "absolute_difference_1s"
    ].to_numpy()
).all()

assert (
    events_df["sequence_id"]
    .between(0, 3)
    .all()
)

print(
    "PASS: All target features investigated"
)

print(
    "PASS: Top 20 events recorded per feature"
)

print(
    "PASS: Sequence boundaries respected"
)

print(
    "PASS: No boundary rows included"
)

print(
    "PASS: Extreme-event values are finite"
)

print(
    "PASS: No test data used"
)

print(
    "\nOutput:"
)

print(OUTPUT_FILE)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 22D: COMPLETE"
)

print(
    "=" * 70
)