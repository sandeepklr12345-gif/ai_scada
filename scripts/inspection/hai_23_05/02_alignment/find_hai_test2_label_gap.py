import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

LABEL_FILE = (
    PROJECT_ROOT
    / "data/raw/hai/hai-23.05/label-test2.csv"
)

SCADA_FILE = (
    PROJECT_ROOT
    / "data/raw/hai/hai-23.05/hai-test2.csv"
)

print("=" * 80)
print("HAI 23.05 TEST2 INTERNAL LABEL GAP INSPECTION")
print("=" * 80)

labels = pd.read_csv(LABEL_FILE)
scada = pd.read_csv(SCADA_FILE)

labels["timestamp"] = pd.to_datetime(labels["timestamp"])
scada["timestamp"] = pd.to_datetime(scada["timestamp"])


# ---------------------------------------------------------
# FIND MINUTES WITH UNEXPECTED NUMBER OF LABEL ROWS
# ---------------------------------------------------------

counts = (
    labels["timestamp"]
    .value_counts()
    .sort_index()
)

print("\nMinutes with label counts other than 60:")

unusual = counts[counts != 60]

print(unusual)


# ---------------------------------------------------------
# IDENTIFY THE INTERNAL 59-ROW MINUTE
# ---------------------------------------------------------

internal_59 = unusual[
    (unusual.index > labels["timestamp"].min())
    & (unusual.index < labels["timestamp"].max())
    & (unusual == 59)
]

print("\n" + "-" * 80)
print("INTERNAL 59-ROW MINUTE")
print("-" * 80)

print(internal_59)


if len(internal_59) == 0:
    print("\nNo internal 59-row minute found.")
    print("Inspection complete.")
    raise SystemExit


gap_minute = internal_59.index[0]

print("\nGap minute:", gap_minute)
print("Label rows:", counts.loc[gap_minute])


# ---------------------------------------------------------
# SHOW LABELS FROM GAP MINUTE
# ---------------------------------------------------------

gap_labels = (
    labels[
        labels["timestamp"] == gap_minute
    ]
    .reset_index(drop=True)
)

print("\nLabels stored for this minute:")
print(gap_labels.to_string(index=False))


# ---------------------------------------------------------
# SHOW PREVIOUS / CURRENT / NEXT MINUTE
# ---------------------------------------------------------

previous_minute = gap_minute - pd.Timedelta(minutes=1)
next_minute = gap_minute + pd.Timedelta(minutes=1)

print("\n" + "-" * 80)
print("PREVIOUS MINUTE")
print("-" * 80)

previous = labels[
    labels["timestamp"] == previous_minute
].reset_index(drop=True)

print("Minute:", previous_minute)
print("Rows:", len(previous))
print("Labels:", previous["label"].tolist())


print("\n" + "-" * 80)
print("GAP MINUTE")
print("-" * 80)

print("Minute:", gap_minute)
print("Rows:", len(gap_labels))
print("Labels:", gap_labels["label"].tolist())


print("\n" + "-" * 80)
print("NEXT MINUTE")
print("-" * 80)

next_group = labels[
    labels["timestamp"] == next_minute
].reset_index(drop=True)

print("Minute:", next_minute)
print("Rows:", len(next_group))
print("Labels:", next_group["label"].tolist())


# ---------------------------------------------------------
# DETERMINE EXPECTED SCADA TIMESTAMP RANGE
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("CORRESPONDING SCADA TIMESTAMPS")
print("-" * 80)

expected_start = gap_minute + pd.Timedelta(seconds=1)
expected_end = gap_minute + pd.Timedelta(minutes=1)

print("Expected SCADA range:")
print(expected_start)
print("to")
print(expected_end)

scada_window = scada[
    (scada["timestamp"] >= expected_start)
    & (scada["timestamp"] <= expected_end)
][["timestamp"]]

print("\nSCADA rows in this range:")
print(scada_window.to_string(index=False))


print("\n" + "=" * 80)
print("INTERNAL GAP INSPECTION COMPLETE")
print("=" * 80)