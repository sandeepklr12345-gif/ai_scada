import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_DIR = PROJECT_ROOT / "data/raw/hai/hai-23.05"

SCADA_FILE = INPUT_DIR / "hai-test2.csv"
LABEL_FILE = INPUT_DIR / "label-test2.csv"

print("=" * 80)
print("HAI 23.05 TEST2 BOUNDARY INSPECTION")
print("=" * 80)

scada = pd.read_csv(SCADA_FILE)
labels = pd.read_csv(LABEL_FILE)

scada["timestamp"] = pd.to_datetime(scada["timestamp"])
labels["timestamp"] = pd.to_datetime(labels["timestamp"])

# FIRST LABEL MINUTE
first_label_time = labels["timestamp"].min()
first_labels = labels[
    labels["timestamp"] == first_label_time
].reset_index(drop=True)

print("\n" + "-" * 80)
print("FIRST LABEL MINUTE")
print("-" * 80)

print("Minute:", first_label_time)
print("Rows:", len(first_labels))
print("Labels:", first_labels["label"].tolist())

print("\nCorresponding first SCADA timestamps:")

first_scada = scada.head(65)[["timestamp"]]

print(
    first_scada.to_string(index=False)
)

# LAST LABEL MINUTE
last_label_time = labels["timestamp"].max()

last_labels = labels[
    labels["timestamp"] == last_label_time
].reset_index(drop=True)

print("\n" + "-" * 80)
print("LAST LABEL MINUTE")
print("-" * 80)

print("Minute:", last_label_time)
print("Rows:", len(last_labels))
print("Labels:", last_labels["label"].tolist())

print("\nCorresponding last SCADA timestamps:")

last_scada = scada.tail(65)[["timestamp"]]

print(
    last_scada.to_string(index=False)
)

# LAST LABEL TIMESTAMP COUNTS
print("\n" + "-" * 80)
print("LAST LABEL TIMESTAMP DETAILS")
print("-" * 80)

print(
    labels["timestamp"]
    .value_counts()
    .sort_index()
    .tail(5)
)

# FINAL SCADA TIMESTAMPS
print("\n" + "-" * 80)
print("FINAL SCADA TIMESTAMPS")
print("-" * 80)

print(
    scada["timestamp"]
    .tail(10)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("BOUNDARY INSPECTION COMPLETE")
print("=" * 80)