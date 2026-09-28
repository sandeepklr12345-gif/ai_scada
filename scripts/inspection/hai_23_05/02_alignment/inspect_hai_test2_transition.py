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

labels = pd.read_csv(LABEL_FILE)
scada = pd.read_csv(SCADA_FILE)

labels["timestamp"] = pd.to_datetime(labels["timestamp"])
scada["timestamp"] = pd.to_datetime(scada["timestamp"])


print("=" * 80)
print("HAI TEST2 FIRST-MINUTE TRANSITION VALIDATION")
print("=" * 80)


# ---------------------------------------------------------
# FIRST LABEL GROUP
# ---------------------------------------------------------

first_minute = labels["timestamp"].min()
second_minute = first_minute + pd.Timedelta(minutes=1)

first_group = (
    labels[labels["timestamp"] == first_minute]
    .reset_index(drop=True)
)

second_group = (
    labels[labels["timestamp"] == second_minute]
    .reset_index(drop=True)
)


print("\nFIRST LABEL GROUP")
print("-" * 80)

print("Minute:", first_minute)
print("Rows:", len(first_group))
print("First 10 labels:", first_group["label"].head(10).tolist())
print("Last 10 labels:", first_group["label"].tail(10).tolist())


print("\nSECOND LABEL GROUP")
print("-" * 80)

print("Minute:", second_minute)
print("Rows:", len(second_group))
print("First 10 labels:", second_group["label"].head(10).tolist())
print("Last 10 labels:", second_group["label"].tail(10).tolist())


# ---------------------------------------------------------
# EXPECTED MAPPING
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("EXPECTED TIMESTAMP MAPPING")
print("-" * 80)

print("\nFirst group:")
for i in range(5):
    label_time = first_minute + pd.Timedelta(seconds=i)
    scada_time = label_time + pd.Timedelta(seconds=1)

    print(
        f"label {label_time} -> SCADA {scada_time}"
    )

print("\nBoundary:")
print(
    f"last first-group label "
    f"{first_minute + pd.Timedelta(seconds=58)} "
    f"-> SCADA "
    f"{first_minute + pd.Timedelta(seconds=59)}"
)

print(
    f"first second-group label "
    f"{second_minute} "
    f"-> SCADA "
    f"{second_minute}"
)


# ---------------------------------------------------------
# CHECK SCADA BOUNDARY
# ---------------------------------------------------------

print("\n" + "-" * 80)
print("CORRESPONDING SCADA ROWS")
print("-" * 80)

start = first_minute + pd.Timedelta(seconds=1)
end = second_minute + pd.Timedelta(seconds=5)

scada_window = scada[
    (scada["timestamp"] >= start)
    & (scada["timestamp"] <= end)
][["timestamp"]]

print(
    scada_window.to_string(index=False)
)


print("\n" + "=" * 80)
print("TRANSITION VALIDATION COMPLETE")
print("=" * 80)