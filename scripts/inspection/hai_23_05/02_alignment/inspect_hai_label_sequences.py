import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_DIR = (
    PROJECT_ROOT
    / "data/raw/hai/hai-23.05"
)

LABEL_FILE = (
    INPUT_DIR
    / "label-test2.csv"
)


print("=" * 80)
print("HAI 23.05 TEST2 LABEL SEQUENCE INSPECTION")
print("=" * 80)


df = pd.read_csv(LABEL_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


target_minutes = [
    "2022-08-17 01:27:00",
    "2022-08-17 01:28:00",
    "2022-08-17 01:29:00"
]


for minute in target_minutes:

    minute = pd.Timestamp(minute)

    subset = df[
        df["timestamp"] == minute
    ].copy()

    print("\n" + "-" * 80)
    print("MINUTE:", minute)
    print("-" * 80)

    print("Rows:", len(subset))

    print("\nLabels in stored order:")

    print(
        subset["label"]
        .to_list()
    )

    print("\nLabel counts:")

    print(
        subset["label"]
        .value_counts()
        .sort_index()
    )


print("\n" + "=" * 80)
print("SEQUENCE INSPECTION COMPLETE")
print("=" * 80)