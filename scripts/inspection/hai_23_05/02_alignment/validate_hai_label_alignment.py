import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_DIR = (
    PROJECT_ROOT
    / "data/raw/hai/hai-23.05"
)


TEST_PAIRS = [
    ("hai-test1.csv", "label-test1.csv"),
    ("hai-test2.csv", "label-test2.csv")
]


print("=" * 80)
print("HAI 23.05 LABEL / SCADA ALIGNMENT VALIDATION")
print("=" * 80)


for scada_file, label_file in TEST_PAIRS:

    print("\n" + "=" * 80)
    print("SCADA:", scada_file)
    print("LABEL:", label_file)
    print("=" * 80)

    scada = pd.read_csv(
        INPUT_DIR / scada_file
    )

    labels = pd.read_csv(
        INPUT_DIR / label_file
    )

    # --------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------

    scada["timestamp"] = pd.to_datetime(
        scada["timestamp"],
        errors="coerce"
    )

    labels["timestamp"] = pd.to_datetime(
        labels["timestamp"],
        errors="coerce"
    )

    # --------------------------------------------------
    # Basic information
    # --------------------------------------------------

    print("\n--- ROW COUNTS ---")

    print("SCADA rows:", len(scada))
    print("Label rows:", len(labels))

    print("\n--- TIMESTAMP COUNTS ---")

    print(
        "SCADA unique timestamps:",
        scada["timestamp"].nunique()
    )

    print(
        "Label unique timestamps:",
        labels["timestamp"].nunique()
    )

    # --------------------------------------------------
    # Timestamp ranges
    # --------------------------------------------------

    print("\n--- TIMESTAMP RANGES ---")

    print(
        "SCADA:",
        scada["timestamp"].min(),
        "->",
        scada["timestamp"].max()
    )

    print(
        "Labels:",
        labels["timestamp"].min(),
        "->",
        labels["timestamp"].max()
    )

    # --------------------------------------------------
    # First 20 rows
    # --------------------------------------------------

    print("\n--- FIRST 20 SCADA TIMESTAMPS ---")

    print(
        scada[
            ["timestamp"]
        ].head(20).to_string(index=False)
    )

    print("\n--- FIRST 20 LABEL ROWS ---")

    print(
        labels[
            ["timestamp", "label"]
        ].head(20).to_string(index=False)
    )

    # --------------------------------------------------
    # Last 10 rows
    # --------------------------------------------------

    print("\n--- LAST 10 SCADA TIMESTAMPS ---")

    print(
        scada[
            ["timestamp"]
        ].tail(10).to_string(index=False)
    )

    print("\n--- LAST 10 LABEL ROWS ---")

    print(
        labels[
            ["timestamp", "label"]
        ].tail(10).to_string(index=False)
    )

    # --------------------------------------------------
    # Exact timestamp overlap
    # --------------------------------------------------

    scada_times = set(
        scada["timestamp"]
    )

    label_times = set(
        labels["timestamp"]
    )

    common_times = (
        scada_times
        & label_times
    )

    print("\n--- EXACT TIMESTAMP OVERLAP ---")

    print(
        "Common timestamps:",
        len(common_times)
    )

    print(
        "SCADA timestamps without exact label timestamp:",
        len(scada_times - label_times)
    )

    print(
        "Label timestamps without exact SCADA timestamp:",
        len(label_times - scada_times)
    )

    # --------------------------------------------------
    # Minute-level alignment
    # --------------------------------------------------

    scada_minutes = (
        scada["timestamp"]
        .dt.floor("min")
    )

    label_minutes = (
        labels["timestamp"]
        .dt.floor("min")
    )

    common_minutes = (
        set(scada_minutes)
        & set(label_minutes)
    )

    print("\n--- MINUTE-LEVEL OVERLAP ---")

    print(
        "Common minutes:",
        len(common_minutes)
    )

    print(
        "Unique SCADA minutes:",
        scada_minutes.nunique()
    )

    print(
        "Unique label minutes:",
        label_minutes.nunique()
    )

    # --------------------------------------------------
    # Labels per timestamp
    # --------------------------------------------------

    print("\n--- LABEL TIMESTAMP REPETITION ---")

    repetition_counts = (
        labels["timestamp"]
        .value_counts()
        .value_counts()
        .sort_index()
    )

    print(
        repetition_counts.head(20)
    )

    # --------------------------------------------------
    # Label distribution by timestamp
    # --------------------------------------------------

    print("\n--- LABELS PER TIMESTAMP ---")

    label_per_timestamp = (
        labels
        .groupby("timestamp")["label"]
        .agg(
            rows="size",
            unique_labels="nunique",
            label_sum="sum"
        )
    )

    print(
        label_per_timestamp.head(20)
    )

    print("\n--- TIMESTAMPS CONTAINING LABEL 1 ---")

    attack_timestamps = (
        label_per_timestamp[
            label_per_timestamp["label_sum"] > 0
        ]
    )

    print(
        attack_timestamps.head(20)
    )

    print(
        "\nTotal timestamps containing label 1:",
        len(attack_timestamps)
    )


print("\n" + "=" * 80)
print("HAI 23.05 ALIGNMENT VALIDATION COMPLETE")
print("=" * 80)