from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCADA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
    / "hai-test1.csv"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
    / "label-test1.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAPPING_OUTPUT = OUTPUT_DIR / "hai_test1_label_mapping.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "hai_test1_label_summary.csv"


def main():

    print("=" * 80)
    print("HAI 23.05 ATTACK LABEL ANALYSIS")
    print("=" * 80)

    print("\nLoading SCADA data...")
    scada = pd.read_csv(SCADA_PATH)

    print("Loading labels...")
    labels = pd.read_csv(LABEL_PATH)

    print(f"SCADA shape : {scada.shape}")
    print(f"Label shape : {labels.shape}")

    scada["timestamp"] = pd.to_datetime(scada["timestamp"])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])

    # ------------------------------------------------------------
    # TIMESTAMP VALIDATION
    # ------------------------------------------------------------

    if not scada["timestamp"].is_unique:
        raise ValueError("SCADA timestamps are not unique.")

    if not labels["timestamp"].is_unique:
        raise ValueError("Label timestamps are not unique.")

    if not scada["timestamp"].equals(labels["timestamp"]):
        raise ValueError(
            "SCADA and label timestamps are not perfectly aligned."
        )

    print("\nTimestamp alignment: PASS")

    # ------------------------------------------------------------
    # LABEL INSPECTION
    # ------------------------------------------------------------

    print("\nLabel columns:")
    print(labels.columns.tolist())

    label_columns = [
        column
        for column in labels.columns
        if column != "timestamp"
    ]

    if len(label_columns) != 1:
        raise ValueError(
            f"Expected one label column, found: {label_columns}"
        )

    label_column = label_columns[0]

    print(f"\nLabel column: {label_column}")

    print("\nLabel distribution:")
    print(labels[label_column].value_counts(dropna=False))

    # ------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------

    merged = scada.merge(
        labels,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    print(f"\nMerged shape: {merged.shape}")

    # ------------------------------------------------------------
    # ATTACK INTERVALS
    # ------------------------------------------------------------

    merged["label_change"] = (
        merged[label_column]
        .ne(merged[label_column].shift())
    )

    merged["segment_id"] = (
        merged["label_change"]
        .cumsum()
    )

    segments = (
        merged
        .groupby("segment_id")
        .agg(
            start_timestamp=("timestamp", "first"),
            end_timestamp=("timestamp", "last"),
            label=(label_column, "first"),
            rows=("timestamp", "size"),
        )
        .reset_index(drop=True)
    )

    segments["duration_seconds"] = segments["rows"]

    print("\nLabel segments:")
    print(segments.to_string(index=False))

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    merged.to_csv(
        MAPPING_OUTPUT,
        index=False,
    )

    segments.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\nOutputs:")
    print(MAPPING_OUTPUT)
    print(SUMMARY_OUTPUT)

    print("\n" + "=" * 80)
    print("HAI 23.05 ATTACK LABEL ANALYSIS: COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()