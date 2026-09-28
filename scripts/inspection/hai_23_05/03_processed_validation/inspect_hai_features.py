from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
)

FILE = PROCESSED_DIR / "hai-test1_aligned.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HAI 23.05 SCADA FEATURE INSPECTION")
    print("=" * 70)

    df = pd.read_csv(FILE)

    # Remove timestamp and label
    feature_columns = [
        col for col in df.columns
        if col not in ["timestamp", "label"]
    ]

    print(f"\nTotal SCADA features: {len(feature_columns)}")

    # --------------------------------------------------------
    # Group features by process prefix
    # --------------------------------------------------------

    processes = {
        "P1": [],
        "P2": [],
        "P3": [],
        "P4": [],
        "Other": []
    }

    for column in feature_columns:

        prefix = column.split("_")[0]

        if prefix in processes:
            processes[prefix].append(column)
        else:
            processes["Other"].append(column)

    # --------------------------------------------------------
    # Print process groups
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURES BY PROCESS")
    print("=" * 70)

    for process, features in processes.items():

        if not features:
            continue

        print(f"\n{process}: {len(features)} features")

        for feature in features:
            print(f"  - {feature}")

    # --------------------------------------------------------
    # Prefix / tag-type analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TAG TYPE ANALYSIS")
    print("=" * 70)

    tag_types = {}

    for column in feature_columns:

        parts = column.split("_")

        if len(parts) >= 2:

            tag_type = parts[1]

            tag_types.setdefault(tag_type, []).append(column)

    for tag_type, features in sorted(tag_types.items()):

        print(
            f"{tag_type:10s} : "
            f"{len(features):2d} features"
        )

    # --------------------------------------------------------
    # Feature name table
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL SCADA FEATURES")
    print("=" * 70)

    for index, column in enumerate(feature_columns, start=1):

        print(f"{index:02d}. {column}")

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE VALUE RANGES")
    print("=" * 70)

    for column in feature_columns:

        minimum = df[column].min()
        maximum = df[column].max()
        mean = df[column].mean()

        print(
            f"{column:20s} "
            f"min={minimum:12.5f} "
            f"max={maximum:12.5f} "
            f"mean={mean:12.5f}"
        )

    print("\n" + "=" * 70)
    print("HAI 23.05 FEATURE INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()