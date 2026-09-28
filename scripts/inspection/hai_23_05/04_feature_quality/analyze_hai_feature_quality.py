from pathlib import Path
import pandas as pd
import numpy as np


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
    print("HAI 23.05 FEATURE QUALITY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(FILE)

    # --------------------------------------------------------
    # Select SCADA features
    # --------------------------------------------------------

    feature_columns = [
        col for col in df.columns
        if col not in ["timestamp", "label"]
    ]

    feature_df = df[feature_columns]

    print(f"\nTotal SCADA features: {len(feature_columns)}")
    print(f"Total rows: {len(df):,}")

    # --------------------------------------------------------
    # 1. Unique value analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[1] UNIQUE VALUE ANALYSIS")
    print("=" * 70)

    unique_counts = feature_df.nunique()

    constant_features = unique_counts[unique_counts == 1]

    print(f"\nConstant features: {len(constant_features)}")

    if len(constant_features) > 0:

        for feature, count in constant_features.items():

            value = feature_df[feature].iloc[0]

            print(
                f"  {feature:25s} "
                f"value={value}"
            )

    # --------------------------------------------------------
    # 2. Near-constant analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[2] NEAR-CONSTANT FEATURE ANALYSIS")
    print("=" * 70)

    near_constant_features = []

    threshold = 0.99

    for feature in feature_columns:

        value_counts = (
            feature_df[feature]
            .value_counts(normalize=True)
        )

        most_common_ratio = value_counts.iloc[0]

        if most_common_ratio >= threshold:

            near_constant_features.append(
                (feature, most_common_ratio)
            )

    print(
        f"\nFeatures where one value occurs "
        f">= {threshold * 100:.0f}% of the time: "
        f"{len(near_constant_features)}"
    )

    for feature, ratio in near_constant_features:

        print(
            f"  {feature:25s} "
            f"{ratio * 100:.2f}%"
        )

    # --------------------------------------------------------
    # 3. Binary / status feature analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[3] BINARY / STATUS-LIKE FEATURES")
    print("=" * 70)

    binary_features = []

    for feature in feature_columns:

        unique_values = feature_df[feature].nunique()

        if unique_values == 2:

            values = sorted(
                feature_df[feature].unique().tolist()
            )

            binary_features.append(
                (feature, values)
            )

    print(f"\nBinary features: {len(binary_features)}")

    for feature, values in binary_features:

        print(
            f"  {feature:25s} "
            f"values={values}"
        )

    # --------------------------------------------------------
    # 4. Constant / binary / continuous classification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[4] FEATURE TYPE SUMMARY")
    print("=" * 70)

    constant_count = 0
    binary_count = 0
    continuous_count = 0

    for feature in feature_columns:

        unique_values = feature_df[feature].nunique()

        if unique_values == 1:

            constant_count += 1

        elif unique_values == 2:

            binary_count += 1

        else:

            continuous_count += 1

    print(f"\nConstant   : {constant_count}")
    print(f"Binary     : {binary_count}")
    print(f"Continuous : {continuous_count}")
    print(
        f"Total      : "
        f"{constant_count + binary_count + continuous_count}"
    )

    # --------------------------------------------------------
    # 5. Missing values
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[5] MISSING VALUE CHECK")
    print("=" * 70)

    missing = feature_df.isna().sum()

    total_missing = missing.sum()

    print(f"\nTotal missing values: {total_missing}")

    if total_missing > 0:

        print("\nFeatures containing missing values:")

        print(
            missing[missing > 0]
            .sort_values(ascending=False)
        )

    # --------------------------------------------------------
    # 6. Infinite values
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[6] INFINITE VALUE CHECK")
    print("=" * 70)

    infinite_count = np.isinf(feature_df).sum()

    total_infinite = infinite_count.sum()

    print(f"\nTotal infinite values: {total_infinite}")

    if total_infinite > 0:

        print("\nFeatures containing infinite values:")

        print(
            infinite_count[infinite_count > 0]
            .sort_values(ascending=False)
        )

    # --------------------------------------------------------
    # 7. Feature ranges
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("[7] FEATURE RANGE SUMMARY")
    print("=" * 70)

    range_summary = pd.DataFrame({
        "feature": feature_columns,
        "unique_values": [
            feature_df[col].nunique()
            for col in feature_columns
        ],
        "minimum": [
            feature_df[col].min()
            for col in feature_columns
        ],
        "maximum": [
            feature_df[col].max()
            for col in feature_columns
        ],
        "mean": [
            feature_df[col].mean()
            for col in feature_columns
        ],
        "std": [
            feature_df[col].std()
            for col in feature_columns
        ]
    })

    print(
        range_summary.to_string(index=False)
    )

    # --------------------------------------------------------
    # 8. Save analysis
    # --------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "features"
        / "hai"
        / "hai-23.05"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "hai_2305_feature_quality.csv"
    )

    range_summary.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(f"\nSaved feature-quality summary:")
    print(output_file)

    print("\n" + "=" * 70)
    print("HAI 23.05 FEATURE QUALITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()