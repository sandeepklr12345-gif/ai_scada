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

TEST1_FILE = PROCESSED_DIR / "hai-test1_aligned.csv"
TEST2_FILE = PROCESSED_DIR / "hai-test2_aligned.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HAI 23.05 TEST 1 vs TEST 2 FEATURE STABILITY")
    print("=" * 70)

    test1 = pd.read_csv(TEST1_FILE)
    test2 = pd.read_csv(TEST2_FILE)

    features = [
        col
        for col in test1.columns
        if col not in ["timestamp", "label"]
    ]

    # --------------------------------------------------------
    # Basic consistency
    # --------------------------------------------------------

    print("\n[1] FEATURE STRUCTURE")

    test1_features = set(
        col for col in test1.columns
        if col not in ["timestamp", "label"]
    )

    test2_features = set(
        col for col in test2.columns
        if col not in ["timestamp", "label"]
    )

    print(f"Test 1 SCADA features: {len(test1_features)}")
    print(f"Test 2 SCADA features: {len(test2_features)}")

    only_test1 = sorted(test1_features - test2_features)
    only_test2 = sorted(test2_features - test1_features)

    print(f"Features only in Test 1: {len(only_test1)}")
    print(f"Features only in Test 2: {len(only_test2)}")

    if only_test1:
        print(only_test1)

    if only_test2:
        print(only_test2)

    # --------------------------------------------------------
    # Unique-value comparison
    # --------------------------------------------------------

    print("\n[2] CONSTANT FEATURE COMPARISON")

    test1_unique = test1[features].nunique()
    test2_unique = test2[features].nunique()

    test1_constant = set(
        test1_unique[test1_unique == 1].index
    )

    test2_constant = set(
        test2_unique[test2_unique == 1].index
    )

    constant_both = sorted(
        test1_constant & test2_constant
    )

    constant_test1_only = sorted(
        test1_constant - test2_constant
    )

    constant_test2_only = sorted(
        test2_constant - test1_constant
    )

    print(f"\nConstant in Test 1: {len(test1_constant)}")
    print(f"Constant in Test 2: {len(test2_constant)}")
    print(f"Constant in BOTH   : {len(constant_both)}")

    print("\nConstant in BOTH datasets:")

    for feature in constant_both:

        value1 = test1[feature].iloc[0]
        value2 = test2[feature].iloc[0]

        print(
            f"  {feature:25s} "
            f"Test1={value1} "
            f"Test2={value2}"
        )

    print("\nConstant only in Test 1:")

    for feature in constant_test1_only:

        print(
            f"  {feature:25s} "
            f"Test1 value={test1[feature].iloc[0]}"
        )

    print("\nConstant only in Test 2:")

    for feature in constant_test2_only:

        print(
            f"  {feature:25s} "
            f"Test2 value={test2[feature].iloc[0]}"
        )

    # --------------------------------------------------------
    # Unique-value count comparison
    # --------------------------------------------------------

    print("\n[3] UNIQUE VALUE COUNT COMPARISON")

    comparison = pd.DataFrame({
        "feature": features,
        "test1_unique_values": [
            test1_unique[col]
            for col in features
        ],
        "test2_unique_values": [
            test2_unique[col]
            for col in features
        ]
    })

    comparison["constant_test1"] = (
        comparison["test1_unique_values"] == 1
    )

    comparison["constant_test2"] = (
        comparison["test2_unique_values"] == 1
    )

    comparison["constant_both"] = (
        comparison["constant_test1"]
        & comparison["constant_test2"]
    )

    # --------------------------------------------------------
    # Basic statistics comparison
    # --------------------------------------------------------

    print("\n[4] RANGE COMPARISON")

    comparison["test1_min"] = [
        test1[col].min()
        for col in features
    ]

    comparison["test1_max"] = [
        test1[col].max()
        for col in features
    ]

    comparison["test2_min"] = [
        test2[col].min()
        for col in features
    ]

    comparison["test2_max"] = [
        test2[col].max()
        for col in features
    ]

    comparison["test1_mean"] = [
        test1[col].mean()
        for col in features
    ]

    comparison["test2_mean"] = [
        test2[col].mean()
        for col in features
    ]

    # --------------------------------------------------------
    # Features that vary in both datasets
    # --------------------------------------------------------

    varying_both = sorted(
        test1_features
        - set(constant_both)
    )

    print(
        f"\nFeatures varying in at least one dataset: "
        f"{len(varying_both)}"
    )

    # --------------------------------------------------------
    # Save result
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
        / "hai_2305_feature_stability.csv"
    )

    comparison.to_csv(
        output_file,
        index=False
    )

    print("\n[5] OUTPUT")

    print(f"Saved comparison:")
    print(output_file)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE STABILITY SUMMARY")
    print("=" * 70)

    print(f"Total SCADA features       : {len(features)}")
    print(f"Constant in Test 1         : {len(test1_constant)}")
    print(f"Constant in Test 2         : {len(test2_constant)}")
    print(f"Constant in BOTH           : {len(constant_both)}")
    print(
        f"Constant only in Test 1   : "
        f"{len(constant_test1_only)}"
    )
    print(
        f"Constant only in Test 2   : "
        f"{len(constant_test2_only)}"
    )

    print("\n" + "=" * 70)
    print("HAI 23.05 FEATURE STABILITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()