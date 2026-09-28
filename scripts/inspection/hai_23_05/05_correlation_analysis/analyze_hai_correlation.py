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

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
)

TEST1_FILE = PROCESSED_DIR / "hai-test1_aligned.csv"
TEST2_FILE = PROCESSED_DIR / "hai-test2_aligned.csv"


# ============================================================
# SETTINGS
# ============================================================

CORRELATION_THRESHOLD = 0.95


# ============================================================
# LOAD DATA
# ============================================================

def load_features(file_path):

    df = pd.read_csv(file_path)

    feature_columns = [
        col
        for col in df.columns
        if col not in ["timestamp", "label"]
    ]

    return df[feature_columns]


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def analyze_correlation(df, dataset_name):

    print("\n" + "=" * 70)
    print(f"{dataset_name} CORRELATION ANALYSIS")
    print("=" * 70)

    correlation_matrix = df.corr()

    # --------------------------------------------------------
    # Find highly correlated pairs
    # --------------------------------------------------------

    pairs = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            feature_a = columns[i]
            feature_b = columns[j]

            correlation = correlation_matrix.loc[
                feature_a,
                feature_b
            ]

            if abs(correlation) >= CORRELATION_THRESHOLD:

                pairs.append({
                    "feature_1": feature_a,
                    "feature_2": feature_b,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation)
                })

    pair_df = pd.DataFrame(pairs)

    if not pair_df.empty:

        pair_df = pair_df.sort_values(
            "absolute_correlation",
            ascending=False
        )

    print(
        f"\nHighly correlated pairs "
        f"(|r| >= {CORRELATION_THRESHOLD}): "
        f"{len(pair_df)}"
    )

    if not pair_df.empty:

        print("\nTop highly correlated pairs:")

        print(
            pair_df.head(30).to_string(index=False)
        )

    else:

        print("No highly correlated pairs found.")

    # --------------------------------------------------------
    # Correlation with label
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CORRELATION WITH HAI LABEL")
    print("-" * 70)

    label_series = (
        pd.read_csv(
            TEST1_FILE
            if dataset_name == "TEST 1"
            else TEST2_FILE
        )["label"]
    )

    label_correlation = {}

    for feature in df.columns:

        label_correlation[feature] = (
            df[feature].corr(label_series)
        )

    label_corr_df = pd.DataFrame({
        "feature": list(label_correlation.keys()),
        "label_correlation": list(label_correlation.values())
    })

    label_corr_df[
        "absolute_label_correlation"
    ] = label_corr_df[
        "label_correlation"
    ].abs()

    label_corr_df = label_corr_df.sort_values(
        "absolute_label_correlation",
        ascending=False
    )

    print("\nTop features by absolute label correlation:")

    print(
        label_corr_df.head(20).to_string(index=False)
    )

    return pair_df, label_corr_df, correlation_matrix


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HAI 23.05 CORRELATION AND REDUNDANCY ANALYSIS")
    print("=" * 70)

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    test1_features = load_features(TEST1_FILE)
    test2_features = load_features(TEST2_FILE)

    print(
        f"\nTest 1 feature matrix: "
        f"{test1_features.shape}"
    )

    print(
        f"Test 2 feature matrix: "
        f"{test2_features.shape}"
    )

    # --------------------------------------------------------
    # Test 1
    # --------------------------------------------------------

    test1_pairs, test1_label_corr, test1_matrix = (
        analyze_correlation(
            test1_features,
            "TEST 1"
        )
    )

    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    test2_pairs, test2_label_corr, test2_matrix = (
        analyze_correlation(
            test2_features,
            "TEST 2"
        )
    )

    # --------------------------------------------------------
    # Save correlation matrices
    # --------------------------------------------------------

    test1_matrix.to_csv(
        FEATURE_DIR
        / "hai_test1_correlation_matrix.csv"
    )

    test2_matrix.to_csv(
        FEATURE_DIR
        / "hai_test2_correlation_matrix.csv"
    )

    # --------------------------------------------------------
    # Save highly correlated pairs
    # --------------------------------------------------------

    test1_pairs.to_csv(
        FEATURE_DIR
        / "hai_test1_high_correlation_pairs.csv",
        index=False
    )

    test2_pairs.to_csv(
        FEATURE_DIR
        / "hai_test2_high_correlation_pairs.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save label correlations
    # --------------------------------------------------------

    test1_label_corr.to_csv(
        FEATURE_DIR
        / "hai_test1_label_correlations.csv",
        index=False
    )

    test2_label_corr.to_csv(
        FEATURE_DIR
        / "hai_test2_label_correlations.csv",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        FEATURE_DIR
        / "hai_test1_correlation_matrix.csv"
    )

    print(
        FEATURE_DIR
        / "hai_test2_correlation_matrix.csv"
    )

    print(
        FEATURE_DIR
        / "hai_test1_high_correlation_pairs.csv"
    )

    print(
        FEATURE_DIR
        / "hai_test2_high_correlation_pairs.csv"
    )

    print(
        FEATURE_DIR
        / "hai_test1_label_correlations.csv"
    )

    print(
        FEATURE_DIR
        / "hai_test2_label_correlations.csv"
    )

    print("\n" + "=" * 70)
    print("HAI 23.05 CORRELATION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()