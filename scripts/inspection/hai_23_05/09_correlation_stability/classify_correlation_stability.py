from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
)

PAIRS_FILE = (
    FEATURE_DIR
    / "hai_2305_keep_high_correlation_pairs.csv"
)

OUTPUT_FILE = (
    FEATURE_DIR
    / "hai_2305_correlation_stability.csv"
)


# ============================================================
# THRESHOLDS
# ============================================================

HIGH_THRESHOLD = 0.95
MODERATE_THRESHOLD = 0.80


# ============================================================
# LOAD HIGH-CORRELATION PAIRS
# ============================================================

df = pd.read_csv(PAIRS_FILE)

print("=" * 70)
print("HAI 23.05 CORRELATION STABILITY CLASSIFICATION")
print("=" * 70)

print(f"\nPairs analysed: {len(df)}")


# ============================================================
# CLASSIFICATION FUNCTION
# ============================================================

def classify_stability(row):

    r1 = row["test1_correlation"]
    r2 = row["test2_correlation"]

    a1 = abs(r1)
    a2 = abs(r2)

    # --------------------------------------------------------
    # HIGH IN BOTH TESTS
    # --------------------------------------------------------

    if a1 >= HIGH_THRESHOLD and a2 >= HIGH_THRESHOLD:

        if r1 * r2 < 0:
            return (
                "HIGH_BUT_SIGN_CHANGED",
                "Strong correlation magnitude in both tests, "
                "but correlation direction changed."
            )

        return (
            "STABLE_HIGH_CORRELATION",
            "Strong correlation in both Test 1 and Test 2."
        )

    # --------------------------------------------------------
    # HIGH IN TEST 1 ONLY
    # --------------------------------------------------------

    if a1 >= HIGH_THRESHOLD and a2 < HIGH_THRESHOLD:

        return (
            "TEST1_SPECIFIC",
            "Strong correlation in Test 1 but not Test 2."
        )

    # --------------------------------------------------------
    # HIGH IN TEST 2 ONLY
    # --------------------------------------------------------

    if a2 >= HIGH_THRESHOLD and a1 < HIGH_THRESHOLD:

        return (
            "TEST2_SPECIFIC",
            "Strong correlation in Test 2 but not Test 1."
        )

    # --------------------------------------------------------
    # MODERATE IN BOTH
    # --------------------------------------------------------

    if a1 >= MODERATE_THRESHOLD and a2 >= MODERATE_THRESHOLD:

        return (
            "MODERATE_STABLE",
            "Moderate-to-high correlation in both tests, "
            "but below the high-correlation threshold."
        )

    # --------------------------------------------------------
    # OTHERWISE
    # --------------------------------------------------------

    return (
        "UNSTABLE",
        "Correlation does not remain consistently strong "
        "across both tests."
    )


# ============================================================
# APPLY CLASSIFICATION
# ============================================================

results = df.apply(
    classify_stability,
    axis=1
)

df["stability_class"] = [
    result[0]
    for result in results
]

df["stability_reason"] = [
    result[1]
    for result in results
]


# ============================================================
# ADD CORRELATION DIFFERENCE
# ============================================================

df["absolute_correlation_difference"] = (
    df["test1_correlation"].abs()
    -
    df["test2_correlation"].abs()
).abs()


# ============================================================
# ADD SIGN CONSISTENCY
# ============================================================

df["same_correlation_direction"] = (
    (
        df["test1_correlation"] >= 0
    )
    ==
    (
        df["test2_correlation"] >= 0
    )
)


# ============================================================
# REORDER COLUMNS
# ============================================================

columns = [
    "feature_1",
    "feature_2",
    "test1_correlation",
    "test2_correlation",
    "mean_absolute_correlation",
    "absolute_correlation_difference",
    "same_correlation_direction",
    "stability_class",
    "stability_reason"
]

df = df[columns]


# ============================================================
# SORT
# ============================================================

class_order = {
    "STABLE_HIGH_CORRELATION": 1,
    "HIGH_BUT_SIGN_CHANGED": 2,
    "TEST1_SPECIFIC": 3,
    "TEST2_SPECIFIC": 4,
    "MODERATE_STABLE": 5,
    "UNSTABLE": 6
}

df["_order"] = df["stability_class"].map(class_order)

df = df.sort_values(
    ["_order", "mean_absolute_correlation"],
    ascending=[True, False]
)

df = df.drop(columns=["_order"])


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STABILITY SUMMARY")
print("=" * 70)

print(
    df["stability_class"]
    .value_counts()
    .to_string()
)


# ============================================================
# STABLE HIGH CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("STABLE HIGH CORRELATIONS")
print("=" * 70)

stable = df[
    df["stability_class"]
    == "STABLE_HIGH_CORRELATION"
]

if len(stable) > 0:

    print(
        stable[
            [
                "feature_1",
                "feature_2",
                "test1_correlation",
                "test2_correlation"
            ]
        ].to_string(index=False)
    )

else:

    print("None")


# ============================================================
# TEST-SPECIFIC CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("TEST-SPECIFIC CORRELATIONS")
print("=" * 70)

test_specific = df[
    df["stability_class"].isin(
        [
            "TEST1_SPECIFIC",
            "TEST2_SPECIFIC"
        ]
    )
]

if len(test_specific) > 0:

    print(
        test_specific[
            [
                "feature_1",
                "feature_2",
                "test1_correlation",
                "test2_correlation",
                "stability_class"
            ]
        ].to_string(index=False)
    )

else:

    print("None")


# ============================================================
# SIGN CHANGES
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION SIGN CHANGES")
print("=" * 70)

sign_changes = df[
    df["same_correlation_direction"] == False
]

if len(sign_changes) > 0:

    print(
        sign_changes[
            [
                "feature_1",
                "feature_2",
                "test1_correlation",
                "test2_correlation",
                "stability_class"
            ]
        ].to_string(index=False)
    )

else:

    print("None")


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("HAI 23.05 CORRELATION STABILITY CLASSIFICATION COMPLETE")
print("=" * 70)