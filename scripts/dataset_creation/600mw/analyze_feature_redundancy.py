import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/600mw/600mw_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/features/600mw"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "600mw_correlation_matrix.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("600 MW FEATURE REDUNDANCY ANALYSIS")
print("=" * 80)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 2. Select numerical features
# --------------------------------------------------

numeric_df = df.select_dtypes(
    include="number"
)

print("\nNumeric features:", len(numeric_df.columns))


# --------------------------------------------------
# 3. Calculate correlation matrix
# --------------------------------------------------

correlation_matrix = numeric_df.corr(
    method="pearson"
)


# --------------------------------------------------
# 4. Save correlation matrix
# --------------------------------------------------

correlation_matrix.to_csv(
    OUTPUT_FILE
)

print("\nCorrelation matrix saved:")
print(OUTPUT_FILE)


# --------------------------------------------------
# 5. Find highly correlated feature pairs
# --------------------------------------------------

threshold = 0.95

pairs = []

columns = correlation_matrix.columns

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= threshold:

            pairs.append({
                "feature_1": columns[i],
                "feature_2": columns[j],
                "correlation": correlation
            })


high_corr = pd.DataFrame(pairs)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\n" + "-" * 80)
print("HIGHLY CORRELATED FEATURE PAIRS")
print("-" * 80)

print(
    f"Threshold: |correlation| >= {threshold}"
)

print(
    "Number of highly correlated pairs:",
    len(high_corr)
)


if len(high_corr) > 0:

    high_corr = high_corr.sort_values(
        by="correlation",
        key=lambda x: x.abs(),
        ascending=False
    )

    print(
        "\nTop highly correlated pairs:"
    )

    print(
        high_corr.head(30).to_string(
            index=False
        )
    )

else:

    print(
        "\nNo feature pairs crossed the threshold."
    )


# --------------------------------------------------
# 7. Correlation with current power output
# --------------------------------------------------

target_column = "Power output\n（MW）"

if target_column in correlation_matrix.columns:

    power_correlation = (
        correlation_matrix[target_column]
        .drop(target_column)
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    print("\n" + "-" * 80)
    print("CORRELATION WITH POWER OUTPUT")
    print("-" * 80)

    print(
        power_correlation.to_string()
    )

else:

    print(
        "\nPower output column not found."
    )


print("\n" + "=" * 80)
print("REDUNDANCY ANALYSIS COMPLETE")
print("=" * 80)