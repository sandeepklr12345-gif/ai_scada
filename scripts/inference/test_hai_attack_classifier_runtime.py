import pandas as pd
import numpy as np

from scripts.inference.hai_attack_classifier_runtime import (
    HAIHistoricalFeatureBuilder,
)


RAW_PATH = (
    "data/features/hai/hai-23.05/"
    "fault_classification/"
    "hai_test2_multilabel_classifier_dataset.csv"
)

TEMPORAL_PATH = (
    "data/features/hai/hai-23.05/"
    "fault_classification/"
    "hai_test2_attack_classifier_temporal_dataset.csv"
)


raw = pd.read_csv(RAW_PATH)
temporal = pd.read_csv(TEMPORAL_PATH)

raw["timestamp"] = pd.to_datetime(raw["timestamp"])
temporal["timestamp"] = pd.to_datetime(temporal["timestamp"])

builder = HAIHistoricalFeatureBuilder()

runtime_result = None

for _, row in raw.iloc[:5].iterrows():

    features = {
        column: float(row[column])
        for column in builder.original_features
    }

    runtime_result = builder.update(features)


if runtime_result is None:
    raise RuntimeError(
        "Runtime builder did not produce features "
        "after five observations."
    )


batch_row = temporal.iloc[0]

differences = []

for column in builder.feature_columns:

    runtime_value = float(
        runtime_result.iloc[0][column]
    )

    batch_value = float(
        batch_row[column]
    )

    differences.append(
        abs(runtime_value - batch_value)
    )


max_difference = max(differences)
mean_difference = float(
    np.mean(differences)
)

mismatch_count = sum(
    difference > 1e-10
    for difference in differences
)


print("=" * 70)
print("HAI RUNTIME <-> BATCH FEATURE EQUIVALENCE TEST")
print("=" * 70)

print(
    f"Runtime features  : "
    f"{len(builder.feature_columns)}"
)

print(
    f"Batch features    : "
    f"{len(builder.feature_columns)}"
)

print(
    f"Maximum difference: "
    f"{max_difference:.15g}"
)

print(
    f"Mean difference   : "
    f"{mean_difference:.15g}"
)

print(
    f"Mismatches        : "
    f"{mismatch_count}"
)


if mismatch_count != 0:
    raise AssertionError(
        "Runtime and batch temporal features "
        "do not match."
    )


print()
print(
    "PASS: Runtime and batch 408-feature "
    "representations match."
)
