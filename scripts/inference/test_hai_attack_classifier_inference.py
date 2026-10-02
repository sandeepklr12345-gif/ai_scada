import pandas as pd

from scripts.inference.hai_attack_classifier_runtime import (
    HAIHistoricalFeatureBuilder,
)

from scripts.inference.hai_attack_classifier_inference import (
    HAIAttackClassifierInference,
)


RAW_PATH = (
    "data/features/hai/hai-23.05/"
    "fault_classification/"
    "hai_test2_multilabel_classifier_dataset.csv"
)


raw = pd.read_csv(RAW_PATH)

builder = HAIHistoricalFeatureBuilder()
classifier = HAIAttackClassifierInference()

result = None

for _, row in raw.iloc[:5].iterrows():

    features = {
        column: float(row[column])
        for column in builder.original_features
    }

    temporal_features = builder.update(
        features
    )

    if temporal_features is not None:

        result = classifier.predict(
            temporal_features
        )


if result is None:
    raise RuntimeError(
        "Classifier features were not produced."
    )


scores = result["scores"]

top_predictions = sorted(
    scores.items(),
    key=lambda item: item[1],
    reverse=True,
)[:5]


print("=" * 70)
print(
    "HAI GPU ATTACK CLASSIFIER RUNTIME INFERENCE TEST"
)
print("=" * 70)

print(
    f"Predicted mechanism : "
    f"{result['predicted_attack_mechanism']}"
)

print(
    f"Classifier score    : "
    f"{result['classifier_score']:.8f}"
)

print("\nTop 5 mechanisms")
print("-" * 70)

for rank, (mechanism, score) in enumerate(
    top_predictions,
    start=1,
):
    print(
        f"{rank}. "
        f"{mechanism:<6} "
        f"{score:.8f}"
    )


print("\nTotal scores:", len(scores))

if len(scores) != 39:
    raise AssertionError(
        f"Expected 39 scores, found {len(scores)}."
    )

print()
print(
    "PASS: Runtime feature builder → "
    "GPU classifier inference completed."
)
