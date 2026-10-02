import pandas as pd

from scripts.inference.hai_candidate_C_inference import (
    HAICandidateCInference,
)

from scripts.inference.hai_attack_classifier_runtime import (
    HAIHistoricalFeatureBuilder,
)

from scripts.inference.hai_attack_classifier_inference import (
    HAIAttackClassifierInference,
)

from scripts.decision_support.decision_engine import (
    DecisionInput,
    DecisionSupportEngine,
)


RAW_PATH = (
    "data/features/hai/hai-23.05/"
    "fault_classification/"
    "hai_test2_multilabel_classifier_dataset.csv"
)

HAI_FINAL_DIR = (
    "data/features/hai/hai-23.05/"
    "temporal_representation/"
    "final_candidate"
)

HAI_MODEL_PATH = (
    f"{HAI_FINAL_DIR}/"
    "hai_2305_candidate_C_isolation_forest.joblib"
)

HAI_MANIFEST_PATH = (
    f"{HAI_FINAL_DIR}/"
    "hai_2305_candidate_C_feature_manifest.csv"
)


raw = pd.read_csv(RAW_PATH)

runtime_builder = HAIHistoricalFeatureBuilder()
classifier = HAIAttackClassifierInference()

anomaly_engine = HAICandidateCInference(
    HAI_MODEL_PATH,
    HAI_MANIFEST_PATH,
)

decision_engine = DecisionSupportEngine()


# ---------------------------------------------------------
# Build five-observation history
# ---------------------------------------------------------

classifier_result = None
anomaly_result = None

for _, row in raw.iloc[:5].iterrows():

    features = {
        column: float(row[column])
        for column in runtime_builder.original_features
    }

    temporal_features = runtime_builder.update(
        features
    )

    # Candidate C requires its own feature contract.
    anomaly_input = pd.DataFrame([features])

    anomaly_output = anomaly_engine.predict_stream(
        anomaly_input
    )

    anomaly_result = anomaly_output.iloc[-1]

    if temporal_features is not None:

        classifier_result = classifier.predict(
            temporal_features
        )


if classifier_result is None:
    raise RuntimeError(
        "Attack classifier did not produce a result."
    )

if anomaly_result is None:
    raise RuntimeError(
        "Candidate C did not produce a result."
    )


# ---------------------------------------------------------
# Combine outputs
# ---------------------------------------------------------

decision_input = DecisionInput(
    anomaly_score=(
        None
        if pd.isna(anomaly_result["anomaly_score"])
        else float(anomaly_result["anomaly_score"])
    ),
    anomaly_state=str(
        anomaly_result["status"]
    ),
    persistent_anomaly_count=0,
    predicted_attack_mechanism=(
        classifier_result[
            "predicted_attack_mechanism"
        ]
    ),
    classifier_score=(
        classifier_result[
            "classifier_score"
        ]
    ),
)

decision = decision_engine.evaluate(
    decision_input
)


print("=" * 70)
print("HAI DECISION RUNTIME INTEGRATION TEST")
print("=" * 70)

print(
    f"Anomaly state       : "
    f"{anomaly_result['status']}"
)

print(
    f"Anomaly score       : "
    f"{anomaly_result['anomaly_score']}"
)

print(
    f"Predicted mechanism : "
    f"{decision.predicted_attack_mechanism}"
)

print(
    f"Classifier score    : "
    f"{decision.classifier_score}"
)

print(
    f"Decision level      : "
    f"{decision.decision_level}"
)

print(
    f"Action              : "
    f"{decision.action}"
)

print(
    f"Reason              : "
    f"{decision.reason}"
)


if decision.decision_level not in {
    "LEVEL_1",
    "LEVEL_2",
    "LEVEL_3",
}:
    raise AssertionError(
        "Invalid decision level."
    )


print()
print(
    "PASS: Candidate C + attack classifier "
    "+ Decision Support Engine integrated."
)
