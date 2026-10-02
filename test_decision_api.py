import requests
import pandas as pd

from scripts.inference.hai_attack_classifier_runtime import (
    HAIHistoricalFeatureBuilder,
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

API_URL = "http://127.0.0.1:8000/decision"

DATASET = (
    "data/features/hai/hai-23.05/"
    "fault_classification/"
    "hai_test2_attack_classifier_dataset.csv"
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATASET)

builder = HAIHistoricalFeatureBuilder()

original_features = builder.original_features

print("=" * 70)
print("FASTAPI DECISION ENDPOINT TEST")
print("=" * 70)

print(f"Dataset rows       : {len(df)}")
print(f"Original features  : {len(original_features)}")

# ------------------------------------------------------------
# SEND FIRST FIVE REAL OBSERVATIONS
# ------------------------------------------------------------

for i in range(5):

    row = df.iloc[i]

    features = {
        column: float(row[column])
        for column in original_features
    }

    response = requests.post(
        API_URL,
        json={
            "features": features,
            "persistent_anomaly_count": 0,
        },
        timeout=30,
    )

    response.raise_for_status()

    result = response.json()

    decision = result["decision"]

    print()
    print("-" * 70)
    print(f"Observation {i + 1}")
    print("-" * 70)

    print(
        f"Decision Level          : "
        f"{decision['decision_level']}"
    )

    print(
        f"Action                  : "
        f"{decision['action']}"
    )

    print(
        f"Anomaly State           : "
        f"{decision['anomaly_state']}"
    )

    print(
        f"Anomaly Score           : "
        f"{decision['anomaly_score']}"
    )

    print(
        f"Predicted Mechanism     : "
        f"{decision['predicted_attack_mechanism']}"
    )

    print(
        f"Classifier Score        : "
        f"{decision['classifier_score']}"
    )

    print(
        f"Reason                  : "
        f"{decision['reason']}"
    )

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
