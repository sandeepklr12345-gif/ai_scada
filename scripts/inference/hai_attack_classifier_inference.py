from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
    / "gpu"
    / "hai_2305_temporal_multilabel_gpu.joblib"
)


class HAIAttackClassifierInference:

    def __init__(
        self,
        model_path: Optional[str] = None,
    ):

        path = (
            Path(model_path)
            if model_path
            else MODEL_PATH
        )

        artifact = joblib.load(path)

        self.scaler = artifact["scaler"]
        self.models = artifact["models"]
        self.feature_columns = list(
            artifact["feature_columns"]
        )
        self.target_columns = list(
            artifact["target_columns"]
        )

        if len(self.feature_columns) != 408:
            raise ValueError(
                f"Expected 408 features, found "
                f"{len(self.feature_columns)}."
            )

        if len(self.models) != 39:
            raise ValueError(
                f"Expected 39 classifiers, found "
                f"{len(self.models)}."
            )

        if set(self.models.keys()) != set(
            self.target_columns
        ):
            raise ValueError(
                "Classifier mechanisms do not match "
                "the artifact target columns."
            )

    def predict(
        self,
        features: pd.DataFrame,
    ) -> dict:

        missing = [
            column
            for column in self.feature_columns
            if column not in features.columns
        ]

        if missing:
            raise ValueError(
                "Missing classifier features: "
                + ", ".join(missing)
            )

        X = features[
            self.feature_columns
        ].astype(np.float32)

        if not np.isfinite(
            X.to_numpy()
        ).all():
            raise ValueError(
                "Classifier input contains "
                "NaN or infinite values."
            )

        X_scaled = self.scaler.transform(X)

        scores = {}

        for mechanism, model in self.models.items():

            probabilities = model.predict_proba(
                X_scaled
            )

            probabilities = np.asarray(
                probabilities
            )

            if probabilities.ndim == 2:
                score = float(
                    probabilities[0, -1]
                )
            else:
                score = float(
                    probabilities[0]
                )

            scores[mechanism] = score

        predicted_mechanism = max(
            scores,
            key=scores.get,
        )

        classifier_score = scores[
            predicted_mechanism
        ]

        return {
            "predicted_attack_mechanism":
                predicted_mechanism,
            "classifier_score":
                classifier_score,
            "scores":
                scores,
        }


if __name__ == "__main__":

    engine = HAIAttackClassifierInference()

    print("=" * 70)
    print("HAI ATTACK CLASSIFIER INFERENCE")
    print("=" * 70)

    print(
        f"Features     : "
        f"{len(engine.feature_columns)}"
    )

    print(
        f"Mechanisms   : "
        f"{len(engine.target_columns)}"
    )

    print(
        f"Scaler       : "
        f"{type(engine.scaler).__name__}"
    )

    print(
        f"Classifiers  : "
        f"{len(engine.models)}"
    )

    print()
    print(
        "Classifier inference initialized successfully."
    )

    print()
    print("SELF TEST: PASS")
