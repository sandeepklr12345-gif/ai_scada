from collections import deque
from pathlib import Path
from typing import Dict, Optional

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


class HAIHistoricalFeatureBuilder:
    """
    Runtime feature builder for the HAI 23.05 attack classifier.

    Reproduces the validated Step 6.14 temporal representation:

        68 original features
        68 first-order differences
        68 rolling means
        68 rolling standard deviations
        68 deviations from rolling mean
        68 second-order differences

    Total: 408 classifier features.

    The builder keeps the most recent five observations because
    the training pipeline uses a five-observation temporal window.
    """

    WINDOW_SIZE = 5

    TEMPORAL_SUFFIXES = (
        "__diff_1s",
        "__rolling_mean_5s",
        "__rolling_std_5s",
        "__deviation_5s",
        "__diff2_1s",
    )

    def __init__(self, model_path: Optional[str] = None):

        path = Path(model_path) if model_path else MODEL_PATH

        artifact = joblib.load(path)

        self.feature_columns = list(
            artifact["feature_columns"]
        )

        self.target_columns = list(
            artifact["target_columns"]
        )

        if len(self.feature_columns) != 408:
            raise ValueError(
                f"Expected 408 classifier features, "
                f"found {len(self.feature_columns)}."
            )

        if len(self.target_columns) != 39:
            raise ValueError(
                f"Expected 39 attack mechanisms, "
                f"found {len(self.target_columns)}."
            )

        self.original_features = [
            column
            for column in self.feature_columns
            if not any(
                column.endswith(suffix)
                for suffix in self.TEMPORAL_SUFFIXES
            )
        ]

        if len(self.original_features) != 68:
            raise ValueError(
                f"Expected 68 original features, "
                f"found {len(self.original_features)}."
            )

        self.history = deque(
            maxlen=self.WINDOW_SIZE
        )

    def reset(self):
        """Clear the streaming history."""
        self.history.clear()

    def update(
        self,
        features: Dict[str, float],
    ) -> Optional[pd.DataFrame]:
        """
        Add one SCADA observation.

        Returns None until five observations are available.
        After that, returns one row containing exactly the
        408 classifier features in artifact order.
        """

        missing = [
            column
            for column in self.original_features
            if column not in features
        ]

        if missing:
            raise ValueError(
                "Missing required HAI classifier features: "
                + ", ".join(missing)
            )

        current = {
            column: float(features[column])
            for column in self.original_features
        }

        values = np.asarray(
            [
                [
                    observation[column]
                    for column in self.original_features
                ]
                for observation in self.history
            ]
            + [
                [
                    current[column]
                    for column in self.original_features
                ]
            ],
            dtype=np.float64,
        )

        self.history.append(current)

        if len(self.history) < self.WINDOW_SIZE:
            return None

        # --------------------------------------------------
        # Exact Step 6.14 transformations
        # --------------------------------------------------

        current_values = values[-1]
        previous_values = values[-2]

        diff_1s = (
            current_values
            - previous_values
        )

        rolling_window = values[-self.WINDOW_SIZE:]

        rolling_mean = (
            rolling_window.mean(axis=0)
        )

        # pandas rolling().std() uses sample standard
        # deviation by default, ddof=1.
        rolling_std = (
            rolling_window.std(
                axis=0,
                ddof=1,
            )
        )

        deviation = (
            current_values
            - rolling_mean
        )

        previous_diff = (
            values[-2]
            - values[-3]
        )

        diff2_1s = (
            diff_1s
            - previous_diff
        )

        generated = {}

        for index, column in enumerate(
            self.original_features
        ):
            generated[
                f"{column}__diff_1s"
            ] = diff_1s[index]

            generated[
                f"{column}__rolling_mean_5s"
            ] = rolling_mean[index]

            generated[
                f"{column}__rolling_std_5s"
            ] = rolling_std[index]

            generated[
                f"{column}__deviation_5s"
            ] = deviation[index]

            generated[
                f"{column}__diff2_1s"
            ] = diff2_1s[index]

        row = {}

        for column in self.feature_columns:

            if column in self.original_features:
                row[column] = current[
                    column
                ]
            else:
                row[column] = generated[
                    column
                ]

        result = pd.DataFrame(
            [[
                row[column]
                for column in self.feature_columns
            ]],
            columns=self.feature_columns,
        )

        if not np.isfinite(
            result.to_numpy(
                dtype=np.float64
            )
        ).all():
            raise ValueError(
                "Runtime classifier features contain "
                "NaN or infinite values."
            )

        return result


if __name__ == "__main__":

    builder = HAIHistoricalFeatureBuilder()

    print("=" * 70)
    print(
        "HAI ATTACK CLASSIFIER RUNTIME FEATURE BUILDER"
    )
    print("=" * 70)

    print(
        f"Original features : "
        f"{len(builder.original_features)}"
    )
    print(
        f"Classifier features: "
        f"{len(builder.feature_columns)}"
    )
    print(
        f"Attack mechanisms : "
        f"{len(builder.target_columns)}"
    )

    print(
        "\nRuntime feature builder initialized successfully."
    )

    print(
        "History requirement: "
        f"{builder.WINDOW_SIZE} observations."
    )

    print(
        "\nSELF TEST: PASS"
    )