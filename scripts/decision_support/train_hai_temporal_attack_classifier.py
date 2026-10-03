from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test2_attack_classifier_temporal_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "hai_2305_attack_temporal_logistic_regression.joblib"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "hai_2305_attack_temporal_classifier_manifest.json"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "hai_2305_attack_temporal_classifier_metrics.csv"
)

THRESHOLD = 0.50


TRAIN_SCENARIOS = [
    "A202", "A204", "A205", "A206", "A207",
    "A208", "A210", "A211", "A212", "A215",
    "A216", "A217", "A218", "A219", "A220",
    "A221", "A222", "A223", "A224", "A225",
    "A226", "A227", "A228", "A229", "A230",
    "A231", "A232", "A233", "A234", "A237",
    "A238",
]

EVAL_SCENARIOS = [
    "A201",
    "A203",
    "A209",
    "A213",
    "A214",
    "A235",
    "A236",
]


METADATA_COLUMNS = {
    "timestamp",
    "attack_label",
    "scenario_id",
    "attack_codes",
}


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 100)
    print("HAI 23.05 TEMPORAL MULTI-LABEL ATTACK CLASSIFIER")
    print("=" * 100)

    # -----------------------------------------------------
    # 1. Load temporal dataset
    # -----------------------------------------------------

    print("\n[1] Loading temporal classifier dataset")

    data = pd.read_csv(
        INPUT_PATH
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print(
        "Dataset shape:",
        data.shape
    )

    # -----------------------------------------------------
    # 2. Identify target columns
    # -----------------------------------------------------

    print(
        "\n[2] Identifying features and targets"
    )

    target_columns = [
        column
        for column in data.columns
        if column.startswith("AP")
        or column.startswith("AE")
    ]

    feature_columns = [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
        and column not in target_columns
    ]

    print(
        "Features:",
        len(feature_columns)
    )

    print(
        "Targets:",
        len(target_columns)
    )

    if len(feature_columns) != 408:
        fail(
            f"Expected 408 classifier features, "
            f"found {len(feature_columns)}."
        )

    if len(target_columns) != 39:
        fail(
            f"Expected 39 attack targets, "
            f"found {len(target_columns)}."
        )

    # -----------------------------------------------------
    # 3. Validate feature matrix
    # -----------------------------------------------------

    print(
        "\n[3] Validating temporal feature matrix"
    )

    if data[
        feature_columns
    ].isna().any().any():

        fail(
            "Temporal feature matrix contains missing values."
        )

    if not np.isfinite(
        data[
            feature_columns
        ].to_numpy(dtype=float)
    ).all():

        fail(
            "Temporal feature matrix contains "
            "non-finite values."
        )

    print(
        "PASS: 408 features contain no missing/non-finite values."
    )

    # -----------------------------------------------------
    # 4. Create exact event-aware split
    #
    # Normal data is divided using the time span of the
    # held-out scenarios, matching the approved evaluation
    # methodology.
    # -----------------------------------------------------

    print(
        "\n[4] Creating approved event-aware split"
    )

    eval_attack = data[
        data["scenario_id"].isin(
            EVAL_SCENARIOS
        )
    ]

    if eval_attack.empty:
        fail(
            "Evaluation scenarios not found."
        )

    eval_start = (
        eval_attack["timestamp"].min()
    )

    eval_end = (
        eval_attack["timestamp"].max()
    )

    train = data[
        (
            data["scenario_id"].isin(
                TRAIN_SCENARIOS
            )
        )
        |
        (
            (data["attack_label"] == 0)
            &
            ~(
                (data["timestamp"] >= eval_start)
                &
                (data["timestamp"] <= eval_end)
            )
        )
    ].copy()

    evaluation = data[
        (
            data["scenario_id"].isin(
                EVAL_SCENARIOS
            )
        )
        |
        (
            (data["attack_label"] == 0)
            &
            (data["timestamp"] >= eval_start)
            &
            (data["timestamp"] <= eval_end)
        )
    ].copy()

    print(
        "Training rows:",
        len(train)
    )

    print(
        "Evaluation rows:",
        len(evaluation)
    )

    # -----------------------------------------------------
    # 5. Scenario leakage validation
    # -----------------------------------------------------

    print(
        "\n[5] Scenario leakage validation"
    )

    train_scenarios = set(
        train.loc[
            train["attack_label"] == 1,
            "scenario_id"
        ].dropna()
    )

    eval_scenarios = set(
        evaluation.loc[
            evaluation["attack_label"] == 1,
            "scenario_id"
        ].dropna()
    )

    overlap = (
        train_scenarios
        & eval_scenarios
    )

    print(
        "Training scenarios:",
        sorted(train_scenarios)
    )

    print(
        "Evaluation scenarios:",
        sorted(eval_scenarios)
    )

    print(
        "Overlap:",
        overlap
    )

    if overlap:
        fail(
            "Scenario leakage detected."
        )

    print(
        "PASS: No scenario leakage."
    )

    # -----------------------------------------------------
    # 6. Validate training-label coverage
    # -----------------------------------------------------

    print(
        "\n[6] Training-label coverage"
    )

    missing_labels = [
        code
        for code in target_columns
        if train[code].sum() == 0
    ]

    print(
        "Missing training labels:",
        missing_labels
    )

    if missing_labels:
        fail(
            "One or more attack mechanisms have "
            "zero training examples."
        )

    print(
        "PASS: All 39 attack mechanisms represented."
    )

    # -----------------------------------------------------
    # 7. Combination coverage
    # -----------------------------------------------------

    print(
        "\n[7] Combination attack coverage"
    )

    train_combination_rows = int(
        (
            train[target_columns].sum(axis=1)
            > 1
        ).sum()
    )

    eval_combination_rows = int(
        (
            evaluation[target_columns].sum(axis=1)
            > 1
        ).sum()
    )

    print(
        "Training combination rows:",
        train_combination_rows
    )

    print(
        "Evaluation combination rows:",
        eval_combination_rows
    )

    if train_combination_rows == 0:
        fail(
            "No combination attacks in training."
        )

    # -----------------------------------------------------
    # 8. Prepare matrices
    # -----------------------------------------------------

    print(
        "\n[8] Preparing model matrices"
    )

    X_train = train[
        feature_columns
    ].astype(float)

    y_train = train[
        target_columns
    ].astype(int)

    X_eval = evaluation[
        feature_columns
    ].astype(float)

    y_eval = evaluation[
        target_columns
    ].astype(int)

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_eval:",
        X_eval.shape
    )

    print(
        "y_eval:",
        y_eval.shape
    )

    # -----------------------------------------------------
    # 9. Build temporal classifier
    # -----------------------------------------------------

    print(
        "\n[9] Building One-vs-Rest Logistic Regression"
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # -----------------------------------------------------
    # 10. Train
    # -----------------------------------------------------

    print(
        "\n[10] Training temporal classifier"
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "PASS: Temporal classifier training complete."
    )

    # -----------------------------------------------------
    # 11. Probability prediction
    # -----------------------------------------------------

    print(
        "\n[11] Generating evaluation probabilities"
    )

    probabilities = np.asarray(
        model.predict_proba(
            X_eval
        )
    )

    if probabilities.shape != y_eval.shape:
        fail(
            "Probability matrix shape mismatch."
        )

    print(
        "Probability matrix:",
        probabilities.shape
    )

    # -----------------------------------------------------
    # 12. Binary predictions
    # -----------------------------------------------------

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # -----------------------------------------------------
    # 13. Global metrics
    # -----------------------------------------------------

    print(
        "\n[12] Global multi-label metrics"
    )

    metrics = {
        "micro_precision": precision_score(
            y_eval,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "micro_recall": recall_score(
            y_eval,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "micro_f1": f1_score(
            y_eval,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "macro_precision": precision_score(
            y_eval,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_eval,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_eval,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "hamming_loss": hamming_loss(
            y_eval,
            predictions,
        ),
    }

    for name, value in metrics.items():

        print(
            f"{name}: {value:.6f}"
        )

    # -----------------------------------------------------
    # 14. Per-label metrics
    # -----------------------------------------------------

    print(
        "\n[13] Per-label metrics"
    )

    per_label = []

    for index, code in enumerate(
        target_columns
    ):

        actual = y_eval[
            code
        ].to_numpy()

        predicted = predictions[
            :,
            index
        ]

        score = probabilities[
            :,
            index
        ]

        positive_count = int(
            actual.sum()
        )

        if (
            positive_count > 0
            and positive_count < len(actual)
        ):

            roc_auc = roc_auc_score(
                actual,
                score
            )

            pr_auc = average_precision_score(
                actual,
                score
            )

        else:

            roc_auc = np.nan
            pr_auc = np.nan

        per_label.append(
            {
                "attack_code": code,
                "train_positive": int(
                    y_train[code].sum()
                ),
                "eval_positive": positive_count,
                "precision": precision_score(
                    actual,
                    predicted,
                    zero_division=0,
                ),
                "recall": recall_score(
                    actual,
                    predicted,
                    zero_division=0,
                ),
                "f1": f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                ),
                "roc_auc": roc_auc,
                "average_precision": pr_auc,
            }
        )

    per_label_df = pd.DataFrame(
        per_label
    )

    print(
        per_label_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 15. Probability sanity
    # -----------------------------------------------------

    print(
        "\n[14] Probability sanity check"
    )

    print(
        "Minimum:",
        float(probabilities.min())
    )

    print(
        "Maximum:",
        float(probabilities.max())
    )

    print(
        "Mean:",
        float(probabilities.mean())
    )

    if (
        probabilities.min() < 0
        or probabilities.max() > 1
    ):
        fail(
            "Probability values outside [0,1]."
        )

    print(
        "PASS: All probabilities valid."
    )

    # -----------------------------------------------------
    # 16. Save model
    # -----------------------------------------------------

    print(
        "\n[15] Saving temporal classifier"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    # -----------------------------------------------------
    # 17. Save manifest
    # -----------------------------------------------------

    manifest = {
        "model_name": (
            "HAI 23.05 Temporal "
            "Multi-label Attack Classifier"
        ),
        "model_family": (
            "One-vs-Rest Logistic Regression"
        ),
        "feature_count": len(feature_columns),
        "target_count": len(target_columns),
        "features": feature_columns,
        "targets": target_columns,
        "threshold": THRESHOLD,
        "train_scenarios": TRAIN_SCENARIOS,
        "evaluation_scenarios": EVAL_SCENARIOS,
        "probability_source": (
            "One-vs-Rest classifier predict_proba"
        ),
        "temporal_features": True,
        "candidate_c_modified": False,
        "frozen_pipeline_modified": False,
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    # -----------------------------------------------------
    # 18. Save metrics
    # -----------------------------------------------------

    per_label_df.to_csv(
        METRICS_PATH,
        index=False
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Manifest:",
        MANIFEST_PATH
    )

    print(
        "Metrics:",
        METRICS_PATH
    )

    # -----------------------------------------------------
    # 19. Final comparison
    # -----------------------------------------------------

    print(
        "\n[16] Comparison with raw-feature baseline"
    )

    raw_baseline_f1 = 0.000586

    temporal_f1 = metrics[
        "micro_f1"
    ]

    print(
        f"Raw 68-feature baseline Micro-F1: "
        f"{raw_baseline_f1:.6f}"
    )

    print(
        f"Temporal 408-feature Micro-F1: "
        f"{temporal_f1:.6f}"
    )

    if raw_baseline_f1 > 0:

        improvement = (
            temporal_f1
            / raw_baseline_f1
        )

        print(
            f"Relative F1 ratio: "
            f"{improvement:.4f}x"
        )

    # -----------------------------------------------------
    # 20. Final status
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("STEP 6.15 TEMPORAL CLASSIFIER COMPLETE")
    print("=" * 100)

    print(
        """
The temporal classifier was trained using the approved
event-aware scenario split.

Candidate C was not modified.

The original 68-feature classifier was not modified.

The model produces independent probabilities for the
39 attack mechanisms.

The temporal classifier can now be compared directly
against the raw-feature baseline.
"""
    )


if __name__ == "__main__":
    main()