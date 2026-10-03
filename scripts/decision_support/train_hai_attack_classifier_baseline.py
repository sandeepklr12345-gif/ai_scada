from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test2_multilabel_classifier_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "hai_2305_attack_multilabel_logistic_regression.joblib"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "hai_2305_attack_classifier_manifest.json"
)

THRESHOLD = 0.50


# ---------------------------------------------------------
# Final scenario design from Step 6.11
# ---------------------------------------------------------

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


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 100)
    print("HAI 23.05 MULTI-LABEL ATTACK CLASSIFIER BASELINE")
    print("=" * 100)

    # -----------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------

    print("\n[1] Loading classifier dataset")

    data = pd.read_csv(INPUT_PATH)

    print(
        "Dataset shape:",
        data.shape
    )

    # -----------------------------------------------------
    # 2. Identify columns
    # -----------------------------------------------------

    print(
        "\n[2] Identifying features and targets"
    )

    metadata_columns = {
        "timestamp",
        "attack_label",
        "scenario_id",
        "attack_codes",
    }

    target_columns = [
        column
        for column in data.columns
        if column.startswith("AP")
        or column.startswith("AE")
    ]

    feature_columns = [
        column
        for column in data.columns
        if column not in metadata_columns
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

    if len(feature_columns) != 68:
        fail(
            f"Expected 68 features, "
            f"found {len(feature_columns)}."
        )

    if len(target_columns) != 39:
        fail(
            f"Expected 39 targets, "
            f"found {len(target_columns)}."
        )

    # -----------------------------------------------------
    # 3. Create scenario-level train/evaluation split
    # -----------------------------------------------------

    print(
        "\n[3] Creating event-aware classifier split"
    )

    train = data[
        data["scenario_id"].isin(
            TRAIN_SCENARIOS
        )
        | (
            data["attack_label"] == 0
        )
    ].copy()

    evaluation = data[
        data["scenario_id"].isin(
            EVAL_SCENARIOS
        )
        | (
            data["attack_label"] == 0
        )
    ].copy()

    # -----------------------------------------------------
    # IMPORTANT:
    # Normal samples must also be separated temporally.
    #
    # We therefore assign normal rows according to
    # their timestamp relative to the evaluation scenarios.
    # -----------------------------------------------------

    eval_start = (
        data[
            data["scenario_id"].isin(
                EVAL_SCENARIOS
            )
        ]["timestamp"]
        .min()
    )

    eval_end = (
        data[
            data["scenario_id"].isin(
                EVAL_SCENARIOS
            )]["timestamp"]
        .max()
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
    # 4. Verify scenario separation
    # -----------------------------------------------------

    print(
        "\n[4] Scenario leakage validation"
    )

    train_attack_scenarios = set(
        train.loc[
            train["attack_label"] == 1,
            "scenario_id"
        ].dropna()
    )

    eval_attack_scenarios = set(
        evaluation.loc[
            evaluation["attack_label"] == 1,
            "scenario_id"
        ].dropna()
    )

    overlap = (
        train_attack_scenarios
        & eval_attack_scenarios
    )

    print(
        "Training scenarios:",
        sorted(train_attack_scenarios)
    )

    print(
        "Evaluation scenarios:",
        sorted(eval_attack_scenarios)
    )

    print(
        "Overlap:",
        overlap
    )

    if overlap:
        fail(
            "Scenario leakage detected."
        )

    # -----------------------------------------------------
    # 5. Verify all labels in training
    # -----------------------------------------------------

    print(
        "\n[5] Training-label coverage"
    )

    missing_training_labels = [
        column
        for column in target_columns
        if train[column].sum() == 0
    ]

    print(
        "Missing training labels:",
        missing_training_labels
    )

    if missing_training_labels:
        fail(
            "One or more attack labels have "
            "zero training examples."
        )

    print(
        "PASS: All 39 labels represented."
    )

    # -----------------------------------------------------
    # 6. Prepare matrices
    # -----------------------------------------------------

    print(
        "\n[6] Preparing model matrices"
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
    # 7. Build baseline
    # -----------------------------------------------------

    print(
        "\n[7] Building One-vs-Rest Logistic Regression"
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
                        max_iter=1000,
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
    # 8. Train
    # -----------------------------------------------------

    print(
        "\n[8] Training baseline classifier"
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "PASS: Model training complete."
    )

    # -----------------------------------------------------
    # 9. Predict probabilities
    # -----------------------------------------------------

    print(
        "\n[9] Generating evaluation probabilities"
    )

    probabilities = model.predict_proba(
        X_eval
    )

    probabilities = np.asarray(
        probabilities
    )

    print(
        "Probability matrix:",
        probabilities.shape
    )

    if probabilities.shape != y_eval.shape:
        fail(
            "Probability matrix shape does not "
            "match target matrix."
        )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # -----------------------------------------------------
    # 10. Global metrics
    # -----------------------------------------------------

    print(
        "\n[10] Multi-label evaluation"
    )

    micro_precision = precision_score(
        y_eval,
        predictions,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        y_eval,
        predictions,
        average="micro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_eval,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_eval,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_eval,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_eval,
        predictions,
        average="macro",
        zero_division=0,
    )

    hamming = hamming_loss(
        y_eval,
        predictions,
    )

    print(
        f"Micro Precision : {micro_precision:.6f}"
    )

    print(
        f"Micro Recall    : {micro_recall:.6f}"
    )

    print(
        f"Micro F1        : {micro_f1:.6f}"
    )

    print(
        f"Macro Precision : {macro_precision:.6f}"
    )

    print(
        f"Macro Recall    : {macro_recall:.6f}"
    )

    print(
        f"Macro F1        : {macro_f1:.6f}"
    )

    print(
        f"Hamming Loss    : {hamming:.6f}"
    )

    # -----------------------------------------------------
    # 11. Per-label metrics
    # -----------------------------------------------------

    print(
        "\n[11] Per-label evaluation"
    )

    report = classification_report(
        y_eval,
        predictions,
        target_names=target_columns,
        zero_division=0,
        output_dict=True,
    )

    per_label = []

    for code in target_columns:

        per_label.append(
            {
                "attack_code": code,
                "precision": report[
                    code
                ]["precision"],
                "recall": report[
                    code
                ]["recall"],
                "f1": report[
                    code
                ]["f1-score"],
                "support": report[
                    code
                ]["support"],
                "train_positive": int(
                    y_train[code].sum()
                ),
                "eval_positive": int(
                    y_eval[code].sum()
                ),
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
    # 12. Probability sanity check
    # -----------------------------------------------------

    print(
        "\n[12] Probability sanity check"
    )

    print(
        "Minimum probability:",
        float(probabilities.min())
    )

    print(
        "Maximum probability:",
        float(probabilities.max())
    )

    print(
        "Mean probability:",
        float(probabilities.mean())
    )

    if (
        probabilities.min() < 0
        or probabilities.max() > 1
    ):
        fail(
            "Invalid probability values."
        )

    print(
        "PASS: Probabilities are within [0, 1]."
    )

    # -----------------------------------------------------
    # 13. Save model and manifest
    # -----------------------------------------------------

    print(
        "\n[13] Saving baseline model"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    manifest = {
        "model_name": (
            "HAI 23.05 Multi-label "
            "Attack Classifier"
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

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Manifest:",
        MANIFEST_PATH
    )

    # -----------------------------------------------------
    # 14. Save evaluation report
    # -----------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "hai_2305_attack_classifier_baseline_metrics.csv"
    )

    per_label_df.to_csv(
        metrics_path,
        index=False
    )

    print(
        "Metrics:",
        metrics_path
    )

    # -----------------------------------------------------
    # 15. Final status
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("STEP 6.12 BASELINE CLASSIFIER COMPLETE")
    print("=" * 100)

    print(
        """
The baseline classifier has been trained only on the
approved event-aware scenario set.

Candidate C was not modified.

The classifier produces independent probabilities
for the 39 attack mechanisms.

Those probabilities are classifier probabilities and
must not be confused with the Candidate C anomaly score.
"""
    )


if __name__ == "__main__":
    main()