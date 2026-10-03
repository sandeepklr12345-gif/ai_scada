from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test2_multilabel_classifier_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
    / "hai_2305_attack_multilabel_logistic_regression.joblib"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
    / "hai_2305_attack_classifier_manifest.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
)

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


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
    print("HAI 23.05 ATTACK CLASSIFIER BASELINE DIAGNOSTICS")
    print("=" * 100)

    # -----------------------------------------------------
    # 1. Load model and data
    # -----------------------------------------------------

    print("\n[1] Loading model and dataset")

    data = pd.read_csv(DATA_PATH)

    model = joblib.load(
        MODEL_PATH
    )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(file)

    target_columns = manifest[
        "targets"
    ]

    feature_columns = manifest[
        "features"
    ]

    print(
        "Dataset:",
        data.shape
    )

    print(
        "Features:",
        len(feature_columns)
    )

    print(
        "Targets:",
        len(target_columns)
    )

    # -----------------------------------------------------
    # 2. Recreate exact evaluation split
    # -----------------------------------------------------

    print(
        "\n[2] Recreating approved evaluation split"
    )

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

    X_eval = evaluation[
        feature_columns
    ].astype(float)

    y_eval = evaluation[
        target_columns
    ].astype(int)

    print(
        "Evaluation rows:",
        len(evaluation)
    )

    # -----------------------------------------------------
    # 3. Generate probabilities
    # -----------------------------------------------------

    print(
        "\n[3] Generating classifier probabilities"
    )

    probabilities = np.asarray(
        model.predict_proba(
            X_eval
        )
    )

    print(
        "Probability matrix:",
        probabilities.shape
    )

    if probabilities.shape != y_eval.shape:
        fail(
            "Probability matrix shape mismatch."
        )

    # -----------------------------------------------------
    # 4. Probability distribution
    # -----------------------------------------------------

    print(
        "\n[4] Probability distribution"
    )

    print(
        "Global minimum:",
        probabilities.min()
    )

    print(
        "Global maximum:",
        probabilities.max()
    )

    print(
        "Global mean:",
        probabilities.mean()
    )

    print(
        "Percentiles:"
    )

    percentiles = [
        50,
        75,
        90,
        95,
        99,
        99.5,
        99.9,
    ]

    values = np.percentile(
        probabilities,
        percentiles
    )

    for percentile, value in zip(
        percentiles,
        values
    ):

        print(
            f"  P{percentile:<5}: "
            f"{value:.8f}"
        )

    # -----------------------------------------------------
    # 5. Threshold sensitivity
    # -----------------------------------------------------

    print(
        "\n[5] Threshold sensitivity"
    )

    threshold_results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        macro_f1 = f1_score(
            y_eval,
            predictions,
            average="macro",
            zero_division=0,
        )

        predicted_positive = int(
            predictions.sum()
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "predicted_positive_labels": predicted_positive,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    print(
        threshold_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 6. ROC-AUC and PR-AUC
    # -----------------------------------------------------

    print(
        "\n[6] Per-label probability quality"
    )

    auc_results = []

    for index, code in enumerate(
        target_columns
    ):

        actual = y_eval[
            code
        ].to_numpy()

        score = probabilities[
            :,
            index
        ]

        positive_count = int(
            actual.sum()
        )

        negative_count = int(
            len(actual)
            - positive_count
        )

        if (
            positive_count == 0
            or negative_count == 0
        ):

            roc_auc = np.nan
            pr_auc = np.nan

        else:

            roc_auc = roc_auc_score(
                actual,
                score,
            )

            pr_auc = average_precision_score(
                actual,
                score,
            )

        auc_results.append(
            {
                "attack_code": code,
                "positive_samples": positive_count,
                "negative_samples": negative_count,
                "roc_auc": roc_auc,
                "average_precision": pr_auc,
                "max_probability": float(
                    score.max()
                ),
                "mean_positive_probability": (
                    float(
                        score[actual == 1].mean()
                    )
                    if positive_count > 0
                    else np.nan
                ),
                "mean_negative_probability": (
                    float(
                        score[actual == 0].mean()
                    )
                    if negative_count > 0
                    else np.nan
                ),
            }
        )

    auc_df = pd.DataFrame(
        auc_results
    )

    print(
        auc_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 7. Attack-vs-normal probability analysis
    # -----------------------------------------------------

    print(
        "\n[7] Attack-vs-normal probability analysis"
    )

    attack_mask = (
        evaluation["attack_label"].to_numpy()
        == 1
    )

    normal_mask = (
        evaluation["attack_label"].to_numpy()
        == 0
    )

    attack_probabilities = (
        probabilities[attack_mask]
    )

    normal_probabilities = (
        probabilities[normal_mask]
    )

    print(
        "Attack rows:",
        attack_mask.sum()
    )

    print(
        "Normal rows:",
        normal_mask.sum()
    )

    print(
        "Mean max probability on attack rows:",
        attack_probabilities.max(
            axis=1
        ).mean()
    )

    print(
        "Mean max probability on normal rows:",
        normal_probabilities.max(
            axis=1
        ).mean()
    )

    for threshold in THRESHOLDS:

        attack_hits = (
            attack_probabilities.max(
                axis=1
            ) >= threshold
        ).sum()

        normal_false_hits = (
            normal_probabilities.max(
                axis=1
            ) >= threshold
        ).sum()

        print(
            f"Threshold {threshold:.2f}: "
            f"attack rows detected="
            f"{attack_hits}/"
            f"{len(attack_probabilities)}, "
            f"normal false-positive rows="
            f"{normal_false_hits}/"
            f"{len(normal_probabilities)}"
        )

    # -----------------------------------------------------
    # 8. Correct-label probability analysis
    # -----------------------------------------------------

    print(
        "\n[8] Correct-label probability analysis"
    )

    correct_label_results = []

    for index, code in enumerate(
        target_columns
    ):

        positive_mask = (
            y_eval[code].to_numpy()
            == 1
        )

        if not positive_mask.any():

            continue

        positive_scores = probabilities[
            positive_mask,
            index,
        ]

        negative_scores = probabilities[
            ~positive_mask,
            index,
        ]

        correct_label_results.append(
            {
                "attack_code": code,
                "positive_rows": int(
                    positive_mask.sum()
                ),
                "positive_mean_probability": float(
                    positive_scores.mean()
                ),
                "positive_median_probability": float(
                    np.median(
                        positive_scores
                    )
                ),
                "positive_max_probability": float(
                    positive_scores.max()
                ),
                "negative_mean_probability": float(
                    negative_scores.mean()
                ),
            }
        )

    correct_label_df = pd.DataFrame(
        correct_label_results
    )

    print(
        correct_label_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 9. Save diagnostics
    # -----------------------------------------------------

    print(
        "\n[9] Saving diagnostic results"
    )

    threshold_path = (
        OUTPUT_DIR
        / "hai_2305_baseline_threshold_analysis.csv"
    )

    auc_path = (
        OUTPUT_DIR
        / "hai_2305_baseline_probability_quality.csv"
    )

    correct_label_path = (
        OUTPUT_DIR
        / "hai_2305_baseline_correct_label_probabilities.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False
    )

    auc_df.to_csv(
        auc_path,
        index=False
    )

    correct_label_df.to_csv(
        correct_label_path,
        index=False
    )

    print(
        "Threshold analysis:",
        threshold_path
    )

    print(
        "Probability quality:",
        auc_path
    )

    print(
        "Correct-label probabilities:",
        correct_label_path
    )

    # -----------------------------------------------------
    # 10. Final interpretation
    # -----------------------------------------------------

    print(
        "\n[10] Diagnostic interpretation"
    )

    best_f1_row = threshold_df.loc[
        threshold_df["micro_f1"].idxmax()
    ]

    print(
        "Best tested threshold by micro-F1:",
        best_f1_row["threshold"]
    )

    print(
        "Corresponding micro-F1:",
        best_f1_row["micro_f1"]
    )

    print(
        """
Interpretation rules:

1. If lowering the threshold substantially improves
   F1, the baseline may contain useful probability
   signal but the 0.50 threshold is too conservative.

2. If ROC-AUC / average precision are poor for the
   evaluated labels, the raw 68-feature representation
   may not separate the attack mechanisms adequately.

3. If positive-label probabilities remain low even on
   true attack rows, threshold tuning alone will not
   solve the problem.

4. This diagnostic does not modify the model.

5. This diagnostic does not modify Candidate C.
"""
    )

    print("=" * 100)
    print("STEP 6.13 BASELINE DIAGNOSTICS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()