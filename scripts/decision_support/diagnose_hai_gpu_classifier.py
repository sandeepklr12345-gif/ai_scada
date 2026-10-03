from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
    / "gpu"
    / "hai_2305_temporal_multilabel_gpu.joblib"
)

EVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
    / "temporal_gpu"
    / "evaluation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
    / "gpu"
)

THRESHOLD_ANALYSIS_PATH = (
    OUTPUT_DIR
    / "hai_2305_gpu_threshold_analysis.csv"
)

LABEL_ANALYSIS_PATH = (
    OUTPUT_DIR
    / "hai_2305_gpu_probability_quality.csv"
)


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 100)
    print("HAI 23.05 GPU CLASSIFIER PROBABILITY DIAGNOSTICS")
    print("=" * 100)

    # =========================================================
    # 1. Load model
    # =========================================================

    print("\n[1] Loading GPU classifier")

    artifact = joblib.load(
        MODEL_PATH
    )

    scaler = artifact["scaler"]
    models = artifact["models"]
    feature_columns = artifact["feature_columns"]
    target_columns = artifact["target_columns"]

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
            f"Expected 408 features, found "
            f"{len(feature_columns)}"
        )

    if len(target_columns) != 39:
        fail(
            f"Expected 39 targets, found "
            f"{len(target_columns)}"
        )

    # =========================================================
    # 2. Load evaluation data
    # =========================================================

    print(
        "\n[2] Loading evaluation data"
    )

    evaluation = pd.read_csv(
        EVAL_PATH
    )

    evaluation["timestamp"] = pd.to_datetime(
        evaluation["timestamp"]
    )

    print(
        "Evaluation shape:",
        evaluation.shape
    )

    X_eval = evaluation[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    y_eval = evaluation[
        target_columns
    ].to_numpy(
        dtype=np.int8
    )

    # =========================================================
    # 3. Move evaluation features to GPU
    # =========================================================

    print(
        "\n[3] Preparing GPU evaluation matrix"
    )

    import cupy as cp

    X_eval_gpu = cp.asarray(
        X_eval
    )

    X_eval_scaled = scaler.transform(
        X_eval_gpu
    )

    cp.cuda.Stream.null.synchronize()

    # =========================================================
    # 4. Generate probabilities
    # =========================================================

    print(
        "\n[4] Generating probabilities from trained GPU models"
    )

    probability_matrix = []

    for index, code in enumerate(
        target_columns,
        start=1
    ):

        model = models[code]

        probabilities = cp.asarray(
            model.predict_proba(
                X_eval_scaled
            )
        )

        if probabilities.ndim != 2:
            fail(
                f"Unexpected probability shape "
                f"for {code}: {probabilities.shape}"
            )

        if probabilities.shape[1] != 2:
            fail(
                f"Expected two probability columns "
                f"for {code}, got "
                f"{probabilities.shape}"
            )

        probability_matrix.append(
            probabilities[:, 1]
        )

    probability_matrix = cp.stack(
        probability_matrix,
        axis=1
    )

    cp.cuda.Stream.null.synchronize()

    probabilities = cp.asnumpy(
        probability_matrix
    )

    print(
        "Probability matrix:",
        probabilities.shape
    )

    print(
        "Global probability minimum:",
        float(probabilities.min())
    )

    print(
        "Global probability maximum:",
        float(probabilities.max())
    )

    print(
        "Global probability mean:",
        float(probabilities.mean())
    )

    # =========================================================
    # 5. Threshold sweep
    # =========================================================

    print(
        "\n[5] Global threshold sweep"
    )

    thresholds = [
        0.000001,
        0.000005,
        0.00001,
        0.00002,
        0.00005,
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.50,
    ]

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

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

        predicted_positive = int(
            predictions.sum()
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
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

    # =========================================================
    # 6. Find threshold with highest F1
    # =========================================================

    best_micro = threshold_df.loc[
        threshold_df["micro_f1"].idxmax()
    ]

    best_macro = threshold_df.loc[
        threshold_df["macro_f1"].idxmax()
    ]

    print(
        "\nBest Micro-F1 threshold:"
    )

    print(
        best_micro.to_string()
    )

    print(
        "\nBest Macro-F1 threshold:"
    )

    print(
        best_macro.to_string()
    )

    # =========================================================
    # 7. Per-label probability analysis
    # =========================================================

    print(
        "\n[6] Per-label probability analysis"
    )

    label_results = []

    for index, code in enumerate(
        target_columns
    ):

        actual = y_eval[
            :,
            index
        ]

        scores = probabilities[
            :,
            index
        ]

        positive_mask = (
            actual == 1
        )

        negative_mask = (
            actual == 0
        )

        positive_count = int(
            positive_mask.sum()
        )

        negative_count = int(
            negative_mask.sum()
        )

        positive_probabilities = (
            scores[positive_mask]
        )

        negative_probabilities = (
            scores[negative_mask]
        )

        if positive_count > 0:

            positive_mean = float(
                positive_probabilities.mean()
            )

            positive_median = float(
                np.median(
                    positive_probabilities
                )
            )

            positive_max = float(
                positive_probabilities.max()
            )

            positive_p95 = float(
                np.percentile(
                    positive_probabilities,
                    95
                )
            )

        else:

            positive_mean = np.nan
            positive_median = np.nan
            positive_max = np.nan
            positive_p95 = np.nan

        if negative_count > 0:

            negative_mean = float(
                negative_probabilities.mean()
            )

            negative_median = float(
                np.median(
                    negative_probabilities
                )
            )

            negative_max = float(
                negative_probabilities.max()
            )

            negative_p95 = float(
                np.percentile(
                    negative_probabilities,
                    95
                )
            )

        else:

            negative_mean = np.nan
            negative_median = np.nan
            negative_max = np.nan
            negative_p95 = np.nan

        if (
            positive_count > 0
            and negative_count > 0
        ):

            roc_auc = roc_auc_score(
                actual,
                scores
            )

            average_precision = (
                average_precision_score(
                    actual,
                    scores
                )
            )

        else:

            roc_auc = np.nan
            average_precision = np.nan

        # Best threshold for this individual label
        best_threshold = np.nan
        best_f1 = 0.0

        for threshold in thresholds:

            prediction = (
                scores >= threshold
            ).astype(np.int8)

            label_f1 = f1_score(
                actual,
                prediction,
                zero_division=0,
            )

            if label_f1 > best_f1:

                best_f1 = label_f1
                best_threshold = threshold

        label_results.append(
            {
                "attack_code": code,
                "evaluation_positive": positive_count,
                "evaluation_negative": negative_count,
                "positive_mean_probability": positive_mean,
                "positive_median_probability": positive_median,
                "positive_p95_probability": positive_p95,
                "positive_max_probability": positive_max,
                "negative_mean_probability": negative_mean,
                "negative_median_probability": negative_median,
                "negative_p95_probability": negative_p95,
                "negative_max_probability": negative_max,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "best_threshold": best_threshold,
                "best_f1": best_f1,
            }
        )

    label_df = pd.DataFrame(
        label_results
    )

    print(
        label_df.to_string(
            index=False
        )
    )

    # =========================================================
    # 8. Save diagnostics
    # =========================================================

    print(
        "\n[7] Saving diagnostics"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    threshold_df.to_csv(
        THRESHOLD_ANALYSIS_PATH,
        index=False
    )

    label_df.to_csv(
        LABEL_ANALYSIS_PATH,
        index=False
    )

    print(
        "Threshold analysis:",
        THRESHOLD_ANALYSIS_PATH
    )

    print(
        "Probability analysis:",
        LABEL_ANALYSIS_PATH
    )

    # =========================================================
    # 9. Final summary
    # =========================================================

    print("\n" + "=" * 100)
    print("STEP 6.16C COMPLETE")
    print("=" * 100)

    print(
        "\nNo model was retrained."
    )

    print(
        "The existing GPU classifier was only evaluated "
        "at multiple probability thresholds."
    )

    print(
        "\nBest global Micro-F1:",
        f"{best_micro['micro_f1']:.6f}",
        "at threshold",
        best_micro["threshold"]
    )

    print(
        "Best global Macro-F1:",
        f"{best_macro['macro_f1']:.6f}",
        "at threshold",
        best_macro["threshold"]
    )


if __name__ == "__main__":
    main()