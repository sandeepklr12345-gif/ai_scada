from pathlib import Path
import json
import time

import cupy as cp
import cudf
import joblib
import numpy as np
import pandas as pd

from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
from cuml.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
    / "temporal_gpu"
    / "train.csv"
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

MODEL_PATH = (
    OUTPUT_DIR
    / "hai_2305_temporal_multilabel_gpu.joblib"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "hai_2305_temporal_multilabel_gpu_manifest.json"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "hai_2305_temporal_multilabel_gpu_metrics.csv"
)

THRESHOLD = 0.50


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
    print("HAI 23.05 TEMPORAL MULTI-LABEL GPU CLASSIFIER")
    print("=" * 100)

    # =========================================================
    # 1. GPU CHECK
    # =========================================================

    print("\n[1] Checking GPU")

    device_count = cp.cuda.runtime.getDeviceCount()

    if device_count < 1:
        fail("No CUDA GPU detected.")

    device = cp.cuda.Device()

    props = cp.cuda.runtime.getDeviceProperties(
        device.id
    )

    gpu_name = props["name"]

    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    free_memory, total_memory = (
        cp.cuda.runtime.memGetInfo()
    )

    print(
        "CUDA devices:",
        device_count
    )

    print(
        "GPU:",
        gpu_name
    )

    print(
        f"GPU memory: "
        f"{free_memory / 1024**2:.1f} MiB free / "
        f"{total_memory / 1024**2:.1f} MiB total"
    )

    # =========================================================
    # 2. LOAD DATA
    # =========================================================

    print("\n[2] Loading train/evaluation splits")

    train_pd = pd.read_csv(
        TRAIN_PATH
    )

    eval_pd = pd.read_csv(
        EVAL_PATH
    )

    train_pd["timestamp"] = pd.to_datetime(
        train_pd["timestamp"]
    )

    eval_pd["timestamp"] = pd.to_datetime(
        eval_pd["timestamp"]
    )

    print(
        "Training:",
        train_pd.shape
    )

    print(
        "Evaluation:",
        eval_pd.shape
    )

    # =========================================================
    # 3. IDENTIFY FEATURES AND TARGETS
    # =========================================================

    print(
        "\n[3] Identifying temporal features and targets"
    )

    target_columns = [
        column
        for column in train_pd.columns
        if column.startswith("AP")
        or column.startswith("AE")
    ]

    feature_columns = [
        column
        for column in train_pd.columns
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
            f"Expected 408 features, "
            f"found {len(feature_columns)}."
        )

    if len(target_columns) != 39:
        fail(
            f"Expected 39 targets, "
            f"found {len(target_columns)}."
        )

    # =========================================================
    # 4. BASIC VALIDATION
    # =========================================================

    print(
        "\n[4] Validating data"
    )

    if train_pd[
        feature_columns
    ].isna().any().any():

        fail(
            "Training features contain missing values."
        )

    if eval_pd[
        feature_columns
    ].isna().any().any():

        fail(
            "Evaluation features contain missing values."
        )

    missing_train_labels = [
        code
        for code in target_columns
        if train_pd[code].sum() == 0
    ]

    if missing_train_labels:
        fail(
            "Training is missing labels: "
            + str(missing_train_labels)
        )

    print(
        "PASS: Features and targets validated."
    )

    # =========================================================
    # 5. MOVE FEATURE MATRICES TO GPU
    # =========================================================

    print(
        "\n[5] Moving feature matrices to GPU"
    )

    X_train_cpu = train_pd[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    X_eval_cpu = eval_pd[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    X_train_gpu = cp.asarray(
        X_train_cpu
    )

    X_eval_gpu = cp.asarray(
        X_eval_cpu
    )

    print(
        "X_train GPU:",
        X_train_gpu.shape,
        X_train_gpu.dtype
    )

    print(
        "X_eval GPU:",
        X_eval_gpu.shape,
        X_eval_gpu.dtype
    )

    cp.cuda.Stream.null.synchronize()

    # =========================================================
    # 6. GPU STANDARDIZATION
    # =========================================================

    print(
        "\n[6] GPU feature scaling"
    )

    scaler = StandardScaler()

    scaler_start = time.perf_counter()

    X_train_scaled = scaler.fit_transform(
        X_train_gpu
    )

    X_eval_scaled = scaler.transform(
        X_eval_gpu
    )

    cp.cuda.Stream.null.synchronize()

    scaler_time = (
        time.perf_counter()
        - scaler_start
    )

    print(
        f"GPU scaling time: {scaler_time:.4f} seconds"
    )

    # =========================================================
    # 7. TRAIN 39 GPU CLASSIFIERS
    # =========================================================

    print(
        "\n[7] Training 39 GPU binary classifiers"
    )

    models = {}

    probability_columns = []

    training_times = []

    total_train_start = (
        time.perf_counter()
    )

    for index, code in enumerate(
        target_columns,
        start=1
    ):

        print(
            f"\n[{index:02d}/39] Training {code}"
        )

        y_train = cp.asarray(
            train_pd[code].to_numpy(
                dtype=np.float32
            )
        )

        positive_count = int(
            train_pd[code].sum()
        )

        negative_count = (
            len(train_pd)
            - positive_count
        )

        print(
            f"Positive: {positive_count}, "
            f"Negative: {negative_count}"
        )

        model = cuMLLogisticRegression(
            max_iter=1000,
            C=1.0,
            penalty="l2",
            tol=1e-4,
            fit_intercept=True,
        )

        start = time.perf_counter()

        model.fit(
            X_train_scaled,
            y_train
        )

        cp.cuda.Stream.null.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        models[code] = model

        training_times.append(
            {
                "attack_code": code,
                "positive_training_rows": positive_count,
                "negative_training_rows": negative_count,
                "training_time_seconds": elapsed,
            }
        )

        print(
            f"GPU training time: {elapsed:.4f}s"
        )

    cp.cuda.Stream.null.synchronize()

    total_train_time = (
        time.perf_counter()
        - total_train_start
    )

    print(
        "\nTotal GPU classifier training time:",
        f"{total_train_time:.4f} seconds"
    )

    # =========================================================
    # 8. GPU PROBABILITY PREDICTION
    # =========================================================

    print(
        "\n[8] Generating GPU probabilities"
    )

    probability_start = (
        time.perf_counter()
    )

    probability_matrix = []

    for code in target_columns:

        model = models[code]

        probabilities = model.predict_proba(
            X_eval_scaled
        )

        probabilities = cp.asarray(
            probabilities
        )

        # cuML binary predict_proba normally returns
        # two columns: P(class=0), P(class=1)
        if probabilities.ndim == 2:

            if probabilities.shape[1] != 2:
                fail(
                    f"Unexpected probability shape "
                    f"for {code}: "
                    f"{probabilities.shape}"
                )

            positive_probability = (
                probabilities[:, 1]
            )

        else:

            positive_probability = probabilities

        probability_matrix.append(
            positive_probability
        )

    probability_matrix = cp.stack(
        probability_matrix,
        axis=1
    )

    cp.cuda.Stream.null.synchronize()

    probability_time = (
        time.perf_counter()
        - probability_start
    )

    print(
        "Probability matrix:",
        probability_matrix.shape
    )

    print(
        f"GPU prediction time: "
        f"{probability_time:.4f} seconds"
    )

    # =========================================================
    # 9. TRANSFER ONLY RESULTS TO CPU
    # =========================================================

    print(
        "\n[9] Transferring probabilities to CPU"
    )

    probabilities = (
        cp.asnumpy(
            probability_matrix
        )
    )

    y_eval = eval_pd[
        target_columns
    ].to_numpy(
        dtype=int
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # =========================================================
    # 10. METRICS
    # =========================================================

    print(
        "\n[10] GPU classifier metrics"
    )

    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
        roc_auc_score,
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

    # =========================================================
    # 11. PER-LABEL METRICS
    # =========================================================

    print(
        "\n[11] Per-label GPU results"
    )

    per_label = []

    for index, code in enumerate(
        target_columns
    ):

        actual = y_eval[
            :,
            index
        ]

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

            average_precision = (
                average_precision_score(
                    actual,
                    score
                )
            )

        else:

            roc_auc = np.nan
            average_precision = np.nan

        per_label.append(
            {
                "attack_code": code,
                "train_positive": int(
                    train_pd[code].sum()
                ),
                "evaluation_positive": positive_count,
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
                "average_precision": average_precision,
            }
        )

    metrics_df = pd.DataFrame(
        per_label
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    # =========================================================
    # 12. PROBABILITY VALIDATION
    # =========================================================

    print(
        "\n[12] Probability validation"
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
            "GPU probabilities outside [0,1]."
        )

    print(
        "PASS: GPU probabilities valid."
    )

    # =========================================================
    # 13. GPU MEMORY AFTER TRAINING
    # =========================================================

    free_memory, total_memory = (
        cp.cuda.runtime.memGetInfo()
    )

    used_memory = (
        total_memory
        - free_memory
    )

    print(
        "\n[13] GPU memory after training"
    )

    print(
        f"Used: {used_memory / 1024**2:.1f} MiB"
    )

    print(
        f"Free: {free_memory / 1024**2:.1f} MiB"
    )

    print(
        f"Total: {total_memory / 1024**2:.1f} MiB"
    )

    # =========================================================
    # 14. SAVE ARTIFACTS
    # =========================================================

    print(
        "\n[14] Saving GPU classifier"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    artifact = {
        "scaler": scaler,
        "models": models,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "threshold": THRESHOLD,
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False
    )

    manifest = {
        "model_name": (
            "HAI 23.05 Temporal "
            "Multi-label GPU Classifier"
        ),
        "model_family": (
            "cuML Logistic Regression "
            "One-vs-Rest"
        ),
        "gpu_name": gpu_name,
        "feature_count": len(
            feature_columns
        ),
        "target_count": len(
            target_columns
        ),
        "threshold": THRESHOLD,
        "train_rows": len(train_pd),
        "evaluation_rows": len(eval_pd),
        "train_attack_rows": int(
            train_pd["attack_label"].sum()
        ),
        "evaluation_attack_rows": int(
            eval_pd["attack_label"].sum()
        ),
        "train_scenarios": sorted(
            train_pd.loc[
                train_pd["attack_label"] == 1,
                "scenario_id"
            ].dropna().unique().tolist()
        ),
        "evaluation_scenarios": sorted(
            eval_pd.loc[
                eval_pd["attack_label"] == 1,
                "scenario_id"
            ].dropna().unique().tolist()
        ),
        "gpu_scaling_time_seconds": scaler_time,
        "gpu_training_time_seconds": total_train_time,
        "gpu_prediction_time_seconds": probability_time,
        "probability_source": (
            "cuML Logistic Regression "
            "predict_proba"
        ),
        "candidate_C_modified": False,
        "frozen_models_modified": False,
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
        "\nModel:",
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

    # =========================================================
    # 15. FINAL
    # =========================================================

    print("\n" + "=" * 100)
    print("STEP 6.16B GPU TRAINING COMPLETE")
    print("=" * 100)

    print(
        "\nRTX 4050 GPU training was completed."
    )

    print(
        "Candidate C was not modified."
    )

    print(
        "Frozen 600 MW models were not modified."
    )


if __name__ == "__main__":
    main()