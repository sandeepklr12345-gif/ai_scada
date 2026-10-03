from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
    / "gpu"
    / "hai_2305_temporal_multilabel_gpu.joblib"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
    / "temporal_gpu"
    / "calibration_temporal.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "attack_classification"
    / "hai_2305"
    / "gpu"
    / "calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CALIBRATOR_PATH = (
    OUTPUT_DIR
    / "hai_2305_probability_calibrators.joblib"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "hai_2305_calibration_metrics.csv"
)

PROBABILITY_PATH = (
    OUTPUT_DIR
    / "hai_2305_calibration_probabilities.csv"
)


# ============================================================
# HELPERS
# ============================================================

def get_probability(model, X):
    probability = model.predict_proba(X)
    probability = np.asarray(probability)

    if probability.ndim == 2:
        if probability.shape[1] == 2:
            probability = probability[:, 1]
        else:
            probability = probability.ravel()

    return probability.astype(np.float64)


def clip_probability(values):
    return np.clip(
        np.asarray(values, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("HAI 23.05 GPU CLASSIFIER PROBABILITY CALIBRATION")
print("=" * 100)


# ============================================================
# LOAD GPU MODEL
# ============================================================

print("\n[1] Loading existing GPU classifier")

artifact = joblib.load(MODEL_PATH)

scaler = artifact["scaler"]
models = artifact["models"]
feature_columns = artifact["feature_columns"]
target_columns = artifact["target_columns"]

print(f"GPU classifiers: {len(models)}")
print(f"Features: {len(feature_columns)}")
print(f"Targets: {len(target_columns)}")


if len(target_columns) != 39:
    raise ValueError(
        f"Expected 39 target columns, found {len(target_columns)}"
    )


# ============================================================
# LOAD CALIBRATION DATA
# ============================================================

print("\n[2] Loading calibration data")

df = pd.read_csv(CALIBRATION_PATH)

print(f"Calibration shape: {df.shape}")


# ============================================================
# VALIDATE FEATURES
# ============================================================

print("\n[3] Validating model feature contract")

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features[:20]}"
    )

missing_targets = [
    column
    for column in target_columns
    if column not in df.columns
]

if missing_targets:
    raise ValueError(
        f"Missing target columns: {missing_targets}"
    )


X = df[feature_columns].astype(np.float32).values
Y = df[target_columns].astype(np.int32).values

print(f"Feature matrix: {X.shape}")
print(f"Target matrix:  {Y.shape}")


# ============================================================
# GENERATE RAW GPU PROBABILITIES
# ============================================================

print("\n[4] Generating raw GPU probabilities")

X_scaled = scaler.transform(X)

raw_probabilities = np.zeros(
    (len(df), len(target_columns)),
    dtype=np.float64,
)

for index, target in enumerate(target_columns):

    raw_probabilities[:, index] = get_probability(
        models[target],
        X_scaled,
    )

print(
    f"Raw probability matrix: "
    f"{raw_probabilities.shape}"
)

print(
    f"Raw probability minimum: "
    f"{raw_probabilities.min():.12g}"
)

print(
    f"Raw probability maximum: "
    f"{raw_probabilities.max():.12g}"
)


# ============================================================
# FIT CALIBRATION METHODS
# ============================================================

print("\n[5] Fitting calibration mappings")

calibrators = {}
metric_rows = []

for index, target in enumerate(target_columns):

    y_true = Y[:, index]
    raw = raw_probabilities[:, index]

    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)

    print(
        f"\n{target}: "
        f"positive={positives}, "
        f"negative={negatives}"
    )

    if positives == 0 or negatives == 0:
        print("  SKIPPED: requires both classes")
        continue

    # --------------------------------------------------------
    # Sigmoid / Platt calibration
    # --------------------------------------------------------

    sigmoid = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    sigmoid.fit(
        raw.reshape(-1, 1),
        y_true,
    )

    sigmoid_probability = sigmoid.predict_proba(
        raw.reshape(-1, 1)
    )[:, 1]

    sigmoid_probability = clip_probability(
        sigmoid_probability
    )

    # --------------------------------------------------------
    # Isotonic calibration
    # --------------------------------------------------------

    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic.fit(
        raw,
        y_true,
    )

    isotonic_probability = isotonic.predict(raw)

    isotonic_probability = clip_probability(
        isotonic_probability
    )

    # --------------------------------------------------------
    # Raw metrics
    # --------------------------------------------------------

    raw_clipped = clip_probability(raw)

    raw_brier = brier_score_loss(
        y_true,
        raw_clipped,
    )

    raw_logloss = log_loss(
        y_true,
        raw_clipped,
        labels=[0, 1],
    )

    raw_auc = roc_auc_score(
        y_true,
        raw,
    )

    raw_ap = average_precision_score(
        y_true,
        raw,
    )

    # --------------------------------------------------------
    # Sigmoid metrics
    # --------------------------------------------------------

    sigmoid_brier = brier_score_loss(
        y_true,
        sigmoid_probability,
    )

    sigmoid_logloss = log_loss(
        y_true,
        sigmoid_probability,
        labels=[0, 1],
    )

    sigmoid_auc = roc_auc_score(
        y_true,
        sigmoid_probability,
    )

    sigmoid_ap = average_precision_score(
        y_true,
        sigmoid_probability,
    )

    # --------------------------------------------------------
    # Isotonic metrics
    # --------------------------------------------------------

    isotonic_brier = brier_score_loss(
        y_true,
        isotonic_probability,
    )

    isotonic_logloss = log_loss(
        y_true,
        isotonic_probability,
        labels=[0, 1],
    )

    isotonic_auc = roc_auc_score(
        y_true,
        isotonic_probability,
    )

    isotonic_ap = average_precision_score(
        y_true,
        isotonic_probability,
    )

    metric_rows.extend(
        [
            {
                "attack_code": target,
                "method": "raw",
                "positive_count": positives,
                "negative_count": negatives,
                "brier_score": raw_brier,
                "log_loss": raw_logloss,
                "roc_auc": raw_auc,
                "average_precision": raw_ap,
                "mean_probability": float(
                    np.mean(raw_clipped)
                ),
            },
            {
                "attack_code": target,
                "method": "sigmoid",
                "positive_count": positives,
                "negative_count": negatives,
                "brier_score": sigmoid_brier,
                "log_loss": sigmoid_logloss,
                "roc_auc": sigmoid_auc,
                "average_precision": sigmoid_ap,
                "mean_probability": float(
                    np.mean(sigmoid_probability)
                ),
            },
            {
                "attack_code": target,
                "method": "isotonic",
                "positive_count": positives,
                "negative_count": negatives,
                "brier_score": isotonic_brier,
                "log_loss": isotonic_logloss,
                "roc_auc": isotonic_auc,
                "average_precision": isotonic_ap,
                "mean_probability": float(
                    np.mean(isotonic_probability)
                ),
            },
        ]
    )

    calibrators[target] = {
        "sigmoid": sigmoid,
        "isotonic": isotonic,
    }


# ============================================================
# METRICS TABLE
# ============================================================

metrics_df = pd.DataFrame(metric_rows)


# ============================================================
# CHOOSE METHOD PER LABEL
# ============================================================

print("\n[6] Selecting calibration method per attack mechanism")

selected_methods = {}

for target in target_columns:

    target_metrics = metrics_df[
        metrics_df["attack_code"] == target
    ]

    if target_metrics.empty:
        continue

    candidates = target_metrics[
        target_metrics["method"].isin(
            ["sigmoid", "isotonic"]
        )
    ].copy()

    if candidates.empty:
        continue

    best = candidates.sort_values(
        by=["brier_score", "log_loss"],
        ascending=[True, True],
    ).iloc[0]

    selected_methods[target] = best["method"]

    print(
        f"{target:>5} -> "
        f"{best['method']:8s} | "
        f"Brier={best['brier_score']:.8f} | "
        f"LogLoss={best['log_loss']:.8f}"
    )


# ============================================================
# SAVE CALIBRATORS
# ============================================================

print("\n[7] Saving calibration artifact")

calibration_artifact = {
    "calibrators": calibrators,
    "selected_methods": selected_methods,
    "target_columns": target_columns,
    "feature_columns": feature_columns,
    "calibration_rows": len(df),
}

joblib.dump(
    calibration_artifact,
    CALIBRATOR_PATH,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# SAVE CALIBRATED PROBABILITIES
# ============================================================

print("\n[8] Generating calibrated probabilities")

calibrated_probability_matrix = np.full(
    raw_probabilities.shape,
    np.nan,
    dtype=np.float64,
)

for index, target in enumerate(target_columns):

    if target not in calibrators:
        continue

    method = selected_methods[target]
    calibrator = calibrators[target][method]

    raw = raw_probabilities[:, index]

    if method == "sigmoid":

        calibrated = calibrator.predict_proba(
            raw.reshape(-1, 1)
        )[:, 1]

    elif method == "isotonic":

        calibrated = calibrator.predict(raw)

    else:
        raise ValueError(
            f"Unknown calibration method: {method}"
        )

    calibrated_probability_matrix[:, index] = (
        clip_probability(calibrated)
    )


probability_df = pd.DataFrame(
    calibrated_probability_matrix,
    columns=[
        f"{target}_calibrated_probability"
        for target in target_columns
    ],
)

probability_df.insert(
    0,
    "timestamp",
    df["timestamp"].values,
)

probability_df.to_csv(
    PROBABILITY_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n[9] Calibration summary")

for method in ["sigmoid", "isotonic"]:

    subset = metrics_df[
        metrics_df["method"] == method
    ]

    if subset.empty:
        continue

    print(
        f"{method:8s} | "
        f"Mean Brier: "
        f"{subset['brier_score'].mean():.8f} | "
        f"Mean LogLoss: "
        f"{subset['log_loss'].mean():.8f}"
    )


print("\nSelected calibration methods:")

method_counts = pd.Series(
    selected_methods
).value_counts()

print(method_counts)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("STEP 6.16E.3 COMPLETE")
print("=" * 100)

print("\nThe GPU classifier itself was NOT retrained.")
print("Calibration mappings were fitted separately.")
print("Final seven-scenario evaluation data was NOT used.")
print(f"Calibration artifact: {CALIBRATOR_PATH}")
print(f"Metrics:              {METRICS_PATH}")
print(f"Probabilities:        {PROBABILITY_PATH}")