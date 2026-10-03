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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = (
    OUTPUT_DIR
    / "hai_2305_calibration_oos_metrics.csv"
)


# ============================================================
# HELPERS
# ============================================================

def get_probability(model, X):
    values = np.asarray(model.predict_proba(X))

    if values.ndim == 2:
        values = values[:, 1]

    return values.astype(np.float64)


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
print("HAI 23.05 GPU CALIBRATION OUT-OF-SAMPLE VALIDATION")
print("=" * 100)


# ============================================================
# LOAD GPU CLASSIFIER
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


# ============================================================
# LOAD CALIBRATION DATA
# ============================================================

print("\n[2] Loading calibration dataset")

df = pd.read_csv(CALIBRATION_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Calibration rows: {len(df):,}")


# ============================================================
# TEMPORAL SPLIT
# ============================================================

print("\n[3] Creating out-of-sample calibration split")

# Use chronological 60/40 split.
#
# Earlier observations are used to FIT the calibrator.
# Later observations are used ONLY to VALIDATE the calibrator.
#
# This is deliberately different from random row splitting.

cutoff = int(len(df) * 0.60)

fit_df = df.iloc[:cutoff].copy()
validation_df = df.iloc[cutoff:].copy()

print(f"Calibration-fit rows:      {len(fit_df):,}")
print(f"Calibration-validation:    {len(validation_df):,}")


# ============================================================
# FEATURE / TARGET MATRICES
# ============================================================

X_fit = fit_df[feature_columns].astype(np.float32).values
Y_fit = fit_df[target_columns].astype(np.int32).values

X_validation = (
    validation_df[feature_columns]
    .astype(np.float32)
    .values
)

Y_validation = (
    validation_df[target_columns]
    .astype(np.int32)
    .values
)


# ============================================================
# RAW GPU PROBABILITIES
# ============================================================

print("\n[4] Generating raw GPU probabilities")

X_fit_scaled = scaler.transform(X_fit)
X_validation_scaled = scaler.transform(X_validation)

raw_fit = np.zeros(
    (len(fit_df), len(target_columns)),
    dtype=np.float64,
)

raw_validation = np.zeros(
    (len(validation_df), len(target_columns)),
    dtype=np.float64,
)

for index, target in enumerate(target_columns):

    model = models[target]

    raw_fit[:, index] = get_probability(
        model,
        X_fit_scaled,
    )

    raw_validation[:, index] = get_probability(
        model,
        X_validation_scaled,
    )

print(
    f"Fit probability matrix: "
    f"{raw_fit.shape}"
)

print(
    f"Validation probability matrix: "
    f"{raw_validation.shape}"
)


# ============================================================
# CALIBRATION
# ============================================================

print("\n[5] Fitting calibrators on earlier data only")

rows = []

for index, target in enumerate(target_columns):

    y_fit = Y_fit[:, index]
    y_validation = Y_validation[:, index]

    p_fit = raw_fit[:, index]
    p_validation = raw_validation[:, index]

    fit_positive = int(y_fit.sum())
    fit_negative = int(len(y_fit) - fit_positive)

    validation_positive = int(y_validation.sum())
    validation_negative = int(
        len(y_validation) - validation_positive
    )

    print(
        f"{target:>5} | "
        f"fit+={fit_positive:4d} | "
        f"val+={validation_positive:4d}"
    )

    # Both calibration fitting and validation require both classes.
    if fit_positive == 0 or fit_negative == 0:
        continue

    if validation_positive == 0 or validation_negative == 0:
        continue

    p_fit_clipped = clip_probability(p_fit)
    p_validation_clipped = clip_probability(
        p_validation
    )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw_brier = brier_score_loss(
        y_validation,
        p_validation_clipped,
    )

    raw_logloss = log_loss(
        y_validation,
        p_validation_clipped,
        labels=[0, 1],
    )

    raw_auc = roc_auc_score(
        y_validation,
        p_validation,
    )

    raw_ap = average_precision_score(
        y_validation,
        p_validation,
    )

    # --------------------------------------------------------
    # SIGMOID
    # --------------------------------------------------------

    sigmoid = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    sigmoid.fit(
        p_fit_clipped.reshape(-1, 1),
        y_fit,
    )

    sigmoid_validation = sigmoid.predict_proba(
        p_validation_clipped.reshape(-1, 1)
    )[:, 1]

    sigmoid_validation = clip_probability(
        sigmoid_validation
    )

    sigmoid_brier = brier_score_loss(
        y_validation,
        sigmoid_validation,
    )

    sigmoid_logloss = log_loss(
        y_validation,
        sigmoid_validation,
        labels=[0, 1],
    )

    sigmoid_auc = roc_auc_score(
        y_validation,
        sigmoid_validation,
    )

    sigmoid_ap = average_precision_score(
        y_validation,
        sigmoid_validation,
    )

    # --------------------------------------------------------
    # ISOTONIC
    # --------------------------------------------------------

    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic.fit(
        p_fit_clipped,
        y_fit,
    )

    isotonic_validation = isotonic.predict(
        p_validation_clipped
    )

    isotonic_validation = clip_probability(
        isotonic_validation
    )

    isotonic_brier = brier_score_loss(
        y_validation,
        isotonic_validation,
    )

    isotonic_logloss = log_loss(
        y_validation,
        isotonic_validation,
        labels=[0, 1],
    )

    isotonic_auc = roc_auc_score(
        y_validation,
        isotonic_validation,
    )

    isotonic_ap = average_precision_score(
        y_validation,
        isotonic_validation,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    rows.extend(
        [
            {
                "attack_code": target,
                "method": "raw",
                "fit_positive": fit_positive,
                "validation_positive": validation_positive,
                "brier_score": raw_brier,
                "log_loss": raw_logloss,
                "roc_auc": raw_auc,
                "average_precision": raw_ap,
            },
            {
                "attack_code": target,
                "method": "sigmoid",
                "fit_positive": fit_positive,
                "validation_positive": validation_positive,
                "brier_score": sigmoid_brier,
                "log_loss": sigmoid_logloss,
                "roc_auc": sigmoid_auc,
                "average_precision": sigmoid_ap,
            },
            {
                "attack_code": target,
                "method": "isotonic",
                "fit_positive": fit_positive,
                "validation_positive": validation_positive,
                "brier_score": isotonic_brier,
                "log_loss": isotonic_logloss,
                "roc_auc": isotonic_auc,
                "average_precision": isotonic_ap,
            },
        ]
    )


# ============================================================
# SAVE
# ============================================================

metrics_df = pd.DataFrame(rows)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n[6] Out-of-sample calibration summary")

for method in [
    "raw",
    "sigmoid",
    "isotonic",
]:

    subset = metrics_df[
        metrics_df["method"] == method
    ]

    if subset.empty:
        continue

    print(
        f"{method:8s} | "
        f"Labels={len(subset):2d} | "
        f"Mean Brier={subset['brier_score'].mean():.8f} | "
        f"Mean LogLoss={subset['log_loss'].mean():.8f} | "
        f"Mean ROC-AUC={subset['roc_auc'].mean():.6f} | "
        f"Mean AP={subset['average_precision'].mean():.6f}"
    )


# ============================================================
# PER-LABEL BEST METHOD
# ============================================================

print("\n[7] Per-label best calibration method")

for target in target_columns:

    subset = metrics_df[
        metrics_df["attack_code"] == target
    ]

    if subset.empty:
        continue

    candidates = subset[
        subset["method"].isin(
            ["sigmoid", "isotonic"]
        )
    ]

    if candidates.empty:
        continue

    best = candidates.sort_values(
        ["brier_score", "log_loss"]
    ).iloc[0]

    raw = subset[
        subset["method"] == "raw"
    ].iloc[0]

    print(
        f"{target:>5} | "
        f"raw={raw['brier_score']:.6f} | "
        f"best={best['method']:8s} | "
        f"{best['brier_score']:.6f}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("STEP 6.16E.4 COMPLETE")
print("=" * 100)

print("\nCalibration mappings were evaluated out-of-sample.")
print("The final seven evaluation scenarios were not used.")
print(f"Results: {METRICS_PATH}")