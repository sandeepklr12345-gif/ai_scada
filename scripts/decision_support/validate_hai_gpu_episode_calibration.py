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

FIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
    / "temporal_gpu"
    / "calibration_episode_fit.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
    / "temporal_gpu"
    / "calibration_episode_validation.csv"
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
    / "hai_2305_episode_calibration_metrics.csv"
)


# ============================================================
# HELPERS
# ============================================================

def get_probability(model, X):
    values = np.asarray(model.predict_proba(X))

    if values.ndim == 2:
        if values.shape[1] == 2:
            values = values[:, 1]
        else:
            values = values.ravel()

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
print("HAI 23.05 GPU EPISODE-LEVEL CALIBRATION VALIDATION")
print("=" * 100)


# ============================================================
# LOAD MODEL
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
        f"Expected 39 targets, found {len(target_columns)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2] Loading calibration fit and validation data")

fit_df = pd.read_csv(FIT_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)



print(f"Fit rows:        {len(fit_df):,}")
print(f"Validation rows: {len(validation_df):,}")


# ============================================================
# VALIDATE FEATURE CONTRACT
# ============================================================

missing_fit = [
    c for c in feature_columns
    if c not in fit_df.columns
]

missing_validation = [
    c for c in feature_columns
    if c not in validation_df.columns
]

if missing_fit:
    raise ValueError(
        f"Missing fit features: {missing_fit[:20]}"
    )

if missing_validation:
    raise ValueError(
        f"Missing validation features: "
        f"{missing_validation[:20]}"
    )


# ============================================================
# BUILD MATRICES
# ============================================================

X_fit = fit_df[
    feature_columns
].astype(np.float32).values

Y_fit = fit_df[
    target_columns
].astype(np.int32).values

X_validation = validation_df[
    feature_columns
].astype(np.float32).values

Y_validation = validation_df[
    target_columns
].astype(np.int32).values


# ============================================================
# GENERATE RAW GPU PROBABILITIES
# ============================================================

print("\n[3] Generating GPU probabilities")

X_fit_scaled = scaler.transform(X_fit)
X_validation_scaled = scaler.transform(
    X_validation
)

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
# CALIBRATION EXPERIMENT
# ============================================================

print("\n[4] Evaluating raw, sigmoid and isotonic")

results = []

for index, target in enumerate(target_columns):

    y_fit = Y_fit[:, index]
    y_validation = Y_validation[:, index]

    p_fit = raw_fit[:, index]
    p_validation = raw_validation[:, index]

    fit_positive = int(y_fit.sum())
    validation_positive = int(
        y_validation.sum()
    )

    fit_negative = len(y_fit) - fit_positive
    validation_negative = (
        len(y_validation) - validation_positive
    )

    print(
        f"{target:>5} | "
        f"fit+={fit_positive:4d} | "
        f"val+={validation_positive:4d}"
    )

    if fit_positive == 0 or fit_negative == 0:
        print("       SKIP: fit lacks both classes")
        continue

    if validation_positive == 0 or validation_negative == 0:
        print(
            "       SKIP: validation lacks both classes"
        )
        continue

    p_fit = clip_probability(p_fit)
    p_validation = clip_probability(
        p_validation
    )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw_brier = brier_score_loss(
        y_validation,
        p_validation,
    )

    raw_logloss = log_loss(
        y_validation,
        p_validation,
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

    results.append(
        {
            "attack_code": target,
            "method": "raw",
            "fit_positive": fit_positive,
            "validation_positive": validation_positive,
            "brier_score": raw_brier,
            "log_loss": raw_logloss,
            "roc_auc": raw_auc,
            "average_precision": raw_ap,
        }
    )

    # --------------------------------------------------------
    # SIGMOID
    # --------------------------------------------------------

    sigmoid = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    sigmoid.fit(
        p_fit.reshape(-1, 1),
        y_fit,
    )

    sigmoid_validation = sigmoid.predict_proba(
        p_validation.reshape(-1, 1)
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

    results.append(
        {
            "attack_code": target,
            "method": "sigmoid",
            "fit_positive": fit_positive,
            "validation_positive": validation_positive,
            "brier_score": sigmoid_brier,
            "log_loss": sigmoid_logloss,
            "roc_auc": sigmoid_auc,
            "average_precision": sigmoid_ap,
        }
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
        p_fit,
        y_fit,
    )

    isotonic_validation = isotonic.predict(
        p_validation
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

    results.append(
        {
            "attack_code": target,
            "method": "isotonic",
            "fit_positive": fit_positive,
            "validation_positive": validation_positive,
            "brier_score": isotonic_brier,
            "log_loss": isotonic_logloss,
            "roc_auc": isotonic_auc,
            "average_precision": isotonic_ap,
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(results)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

print("\n[5] Out-of-sample summary")

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
        f"labels={subset['attack_code'].nunique():2d} | "
        f"mean Brier={subset['brier_score'].mean():.8f} | "
        f"mean LogLoss={subset['log_loss'].mean():.8f} | "
        f"mean ROC-AUC={subset['roc_auc'].mean():.6f} | "
        f"mean AP={subset['average_precision'].mean():.6f}"
    )


# ============================================================
# PER-LABEL SELECTION
# ============================================================

print("\n[6] Best method per attack mechanism")

for target in target_columns:

    subset = metrics_df[
        metrics_df["attack_code"] == target
    ]

    if subset.empty:
        continue

    raw = subset[
        subset["method"] == "raw"
    ]

    calibrated = subset[
        subset["method"].isin(
            ["sigmoid", "isotonic"]
        )
    ]

    if calibrated.empty:
        continue

    best = calibrated.sort_values(
        ["brier_score", "log_loss"]
    ).iloc[0]

    raw_brier = (
        raw["brier_score"].iloc[0]
        if not raw.empty
        else np.nan
    )

    print(
        f"{target:>5} | "
        f"raw={raw_brier:.8f} | "
        f"best={best['method']:8s} | "
        f"brier={best['brier_score']:.8f}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("STEP 6.16E.6 COMPLETE")
print("=" * 100)

print("\nNo GPU classifier was retrained.")
print("No frozen model was modified.")
print("The seven final evaluation scenarios were not used.")
print(f"Results: {METRICS_PATH}")