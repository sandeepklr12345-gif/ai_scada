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
    / "calibration"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = (
    OUTPUT_DIR
    / "hai_2305_final_calibration_evaluation.csv"
)

PROBABILITY_PATH = (
    OUTPUT_DIR
    / "hai_2305_final_calibration_probabilities.csv"
)


FINAL_SCENARIOS = {
    "A201",
    "A203",
    "A209",
    "A213",
    "A214",
    "A235",
    "A236",
}


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


print("=" * 100)
print("HAI 23.05 GPU FINAL UNTOUCHED CALIBRATION EVALUATION")
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


# ============================================================
# LOAD CALIBRATION-FIT DATA
# ============================================================

print("\n[2] Loading calibration-fit data")

fit_df = pd.read_csv(FIT_PATH)

print(f"Calibration-fit rows: {len(fit_df):,}")


# ============================================================
# LOAD FINAL EVALUATION
# ============================================================

print("\n[3] Loading final evaluation data")

eval_df = pd.read_csv(EVAL_PATH)

print(f"Evaluation rows: {len(eval_df):,}")


if "scenario_id" not in eval_df.columns:
    raise ValueError("scenario_id column missing")


evaluation_scenarios = set(
    eval_df["scenario_id"]
    .dropna()
    .astype(str)
)

print(
    "Evaluation scenarios:",
    sorted(evaluation_scenarios),
)


expected_evaluation_scenarios = (
    FINAL_SCENARIOS | {"NORMAL"}
)

if evaluation_scenarios != expected_evaluation_scenarios:
    raise ValueError(
        "Final evaluation scenario set does not match "
        "expected seven attack scenarios plus NORMAL.\n"
        f"Expected: {sorted(expected_evaluation_scenarios)}\n"
        f"Found: {sorted(evaluation_scenarios)}"
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

fit_scenarios = set(
    fit_df["scenario_id"]
    .dropna()
    .astype(str)
)

overlap = fit_scenarios & FINAL_SCENARIOS

if overlap:
    raise ValueError(
        "Calibration fit contains final evaluation scenarios: "
        f"{sorted(overlap)}"
    )

print("Calibration/test scenario separation: PASS")


# ============================================================
# FEATURE CONTRACT
# ============================================================

missing_fit = [
    c for c in feature_columns
    if c not in fit_df.columns
]

missing_eval = [
    c for c in feature_columns
    if c not in eval_df.columns
]

if missing_fit:
    raise ValueError(
        f"Missing calibration features: {missing_fit[:20]}"
    )

if missing_eval:
    raise ValueError(
        f"Missing evaluation features: {missing_eval[:20]}"
    )


X_fit = fit_df[
    feature_columns
].astype(np.float32).values

Y_fit = fit_df[
    target_columns
].astype(np.int32).values

X_eval = eval_df[
    feature_columns
].astype(np.float32).values

Y_eval = eval_df[
    target_columns
].astype(np.int32).values


if np.isnan(X_fit).any():
    raise ValueError(
        "Calibration features contain NaN"
    )

if np.isnan(X_eval).any():
    raise ValueError(
        "Evaluation features contain NaN"
    )


# ============================================================
# GPU PROBABILITIES
# ============================================================

print("\n[4] Generating probabilities")

X_fit_scaled = scaler.transform(X_fit)
X_eval_scaled = scaler.transform(X_eval)

raw_fit = np.zeros(
    (len(fit_df), len(target_columns)),
    dtype=np.float64,
)

raw_eval = np.zeros(
    (len(eval_df), len(target_columns)),
    dtype=np.float64,
)

for index, target in enumerate(target_columns):

    model = models[target]

    raw_fit[:, index] = get_probability(
        model,
        X_fit_scaled,
    )

    raw_eval[:, index] = get_probability(
        model,
        X_eval_scaled,
    )

print(
    f"Fit probability matrix: "
    f"{raw_fit.shape}"
)

print(
    f"Evaluation probability matrix: "
    f"{raw_eval.shape}"
)


# ============================================================
# FIT CALIBRATORS USING FIT DATA ONLY
# ============================================================

print("\n[5] Fitting calibration mappings")

calibrators = {}

for index, target in enumerate(target_columns):

    y_fit = Y_fit[:, index]
    p_fit = clip_probability(
        raw_fit[:, index]
    )

    positives = int(y_fit.sum())
    negatives = len(y_fit) - positives

    if positives == 0 or negatives == 0:
        continue

    sigmoid = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    sigmoid.fit(
        p_fit.reshape(-1, 1),
        y_fit,
    )

    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic.fit(
        p_fit,
        y_fit,
    )

    calibrators[target] = {
        "sigmoid": sigmoid,
        "isotonic": isotonic,
    }


# ============================================================
# EVALUATE
# ============================================================

print("\n[6] Evaluating untouched scenarios")

results = []
probability_columns = {
    "timestamp": eval_df["timestamp"]
    if "timestamp" in eval_df.columns
    else np.arange(len(eval_df)),
    "scenario_id": eval_df["scenario_id"],
}

for index, target in enumerate(target_columns):

    y_true = Y_eval[:, index]

    positives = int(y_true.sum())
    negatives = len(y_true) - positives

    p_raw = clip_probability(
        raw_eval[:, index]
    )

    # Raw
    raw_brier = brier_score_loss(
        y_true,
        p_raw,
    )

    raw_logloss = log_loss(
        y_true,
        p_raw,
        labels=[0, 1],
    )

    raw_auc = roc_auc_score(
        y_true,
        p_raw,
    )

    raw_ap = average_precision_score(
        y_true,
        p_raw,
    )

    results.append(
        {
            "attack_code": target,
            "method": "raw",
            "positive_count": positives,
            "negative_count": negatives,
            "brier_score": raw_brier,
            "log_loss": raw_logloss,
            "roc_auc": raw_auc,
            "average_precision": raw_ap,
        }
    )

    if target not in calibrators:
        continue

    # Sigmoid
    sigmoid = calibrators[target]["sigmoid"]

    p_sigmoid = sigmoid.predict_proba(
        p_raw.reshape(-1, 1)
    )[:, 1]

    p_sigmoid = clip_probability(
        p_sigmoid
    )

    results.append(
        {
            "attack_code": target,
            "method": "sigmoid",
            "positive_count": positives,
            "negative_count": negatives,
            "brier_score": brier_score_loss(
                y_true,
                p_sigmoid,
            ),
            "log_loss": log_loss(
                y_true,
                p_sigmoid,
                labels=[0, 1],
            ),
            "roc_auc": roc_auc_score(
                y_true,
                p_sigmoid,
            ),
            "average_precision": average_precision_score(
                y_true,
                p_sigmoid,
            ),
        }
    )

    # Isotonic
    isotonic = calibrators[target]["isotonic"]

    p_isotonic = isotonic.predict(
        p_raw
    )

    p_isotonic = clip_probability(
        p_isotonic
    )

    results.append(
        {
            "attack_code": target,
            "method": "isotonic",
            "positive_count": positives,
            "negative_count": negatives,
            "brier_score": brier_score_loss(
                y_true,
                p_isotonic,
            ),
            "log_loss": log_loss(
                y_true,
                p_isotonic,
                labels=[0, 1],
            ),
            "roc_auc": roc_auc_score(
                y_true,
                p_isotonic,
            ),
            "average_precision": average_precision_score(
                y_true,
                p_isotonic,
            ),
        }
    )

    probability_columns[
        f"{target}_raw_probability"
    ] = p_raw

    probability_columns[
        f"{target}_sigmoid_probability"
    ] = p_sigmoid

    probability_columns[
        f"{target}_isotonic_probability"
    ] = p_isotonic


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(results)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

print("\n[7] FINAL UNTOUCHED EVALUATION SUMMARY")

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
# SAVE PROBABILITIES
# ============================================================

probability_df = pd.DataFrame(
    probability_columns
)

probability_df.to_csv(
    PROBABILITY_PATH,
    index=False,
)


# ============================================================
# PER-LABEL COMPARISON
# ============================================================

print("\n[8] Per-mechanism final comparison")

for target in target_columns:

    subset = metrics_df[
        metrics_df["attack_code"] == target
    ]

    if subset.empty:
        continue

    print(f"\n{target}")

    for _, row in subset.iterrows():

        print(
            f"  {row['method']:8s} | "
            f"Brier={row['brier_score']:.8f} | "
            f"LogLoss={row['log_loss']:.8f} | "
            f"AUC={row['roc_auc']:.6f} | "
            f"AP={row['average_precision']:.6f}"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("STEP 6.16F COMPLETE")
print("=" * 100)

print("\nFINAL evaluation scenarios were used ONLY for evaluation.")
print("No calibrator was fitted using final evaluation rows.")
print("No GPU classifier was retrained.")
print("No frozen model was modified.")

print(f"\nMetrics:       {METRICS_PATH}")
print(f"Probabilities: {PROBABILITY_PATH}")