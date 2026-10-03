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
    precision_score,
    recall_score,
    f1_score,
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
    / "final_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SCENARIO_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_final_scenario_diagnostics.csv"
)

MECHANISM_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_final_mechanism_diagnostics.csv"
)

TOP_PREDICTION_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_final_top_predictions.csv"
)

NORMAL_OUTPUT = (
    OUTPUT_DIR
    / "hai_2305_final_normal_false_positives.csv"
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


# ============================================================
# HELPERS
# ============================================================

def get_probability(model, X):
    values = np.asarray(
        model.predict_proba(X)
    )

    if values.ndim == 2:
        values = values[:, 1]

    return values.astype(np.float64)


def clip_probability(values):
    return np.clip(
        np.asarray(values, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )


def scenario_attack_codes(frame, target_columns):
    codes = []

    for target in target_columns:
        if int(frame[target].sum()) > 0:
            codes.append(target)

    return codes


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("HAI 23.05 GPU FINAL HELD-OUT EVALUATION DIAGNOSTIC")
print("=" * 100)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[1] Loading existing GPU classifier")

artifact = joblib.load(
    MODEL_PATH
)

scaler = artifact["scaler"]
models = artifact["models"]
feature_columns = artifact["feature_columns"]
target_columns = artifact["target_columns"]

print(
    f"GPU classifiers: {len(models)}"
)

print(
    f"Features: {len(feature_columns)}"
)

print(
    f"Targets: {len(target_columns)}"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2] Loading calibration and final evaluation data")

fit_df = pd.read_csv(FIT_PATH)
eval_df = pd.read_csv(EVAL_PATH)

print(
    f"Calibration-fit rows: {len(fit_df):,}"
)

print(
    f"Evaluation rows:      {len(eval_df):,}"
)


# ============================================================
# FINAL SCENARIO VALIDATION
# ============================================================

evaluation_scenarios = set(
    eval_df["scenario_id"]
    .dropna()
    .astype(str)
)

expected_scenarios = (
    FINAL_SCENARIOS | {"NORMAL"}
)

if evaluation_scenarios != expected_scenarios:
    raise ValueError(
        "Unexpected evaluation scenarios.\n"
        f"Expected: {sorted(expected_scenarios)}\n"
        f"Found: {sorted(evaluation_scenarios)}"
    )

fit_scenarios = set(
    fit_df["scenario_id"]
    .dropna()
    .astype(str)
)

overlap = (
    fit_scenarios
    & FINAL_SCENARIOS
)

if overlap:
    raise ValueError(
        "Calibration/final scenario leakage: "
        f"{sorted(overlap)}"
    )

print(
    "Calibration/final scenario separation: PASS"
)


# ============================================================
# FEATURE CONTRACT
# ============================================================

print("\n[3] Validating feature contract")

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
        f"Missing fit features: {missing_fit[:20]}"
    )

if missing_eval:
    raise ValueError(
        f"Missing evaluation features: {missing_eval[:20]}"
    )

X_fit = (
    fit_df[feature_columns]
    .astype(np.float32)
    .values
)

X_eval = (
    eval_df[feature_columns]
    .astype(np.float32)
    .values
)

Y_eval = (
    eval_df[target_columns]
    .astype(np.int32)
    .values
)

if np.isnan(X_fit).any():
    raise ValueError(
        "Calibration features contain NaN"
    )

if np.isnan(X_eval).any():
    raise ValueError(
        "Evaluation features contain NaN"
    )

if np.isinf(X_fit).any():
    raise ValueError(
        "Calibration features contain infinity"
    )

if np.isinf(X_eval).any():
    raise ValueError(
        "Evaluation features contain infinity"
    )

print(
    f"Feature matrix: {X_eval.shape}"
)

print(
    "NaN/Infinity validation: PASS"
)


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print("\n[4] Generating GPU probabilities")

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
    f"Probability matrix: {raw_eval.shape}"
)


# ============================================================
# FIT CALIBRATORS ONLY ON FIT DATA
# ============================================================

print("\n[5] Creating calibration mappings")

calibrators = {}

for index, target in enumerate(target_columns):

    y_fit = (
        fit_df[target]
        .astype(np.int32)
        .values
    )

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
# CALIBRATED PROBABILITIES
# ============================================================

print("\n[6] Generating calibrated probabilities")

probability_data = {
    "scenario_id": eval_df[
        "scenario_id"
    ].astype(str).values,
}

if "timestamp" in eval_df.columns:
    probability_data["timestamp"] = (
        eval_df["timestamp"].values
    )

for index, target in enumerate(target_columns):

    raw_probability = clip_probability(
        raw_eval[:, index]
    )

    probability_data[
        f"{target}_raw"
    ] = raw_probability

    if target in calibrators:

        sigmoid_probability = clip_probability(
            calibrators[target]["sigmoid"]
            .predict_proba(
                raw_probability.reshape(-1, 1)
            )[:, 1]
        )

        isotonic_probability = clip_probability(
            calibrators[target]["isotonic"]
            .predict(
                raw_probability
            )
        )

        probability_data[
            f"{target}_sigmoid"
        ] = sigmoid_probability

        probability_data[
            f"{target}_isotonic"
        ] = isotonic_probability


probability_df = pd.DataFrame(
    probability_data
)


# ============================================================
# SCENARIO DIAGNOSTICS
# ============================================================

print("\n[7] Per-scenario diagnostics")

scenario_rows = []

attack_scenarios = sorted(
    FINAL_SCENARIOS
)

for scenario in attack_scenarios:

    mask = (
        eval_df["scenario_id"]
        .astype(str)
        == scenario
    )

    scenario_indices = np.where(
        mask.values
    )[0]

    scenario_frame = eval_df.loc[mask]

    true_codes = scenario_attack_codes(
        scenario_frame,
        target_columns,
    )

    raw_values = raw_eval[
        scenario_indices
    ]

    max_per_row = raw_values.max(
        axis=1
    )

    argmax_per_row = raw_values.argmax(
        axis=1
    )

    top_counts = {}

    for label_index in argmax_per_row:

        code = target_columns[
            label_index
        ]

        top_counts[code] = (
            top_counts.get(code, 0) + 1
        )

    top_prediction = max(
        top_counts,
        key=top_counts.get,
    )

    top_prediction_fraction = (
        top_counts[top_prediction]
        / len(scenario_indices)
    )

    scenario_rows.append(
        {
            "scenario_id": scenario,
            "rows": len(scenario_indices),
            "true_attack_codes": ",".join(
                true_codes
            ),
            "true_attack_code_count": len(
                true_codes
            ),
            "mean_max_probability": float(
                max_per_row.mean()
            ),
            "max_probability": float(
                max_per_row.max()
            ),
            "median_max_probability": float(
                np.median(max_per_row)
            ),
            "top_1_predicted_code": (
                top_prediction
            ),
            "top_1_prediction_fraction": (
                top_prediction_fraction
            ),
        }
    )


scenario_df = pd.DataFrame(
    scenario_rows
)

scenario_df.to_csv(
    SCENARIO_OUTPUT,
    index=False,
)


# ============================================================
# TOP-1 / TOP-3 PREDICTIONS
# ============================================================

print("\n[8] Top attack predictions")

top_rows = []

for scenario in attack_scenarios:

    mask = (
        eval_df["scenario_id"]
        .astype(str)
        == scenario
    )

    indices = np.where(
        mask.values
    )[0]

    scenario_probabilities = (
        raw_eval[indices]
    )

    mean_probabilities = (
        scenario_probabilities.mean(
            axis=0
        )
    )

    top_indices = np.argsort(
        mean_probabilities
    )[::-1][:3]

    true_codes = scenario_attack_codes(
        eval_df.loc[mask],
        target_columns,
    )

    print(
        f"\n{scenario}"
    )

    print(
        "  True:",
        ", ".join(true_codes)
    )

    for rank, label_index in enumerate(
        top_indices,
        start=1,
    ):

        code = target_columns[
            label_index
        ]

        probability = (
            mean_probabilities[
                label_index
            ]
        )

        print(
            f"  Top-{rank}: "
            f"{code} "
            f"({probability:.8f})"
        )

        top_rows.append(
            {
                "scenario_id": scenario,
                "rank": rank,
                "predicted_code": code,
                "mean_raw_probability": float(
                    probability
                ),
                "is_true_code": int(
                    code in true_codes
                ),
            }
        )


top_df = pd.DataFrame(
    top_rows
)

top_df.to_csv(
    TOP_PREDICTION_OUTPUT,
    index=False,
)


# ============================================================
# MECHANISM DIAGNOSTICS
# ============================================================

print("\n[9] Mechanism-level diagnostics")

mechanism_rows = []

for index, target in enumerate(
    target_columns
):

    y_true = Y_eval[:, index]

    positive_count = int(
        y_true.sum()
    )

    negative_count = int(
        len(y_true) - positive_count
    )

    raw_probability = clip_probability(
        raw_eval[:, index]
    )

    # --------------------------------------------------------
    # Use threshold 0.5 only as a diagnostic.
    # This is NOT the final operating threshold.
    # --------------------------------------------------------

    raw_prediction = (
        raw_probability >= 0.5
    ).astype(int)

    precision = precision_score(
        y_true,
        raw_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        raw_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        raw_prediction,
        zero_division=0,
    )

    if (
        positive_count > 0
        and negative_count > 0
    ):

        auc = roc_auc_score(
            y_true,
            raw_probability,
        )

        ap = average_precision_score(
            y_true,
            raw_probability,
        )

    else:

        auc = np.nan
        ap = np.nan

    mechanism_rows.append(
        {
            "attack_code": target,
            "positive_rows": positive_count,
            "negative_rows": negative_count,
            "mean_probability": float(
                raw_probability.mean()
            ),
            "max_probability": float(
                raw_probability.max()
            ),
            "precision_at_0.5": precision,
            "recall_at_0.5": recall,
            "f1_at_0.5": f1,
            "roc_auc": auc,
            "average_precision": ap,
        }
    )


mechanism_df = pd.DataFrame(
    mechanism_rows
)

mechanism_df.to_csv(
    MECHANISM_OUTPUT,
    index=False,
)


# ============================================================
# NORMAL FALSE POSITIVES
# ============================================================

print("\n[10] Normal-row false-positive analysis")

normal_mask = (
    eval_df["scenario_id"]
    .astype(str)
    == "NORMAL"
)

normal_indices = np.where(
    normal_mask.values
)[0]

normal_probability_matrix = raw_eval[
    normal_indices
]

normal_max = (
    normal_probability_matrix.max(
        axis=1
    )
)

normal_argmax = (
    normal_probability_matrix.argmax(
        axis=1
    )
)

false_positive_mask = (
    normal_max >= 0.5
)

false_positive_indices = (
    normal_indices[
        false_positive_mask
    ]
)

normal_rows = []

for local_index in np.where(
    false_positive_mask
)[0]:

    original_index = (
        normal_indices[local_index]
    )

    label_index = normal_argmax[
        local_index
    ]

    normal_rows.append(
        {
            "row_index": int(
                original_index
            ),
            "predicted_code": (
                target_columns[
                    label_index
                ]
            ),
            "raw_probability": float(
                normal_max[local_index]
            ),
        }
    )


normal_fp_df = pd.DataFrame(
    normal_rows
)

normal_fp_df.to_csv(
    NORMAL_OUTPUT,
    index=False,
)


print(
    f"Normal rows: {len(normal_indices):,}"
)

print(
    f"Normal false positives "
    f"(threshold 0.50): "
    f"{len(normal_rows):,}"
)

if len(normal_indices) > 0:

    print(
        "Maximum normal-row probability:",
        f"{normal_max.max():.8f}",
    )

    print(
        "Mean normal-row maximum probability:",
        f"{normal_max.mean():.8f}",
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n[11] Final diagnostic summary")

for scenario in attack_scenarios:

    row = scenario_df[
        scenario_df["scenario_id"]
        == scenario
    ].iloc[0]

    print(
        f"{scenario} | "
        f"true={row['true_attack_codes']} | "
        f"top1={row['top_1_predicted_code']} | "
        f"top1_fraction="
        f"{row['top_1_prediction_fraction']:.3f} | "
        f"max_prob="
        f"{row['max_probability']:.6f}"
    )


print("\n" + "=" * 100)
print("STEP 6.16G COMPLETE")
print("=" * 100)

print("\nNo GPU classifier was retrained.")
print("No frozen model was modified.")
print("Final evaluation was used only for diagnosis.")

print(f"\nScenario diagnostics:")
print(SCENARIO_OUTPUT)

print(f"\nMechanism diagnostics:")
print(MECHANISM_OUTPUT)

print(f"\nTop predictions:")
print(TOP_PREDICTION_OUTPUT)

print(f"\nNormal false positives:")
print(NORMAL_OUTPUT)
