from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config.project_paths import PROJECT_ROOT


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(PROJECT_ROOT)

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_OUTPUT = (
    OUTPUT_DIR / "hai_2305_gpu_calibration_analysis.csv"
)

RELIABILITY_OUTPUT = (
    OUTPUT_DIR / "hai_2305_gpu_reliability.csv"
)


# ============================================================
# HELPERS
# ============================================================

def get_target_columns(df):
    return [
        column
        for column in df.columns
        if column.startswith("AP") or column.startswith("AE")
    ]


def safe_float(value):
    if np.isnan(value):
        return np.nan
    return float(value)


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("HAI 23.05 GPU CLASSIFIER CALIBRATION ANALYSIS")
print("=" * 100)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[1] Loading existing GPU classifier")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

if not isinstance(artifact, dict):
    raise ValueError(
        "Unexpected GPU model artifact format. "
        "Expected dictionary containing scaler and classifiers."
    )

print(f"Model path: {MODEL_PATH}")

scaler = artifact["scaler"]
classifiers = artifact["models"]

print(f"Number of classifiers: {len(classifiers)}")


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

print("\n[2] Loading evaluation data")

if not EVAL_PATH.exists():
    raise FileNotFoundError(f"Evaluation dataset not found: {EVAL_PATH}")

df = pd.read_csv(EVAL_PATH)

target_columns = get_target_columns(df)

if len(target_columns) != 39:
    raise ValueError(
        f"Expected 39 target columns, found {len(target_columns)}"
    )

feature_columns = [
    column
    for column in df.columns
    if column not in target_columns
    and column != "timestamp"
    and column not in {
        "scenario_id",
        "attack_codes",
        "attack_label",
    }
]

print(f"Evaluation shape: {df.shape}")
print(f"Features: {len(feature_columns)}")
print(f"Targets: {len(target_columns)}")


# ============================================================
# PREPARE MATRIX
# ============================================================

print("\n[3] Preparing evaluation matrix")

X = df[feature_columns].astype(np.float32).values
Y = df[target_columns].astype(np.int32).values

X_scaled = scaler.transform(X)

print(f"Feature matrix: {X.shape}")


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print("\n[4] Generating probabilities")

probability_matrix = np.zeros(
    (len(df), len(target_columns)),
    dtype=np.float64,
)

for index, target in enumerate(target_columns):
    model = classifiers[target]

    probabilities = model.predict_proba(X_scaled)

    probabilities = np.asarray(probabilities)

    if probabilities.ndim == 2:
        if probabilities.shape[1] == 2:
            probabilities = probabilities[:, 1]
        else:
            probabilities = probabilities.ravel()

    probability_matrix[:, index] = probabilities

print(f"Probability matrix: {probability_matrix.shape}")


# ============================================================
# GLOBAL PROBABILITY STATISTICS
# ============================================================

print("\n[5] Global probability statistics")

print(
    f"Minimum: {probability_matrix.min():.12g}"
)

print(
    f"Maximum: {probability_matrix.max():.12g}"
)

print(
    f"Mean:    {probability_matrix.mean():.12g}"
)

print(
    f"Median:  {np.median(probability_matrix):.12g}"
)


# ============================================================
# PER-LABEL CALIBRATION ANALYSIS
# ============================================================

print("\n[6] Per-label calibration analysis")

calibration_rows = []
reliability_rows = []

for index, target in enumerate(target_columns):

    y_true = Y[:, index].astype(int)
    y_prob = probability_matrix[:, index]

    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)

    row = {
        "attack_code": target,
        "evaluation_positive": positives,
        "evaluation_negative": negatives,
        "mean_probability": float(np.mean(y_prob)),
        "median_probability": float(np.median(y_prob)),
        "max_probability": float(np.max(y_prob)),
        "brier_score": np.nan,
        "mean_positive_probability": np.nan,
        "mean_negative_probability": np.nan,
        "positive_median_probability": np.nan,
        "negative_median_probability": np.nan,
        "calibration_supported": False,
        "note": "",
    }

    if positives == 0:
        row["note"] = "No positive examples in evaluation set"
        calibration_rows.append(row)
        continue

    positive_probabilities = y_prob[y_true == 1]
    negative_probabilities = y_prob[y_true == 0]

    row["mean_positive_probability"] = float(
        np.mean(positive_probabilities)
    )

    row["mean_negative_probability"] = float(
        np.mean(negative_probabilities)
    )

    row["positive_median_probability"] = float(
        np.median(positive_probabilities)
    )

    row["negative_median_probability"] = float(
        np.median(negative_probabilities)
    )

    try:
        row["brier_score"] = float(
            brier_score_loss(y_true, y_prob)
        )
    except ValueError:
        row["note"] = "Brier score unavailable"

    # Calibration curve requires both classes.
    if positives > 0 and negatives > 0:

        row["calibration_supported"] = True

        try:
            fraction_positive, mean_predicted = calibration_curve(
                y_true,
                y_prob,
                n_bins=10,
                strategy="quantile",
            )

            for bin_index, (
                predicted,
                observed,
            ) in enumerate(
                zip(mean_predicted, fraction_positive)
            ):
                reliability_rows.append(
                    {
                        "attack_code": target,
                        "bin": bin_index + 1,
                        "mean_predicted_probability": float(
                            predicted
                        ),
                        "observed_fraction_positive": float(
                            observed
                        ),
                        "absolute_calibration_gap": float(
                            abs(predicted - observed)
                        ),
                    }
                )

        except ValueError as exc:
            row["note"] = f"Calibration curve failed: {exc}"

    calibration_rows.append(row)


calibration_df = pd.DataFrame(calibration_rows)

reliability_df = pd.DataFrame(reliability_rows)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n[7] Saving calibration analysis")

calibration_df.to_csv(
    CALIBRATION_OUTPUT,
    index=False,
)

reliability_df.to_csv(
    RELIABILITY_OUTPUT,
    index=False,
)

print(f"Calibration analysis:")
print(CALIBRATION_OUTPUT)

print(f"Reliability data:")
print(RELIABILITY_OUTPUT)


# ============================================================
# SUMMARY
# ============================================================

print("\n[8] Calibration summary")

supported = calibration_df[
    calibration_df["calibration_supported"]
]

if len(supported) > 0:

    print(
        f"Labels with both positive and negative evaluation examples: "
        f"{len(supported)}"
    )

    print(
        f"Mean Brier score across supported labels: "
        f"{supported['brier_score'].mean():.6f}"
    )

    print(
        f"Median Brier score across supported labels: "
        f"{supported['brier_score'].median():.6f}"
    )

else:
    print("No labels have sufficient evaluation support.")


print("\nLabels with evaluation positives:")

for _, row in calibration_df[
    calibration_df["evaluation_positive"] > 0
].iterrows():

    print(
        f"{row['attack_code']:>5} | "
        f"positive={int(row['evaluation_positive']):4d} | "
        f"Brier={safe_float(row['brier_score'])} | "
        f"mean_pos={safe_float(row['mean_positive_probability'])}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 100)
print("STEP 6.16D CALIBRATION ANALYSIS COMPLETE")
print("=" * 100)

print("\nNo model was retrained.")
print("No frozen model artifact was modified.")
print("The existing GPU classifier was analyzed only.")
