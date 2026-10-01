from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# STAGE 23D
# HAI 23.05 - Temporal Feature / Score Association Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TEMPORAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "temporal_representation"
)

TRAIN_FILE = (
    TEMPORAL_DIR
    / "hai_2305_training_temporal_model_ready.csv"
)

TEST1_FILE = (
    TEMPORAL_DIR
    / "hai-test1_temporal_model_ready.csv"
)

TEST2_FILE = (
    TEMPORAL_DIR
    / "hai-test2_temporal_model_ready.csv"
)

MODEL_FILE = (
    TEMPORAL_DIR
    / "isolation_forest"
    / "hai_2305_temporal_isolation_forest_baseline.joblib"
)

OUTPUT_DIR = (
    TEMPORAL_DIR
    / "isolation_forest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_feature_score_association.csv"
)

FAMILY_SUMMARY_FILE = (
    OUTPUT_DIR
    / "hai_2305_temporal_feature_family_score_association.csv"
)

print("=" * 70)
print("STAGE 23D: TEMPORAL FEATURE / SCORE ASSOCIATION ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("\nLoading Stage 23B Isolation Forest...")

model = joblib.load(
    MODEL_FILE
)

print(
    "PASS: Model loaded"
)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print(
    "\nLoading training temporal model-ready data..."
)

train_df = pd.read_csv(
    TRAIN_FILE
)

all_features = [
    col
    for col in train_df.columns
    if col != "timestamp"
]

assert len(all_features) == 232

print(
    f"Training rows    : {len(train_df):,}"
)

print(
    f"Numeric features : {len(all_features)}"
)


# ============================================================
# 3. REMOVE TEMPORAL BOUNDARY ROWS
# ============================================================

valid_mask = (
    train_df[
        all_features
    ]
    .notna()
    .all(axis=1)
)

valid_df = train_df[
    valid_mask
].copy()

assert len(valid_df) == 896384

print(
    f"Valid rows       : {len(valid_df):,}"
)


# ============================================================
# 4. CALCULATE ISOLATION FOREST SCORES
# ============================================================

print(
    "\nCalculating training anomaly scores..."
)

X_train = (
    valid_df[
        all_features
    ]
    .to_numpy(
        dtype=np.float32
    )
)

assert np.isfinite(
    X_train
).all()

scores = model.decision_function(
    X_train
)

assert np.isfinite(
    scores
).all()

print(
    "PASS: Training scores calculated"
)


# ============================================================
# 5. FEATURE FAMILY CLASSIFICATION
# ============================================================

def classify_feature(
    feature
):

    if "__diff_1s" in feature:
        return "diff_1s"

    if "__abs_diff_1s" in feature:
        return "abs_diff_1s"

    if "__rolling_std_5s" in feature:
        return "rolling_std_5s"

    return "original"


def base_feature_name(
    feature
):

    for suffix in [
        "__diff_1s",
        "__abs_diff_1s",
        "__rolling_std_5s"
    ]:

        if feature.endswith(
            suffix
        ):
            return feature[
                :-len(suffix)
            ]

    return feature


# ============================================================
# 6. FEATURE ASSOCIATION
# ============================================================

print(
    "\nCalculating feature/score associations..."
)

results = []

for index, feature in enumerate(
    all_features,
    start=1
):

    values = valid_df[
        feature
    ].to_numpy(
        dtype=np.float64
    )

    finite_mask = (
        np.isfinite(values)
        &
        np.isfinite(scores)
    )

    x = values[
        finite_mask
    ]

    y = scores[
        finite_mask
    ]

    if (
        len(x) < 2
        or np.std(x) == 0
        or np.std(y) == 0
    ):

        pearson_r = np.nan
        pearson_p = np.nan
        spearman_r = np.nan
        spearman_p = np.nan

    else:

        pearson_r, pearson_p = (
            pearsonr(
                x,
                y
            )
        )

        spearman_r, spearman_p = (
            spearmanr(
                x,
                y
            )
        )

    family = classify_feature(
        feature
    )

    results.append(
        {
            "feature": feature,
            "base_feature": base_feature_name(
                feature
            ),
            "feature_family": family,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "abs_pearson_r": (
                abs(pearson_r)
                if np.isfinite(
                    pearson_r
                )
                else np.nan
            ),
            "abs_spearman_r": (
                abs(spearman_r)
                if np.isfinite(
                    spearman_r
                )
                else np.nan
            ),
            "score_direction": (
                "lower_score_with_higher_feature"
                if np.isfinite(
                    pearson_r
                ) and pearson_r < 0
                else
                "higher_score_with_higher_feature"
                if np.isfinite(
                    pearson_r
                ) and pearson_r > 0
                else
                "undefined"
            )
        }
    )

    if index % 25 == 0:
        print(
            f"Processed {index}/"
            f"{len(all_features)} features"
        )


results_df = pd.DataFrame(
    results
)


# ============================================================
# 7. RANK FEATURES
# ============================================================

results_df = (
    results_df
    .sort_values(
        [
            "abs_spearman_r",
            "abs_pearson_r"
        ],
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

results_df[
    "overall_rank"
] = np.arange(
    1,
    len(results_df) + 1
)


# ============================================================
# 8. FAMILY SUMMARY
# ============================================================

family_summary = (
    results_df
    .groupby(
        "feature_family"
    )
    .agg(
        feature_count=(
            "feature",
            "count"
        ),
        mean_abs_pearson=(
            "abs_pearson_r",
            "mean"
        ),
        median_abs_pearson=(
            "abs_pearson_r",
            "median"
        ),
        max_abs_pearson=(
            "abs_pearson_r",
            "max"
        ),
        mean_abs_spearman=(
            "abs_spearman_r",
            "mean"
        ),
        median_abs_spearman=(
            "abs_spearman_r",
            "median"
        ),
        max_abs_spearman=(
            "abs_spearman_r",
            "max"
        )
    )
    .reset_index()
)


# ============================================================
# 9. SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

family_summary.to_csv(
    FAMILY_SUMMARY_FILE,
    index=False
)


# ============================================================
# 10. PRINT TOP FEATURES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 20 FEATURES BY ABSOLUTE SPEARMAN ASSOCIATION"
)

print(
    "=" * 70
)

print(
    results_df[
        [
            "overall_rank",
            "feature",
            "feature_family",
            "pearson_r",
            "spearman_r",
            "abs_spearman_r",
            "score_direction"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 11. PRINT FAMILY SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FEATURE FAMILY SUMMARY"
)

print(
    "=" * 70
)

print(
    family_summary.to_string(
        index=False
    )
)


# ============================================================
# 12. VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23D FINAL VALIDATION"
)

print(
    "=" * 70
)

assert len(results_df) == 232

assert set(
    results_df[
        "feature_family"
    ]
) == {
    "original",
    "diff_1s",
    "abs_diff_1s",
    "rolling_std_5s"
}

assert (
    results_df[
        "overall_rank"
    ]
    .min()
    == 1
)

assert (
    results_df[
        "overall_rank"
    ]
    .max()
    == 232
)

assert len(
    family_summary
) == 4

assert Path(
    OUTPUT_FILE
).exists()

assert Path(
    FAMILY_SUMMARY_FILE
).exists()

print(
    "PASS: All 232 features analyzed"
)

print(
    "PASS: Original features identified"
)

print(
    "PASS: Difference features identified"
)

print(
    "PASS: Absolute-difference features identified"
)

print(
    "PASS: Rolling-standard-deviation features identified"
)

print(
    "PASS: Feature/score associations calculated"
)

print(
    "PASS: No model retraining"
)

print(
    "PASS: No threshold tuning"
)

print(
    "PASS: Test labels not used"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    FAMILY_SUMMARY_FILE
)

print(
    "\n" + "=" * 70
)

print(
    "STAGE 23D: COMPLETE"
)

print(
    "=" * 70
)