import os
import sys
import pandas as pd


BASE_DIR = r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"

SCRIPT_DIR = os.path.join(
    BASE_DIR,
    "scripts",
    "inspection",
    "hai_23_05",
    "25_inference"
)

MANIFEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "hai",
    "hai-23.05",
    "temporal_representation",
    "final_candidate",
    "hai_2305_candidate_C_feature_manifest.csv"
)


print("=" * 70)
print("STAGE 25E: CANDIDATE C ENGINE SCHEMA VALIDATION")
print("=" * 70)


# ============================================================
# IMPORT ENGINE
# ============================================================

print("\nLoading inference engine...")

sys.path.insert(0, SCRIPT_DIR)

from candidate_C_inference_engine import CandidateCInferenceEngine

print("PASS: Engine imported")


# ============================================================
# LOAD AUTHORITATIVE MANIFEST
# ============================================================

print("\nLoading authoritative Candidate C manifest...")

manifest = pd.read_csv(MANIFEST_PATH)

print(
    f"Manifest rows: {len(manifest)}"
)

print(
    f"Manifest columns: {list(manifest.columns)}"
)


# ============================================================
# DISPLAY MANIFEST STRUCTURE
# ============================================================

print("\nManifest preview:")

print(
    manifest.head(10).to_string(index=False)
)


# ============================================================
# DETERMINE FEATURE COLUMN
# ============================================================

possible_feature_columns = [
    "feature",
    "feature_name",
    "Feature",
    "Feature_Name",
    "name"
]

feature_column = None

for column in possible_feature_columns:

    if column in manifest.columns:
        feature_column = column
        break


if feature_column is None:

    raise ValueError(
        "Could not identify the feature-name column "
        "in the Candidate C manifest."
    )


print(
    f"\nUsing manifest feature column: {feature_column}"
)


manifest_features = (
    manifest[feature_column]
    .astype(str)
    .tolist()
)


# ============================================================
# CREATE ENGINE INSTANCE
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "hai",
    "hai-23.05",
    "temporal_representation",
    "final_candidate",
    "hai_2305_candidate_C_isolation_forest.joblib"
)

print("\nCreating Candidate C engine...")

engine = CandidateCInferenceEngine(
    MODEL_PATH,
    MANIFEST_PATH
)

print("PASS: Engine initialized")


# ============================================================
# ENGINE FEATURE LIST
# ============================================================

engine_features = engine.feature_names


print(
    f"\nEngine feature count: {len(engine_features)}"
)

print(
    f"Manifest feature count: {len(manifest_features)}"
)


# ============================================================
# COUNT VALIDATION
# ============================================================

assert len(manifest_features) == 118

print(
    "PASS: Manifest contains 118 features"
)

assert len(engine_features) == 118

print(
    "PASS: Engine contains 118 features"
)


# ============================================================
# ORDERED COMPARISON
# ============================================================

print("\nComparing feature order...")

ordered_match = (
    engine_features == manifest_features
)

if ordered_match:

    print(
        "PASS: Feature order matches exactly"
    )

else:

    print(
        "FAIL: Feature order does NOT match"
    )

    for i, (engine_feature, manifest_feature) in enumerate(
        zip(engine_features, manifest_features)
    ):

        if engine_feature != manifest_feature:

            print(
                f"\nFirst mismatch at position {i}"
            )

            print(
                f"Engine   : {engine_feature}"
            )

            print(
                f"Manifest : {manifest_feature}"
            )

            break

    raise AssertionError(
        "Candidate C engine feature order does not "
        "match the authoritative manifest."
    )


# ============================================================
# SET COMPARISON
# ============================================================

print("\nChecking feature membership...")

engine_set = set(engine_features)
manifest_set = set(manifest_features)

missing_from_engine = (
    manifest_set - engine_set
)

extra_in_engine = (
    engine_set - manifest_set
)


assert not missing_from_engine

assert not extra_in_engine

print(
    "PASS: No manifest features missing from engine"
)

print(
    "PASS: No extra engine features"
)


# ============================================================
# FEATURE GROUP COUNTS
# ============================================================

original = [
    f for f in engine_features
    if "__" not in f
]

abs_diff = [
    f for f in engine_features
    if "__abs_diff_1s" in f
]

rolling = [
    f for f in engine_features
    if "__rolling_std_5s" in f
]


print("\nFeature composition:")

print(
    f"Original features      : {len(original)}"
)

print(
    f"Absolute differences   : {len(abs_diff)}"
)

print(
    f"Rolling standard dev.  : {len(rolling)}"
)

print(
    f"Total                   : {len(engine_features)}"
)


assert len(original) == 58

assert len(abs_diff) == 30

assert len(rolling) == 30

assert (
    len(original)
    + len(abs_diff)
    + len(rolling)
    == 118
)


print(
    "PASS: Feature composition is 58 + 30 + 30 = 118"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STAGE 25E VALIDATION")
print("=" * 70)

print(
    "PASS: Inference engine imported"
)

print(
    "PASS: Model loaded"
)

print(
    "PASS: 118 features"
)

print(
    "PASS: 58 original features"
)

print(
    "PASS: 30 abs_diff_1s features"
)

print(
    "PASS: 30 rolling_std_5s features"
)

print(
    "PASS: Feature membership matches manifest"
)

print(
    "PASS: Feature ordering matches manifest"
)

print(
    "PASS: Candidate C engine schema is authoritative"
)

print("\n" + "=" * 70)
print("STAGE 25E: COMPLETE")
print("=" * 70)