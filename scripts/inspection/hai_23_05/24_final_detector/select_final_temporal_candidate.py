import os
import pandas as pd

print("=" * 70)
print("STAGE 24A: FINAL TEMPORAL CANDIDATE SELECTION")
print("=" * 70)

BASE = r"data/features/hai/hai-23.05/temporal_representation/reduced_candidates"

INPUT = os.path.join(
    BASE,
    "hai_2305_temporal_candidate_robustness.csv"
)

OUTPUT = os.path.join(
    BASE,
    "hai_2305_final_temporal_candidate_selection.csv"
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("\nLoading Stage 23H robustness results...")

df = pd.read_csv(INPUT)

print(f"Candidates loaded: {len(df)}")

required_columns = [
    "candidate",
    "feature_count",
    "feature_reduction_percent",
    "test1_f1",
    "test2_f1",
    "mean_f1",
    "f1_std",
    "f1_test_difference",
    "mean_precision",
    "mean_recall",
    "mean_f1_delta_vs_full"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("\n" + "=" * 70)
print("CANDIDATE COMPARISON")
print("=" * 70)

display_columns = [
    "candidate",
    "feature_count",
    "feature_reduction_percent",
    "test1_f1",
    "test2_f1",
    "mean_f1",
    "f1_std",
    "mean_precision",
    "mean_recall",
    "mean_f1_delta_vs_full"
]

print(df[display_columns].to_string(index=False))

# ------------------------------------------------------------------
# Identify full temporal reference
# ------------------------------------------------------------------

full = df[df["candidate"] == "full_temporal_reference"]

if len(full) != 1:
    raise ValueError("Expected exactly one full_temporal_reference row.")

full_mean_f1 = full.iloc[0]["mean_f1"]

# ------------------------------------------------------------------
# Non-reference candidates
# ------------------------------------------------------------------

candidates = df[df["candidate"] != "full_temporal_reference"].copy()

# Absolute performance difference from full reference
candidates["absolute_mean_f1_difference"] = (
    candidates["mean_f1"] - full_mean_f1
).abs()

# Rank by closeness to full reference
candidates["closeness_rank"] = (
    candidates["absolute_mean_f1_difference"]
    .rank(method="min", ascending=True)
    .astype(int)
)

# ------------------------------------------------------------------
# Primary candidate
# ------------------------------------------------------------------

selected = candidates.sort_values(
    by=[
        "absolute_mean_f1_difference",
        "f1_std",
        "feature_count"
    ],
    ascending=[
        True,
        True,
        True
    ]
).iloc[0]

selected_name = selected["candidate"]

print("\n" + "=" * 70)
print("SELECTION RESULT")
print("=" * 70)

print(f"Selected candidate       : {selected_name}")
print(f"Feature count            : {int(selected['feature_count'])}")
print(
    f"Feature reduction        : "
    f"{selected['feature_reduction_percent']:.2f}%"
)
print(f"Mean F1                  : {selected['mean_f1']:.6f}")
print(f"F1 standard deviation    : {selected['f1_std']:.6f}")
print(
    f"Mean F1 difference vs full: "
    f"{selected['absolute_mean_f1_difference']:.6f}"
)
print(f"Mean precision           : {selected['mean_precision']:.6f}")
print(f"Mean recall              : {selected['mean_recall']:.6f}")

# ------------------------------------------------------------------
# Create rationale fields
# ------------------------------------------------------------------

candidates["selection_status"] = "NOT_SELECTED"
candidates.loc[
    candidates["candidate"] == selected_name,
    "selection_status"
] = "SELECTED_COMPACT_REFERENCE"

full_row = df[df["candidate"] == "full_temporal_reference"].copy()
full_row["absolute_mean_f1_difference"] = 0.0
full_row["closeness_rank"] = 0
full_row["selection_status"] = "FULL_REFERENCE"

final_df = pd.concat(
    [candidates, full_row],
    ignore_index=True
)

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 24A VALIDATION")
print("=" * 70)

assert len(df) == 6
print("PASS: Six candidates present")

assert selected_name == "candidate_C"
print("PASS: Candidate C identified as closest compact candidate")

assert selected["feature_count"] == 118
print("PASS: Selected feature count verified")

assert abs(
    selected["mean_f1"] - full_mean_f1
) < 0.001
print("PASS: Selected candidate mean F1 is within 0.001 of full reference")

assert selected["feature_reduction_percent"] > 40
print("PASS: Selected candidate provides >40% feature reduction")

assert "full_temporal_reference" in final_df["candidate"].values
print("PASS: Full temporal reference retained")

assert final_df["selection_status"].value_counts().get(
    "SELECTED_COMPACT_REFERENCE", 0
) == 1
print("PASS: Exactly one compact candidate selected")

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

final_df.to_csv(OUTPUT, index=False)

print("\nOutput:")
print(os.path.abspath(OUTPUT))

print("\n" + "=" * 70)
print("STAGE 24A: COMPLETE")
print("=" * 70)