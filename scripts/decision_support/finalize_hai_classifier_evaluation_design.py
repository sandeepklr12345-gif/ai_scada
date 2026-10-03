from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test2_multilabel_classifier_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "hai_classifier_evaluation_design.csv"
)


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 100)
    print("HAI 23.05 FINAL CLASSIFIER EVALUATION DESIGN")
    print("=" * 100)

    # -----------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------

    print("\n[1] Loading validated multi-label dataset")

    data = pd.read_csv(INPUT_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print(
        "Dataset:",
        data.shape
    )

    # -----------------------------------------------------
    # 2. Identify attack-code columns
    # -----------------------------------------------------

    print(
        "\n[2] Identifying attack-code vocabulary"
    )

    target_columns = [
        column
        for column in data.columns
        if column.startswith("AP")
        or column.startswith("AE")
    ]

    if len(target_columns) != 39:
        fail(
            f"Expected 39 attack codes, "
            f"found {len(target_columns)}."
        )

    print(
        "Attack-code vocabulary:",
        len(target_columns)
    )

    # -----------------------------------------------------
    # 3. Build scenario table
    # -----------------------------------------------------

    print(
        "\n[3] Building scenario-level table"
    )

    attack_data = data[
        data["attack_label"] == 1
    ].copy()

    scenario_records = []

    for scenario_id, group in attack_data.groupby(
        "scenario_id"
    ):

        codes = [
            code
            for code in target_columns
            if group[code].sum() > 0
        ]

        scenario_records.append(
            {
                "scenario_id": scenario_id,
                "start": group["timestamp"].min(),
                "end": group["timestamp"].max(),
                "rows": len(group),
                "codes": codes,
                "code_count": len(codes),
                "combination": len(codes) > 1,
            }
        )

    scenarios = (
        pd.DataFrame(scenario_records)
        .sort_values("start")
        .reset_index(drop=True)
    )

    if len(scenarios) != 38:
        fail(
            f"Expected 38 scenarios, "
            f"found {len(scenarios)}."
        )

    # -----------------------------------------------------
    # 4. Determine scenario coverage of every label
    # -----------------------------------------------------

    print(
        "\n[4] Determining attack-code scenario coverage"
    )

    code_scenarios = {}

    for code in target_columns:

        code_scenarios[code] = (
            scenarios.loc[
                scenarios["codes"].apply(
                    lambda codes: code in codes
                ),
                "scenario_id",
            ]
            .tolist()
        )

    unique_code_scenarios = {
        code: ids
        for code, ids in code_scenarios.items()
        if len(ids) == 1
    }

    print(
        "Codes appearing in only one scenario:",
        len(unique_code_scenarios)
    )

    for code, ids in sorted(
        unique_code_scenarios.items()
    ):
        print(
            f"  {code}: {ids}"
        )

    # -----------------------------------------------------
    # 5. Construct a defensible primary classifier track
    #
    # Principle:
    #   Train must contain all 39 labels.
    #
    # Holdout scenarios may only be selected if removing
    # them still leaves every label represented in TRAIN.
    #
    # We deliberately keep combination scenarios in TRAIN.
    # -----------------------------------------------------

    print(
        "\n[5] Constructing primary classifier track"
    )

    # Start with every scenario in training.
    train_scenarios = set(
        scenarios["scenario_id"]
    )

    # Candidate holdouts are scenarios whose every code
    # occurs in another scenario.
    candidate_holdouts = []

    for _, row in scenarios.iterrows():

        scenario_id = row["scenario_id"]

        removable = True

        for code in row["codes"]:

            occurrences = code_scenarios[code]

            if len(occurrences) <= 1:
                removable = False
                break

        if removable:
            candidate_holdouts.append(
                scenario_id
            )

    print(
        "Potential holdout scenarios:",
        candidate_holdouts
    )

    # -----------------------------------------------------
    # 6. Select broad known-mechanism holdouts
    #
    # Prefer holdouts containing codes that occur
    # elsewhere, while preserving all labels in TRAIN.
    #
    # We use a deterministic chronological selection.
    # -----------------------------------------------------

    selected_holdouts = []

    # Prefer single-label scenarios first.
    single_label_candidates = [
        scenario_id
        for scenario_id in candidate_holdouts
        if not scenarios.loc[
            scenarios["scenario_id"] == scenario_id,
            "combination",
        ].iloc[0]
    ]

    for scenario_id in single_label_candidates:

        trial_train = (
            train_scenarios
            - {scenario_id}
        )

        covered_codes = set()

        for sid in trial_train:

            row = scenarios[
                scenarios["scenario_id"] == sid
            ].iloc[0]

            covered_codes.update(
                row["codes"]
            )

        if covered_codes == set(target_columns):

            selected_holdouts.append(
                scenario_id
            )

            train_scenarios = trial_train

    # -----------------------------------------------------
    # 7. Keep combinations in training
    # -----------------------------------------------------

    combination_holdouts = [
        scenario_id
        for scenario_id in selected_holdouts
        if scenarios.loc[
            scenarios["scenario_id"] == scenario_id,
            "combination",
        ].iloc[0]
    ]

    if combination_holdouts:

        fail(
            "A combination scenario was selected "
            "for holdout. Combination scenarios "
            "must remain in TRAIN for this design."
        )

    print(
        "\nPrimary TRAIN scenarios:",
        len(train_scenarios)
    )

    print(
        sorted(train_scenarios)
    )

    print(
        "\nPrimary HOLDOUT scenarios:",
        len(selected_holdouts)
    )

    print(
        sorted(selected_holdouts)
    )

    # -----------------------------------------------------
    # 8. Verify all labels remain in training
    # -----------------------------------------------------

    print(
        "\n[6] Verifying primary training coverage"
    )

    train_codes = set()

    for scenario_id in train_scenarios:

        row = scenarios[
            scenarios["scenario_id"] == scenario_id
        ].iloc[0]

        train_codes.update(
            row["codes"]
        )

    missing_train_codes = (
        set(target_columns)
        - train_codes
    )

    print(
        "Training attack codes:",
        len(train_codes)
    )

    print(
        "Missing:",
        sorted(missing_train_codes)
    )

    if missing_train_codes:
        fail(
            "Primary training track does not "
            "cover all 39 attack codes."
        )

    print(
        "PASS: All 39 attack codes represented in TRAIN."
    )

    # -----------------------------------------------------
    # 9. Verify combinations in training
    # -----------------------------------------------------

    print(
        "\n[7] Verifying combination coverage"
    )

    combination_scenarios = set(
        scenarios.loc[
            scenarios["combination"],
            "scenario_id",
        ]
    )

    train_combinations = (
        combination_scenarios
        & train_scenarios
    )

    print(
        "Total combination scenarios:",
        len(combination_scenarios)
    )

    print(
        "Combination scenarios in TRAIN:",
        len(train_combinations)
    )

    print(
        sorted(train_combinations)
    )

    if train_combinations != combination_scenarios:

        fail(
            "Not all combination scenarios remain "
            "in primary TRAIN."
        )

    print(
        "PASS: All combination scenarios remain in TRAIN."
    )

    # -----------------------------------------------------
    # 10. Define known-mechanism evaluation scenarios
    # -----------------------------------------------------

    print(
        "\n[8] Known-mechanism evaluation"
    )

    known_eval_scenarios = set(
        selected_holdouts
    )

    known_eval_codes = set()

    for scenario_id in known_eval_scenarios:

        row = scenarios[
            scenarios["scenario_id"] == scenario_id
        ].iloc[0]

        known_eval_codes.update(
            row["codes"]
        )

    print(
        "Known-mechanism evaluation scenarios:",
        sorted(known_eval_scenarios)
    )

    print(
        "Attack codes represented:",
        sorted(known_eval_codes)
    )

    unseen_in_train = (
        known_eval_codes
        - train_codes
    )

    if unseen_in_train:

        fail(
            "Known-mechanism evaluation contains "
            "a code absent from training."
        )

    print(
        "PASS: Evaluation mechanisms are known to TRAIN."
    )

    # -----------------------------------------------------
    # 11. Define strict chronological track
    # -----------------------------------------------------

    print(
        "\n[9] Strict chronological evaluation track"
    )

    chronological_train_end = pd.Timestamp(
        "2022-08-18 14:24:00.400000"
    )

    chronological_validation_end = pd.Timestamp(
        "2022-08-19 03:12:00.200000"
    )

    chronological_train = data[
        data["timestamp"]
        < chronological_train_end
    ]

    chronological_validation = data[
        (data["timestamp"] >= chronological_train_end)
        & (
            data["timestamp"]
            < chronological_validation_end
        )
    ]

    chronological_test = data[
        data["timestamp"]
        >= chronological_validation_end
    ]

    chrono_train_codes = set()

    for code in target_columns:

        if chronological_train[code].sum() > 0:
            chrono_train_codes.add(code)

    chrono_validation_codes = set(
        code
        for code in target_columns
        if chronological_validation[code].sum() > 0
    )

    chrono_test_codes = set(
        code
        for code in target_columns
        if chronological_test[code].sum() > 0
    )

    print(
        "Chronological TRAIN codes:",
        len(chrono_train_codes)
    )

    print(
        "Chronological VALIDATION codes:",
        len(chrono_validation_codes)
    )

    print(
        "Chronological TEST codes:",
        len(chrono_test_codes)
    )

    chronological_unseen = (
        chrono_test_codes
        - chrono_train_codes
    )

    print(
        "TEST mechanisms unseen during TRAIN:",
        len(chronological_unseen)
    )

    print(
        sorted(chronological_unseen)
    )

    # -----------------------------------------------------
    # 12. Define final interpretation
    # -----------------------------------------------------

    print(
        "\n[10] Final methodological decision"
    )

    print(
        """
FINAL DESIGN

PRIMARY CLASSIFIER TRACK
------------------------
Purpose:
Evaluate supervised attack-mechanism classification
on complete attack scenarios that were not used during
training.

Rules:
1. Every one of the 39 attack codes occurs in TRAIN.
2. No attack scenario is divided between splits.
3. All combination scenarios remain in TRAIN.
4. Held-out mechanisms are already represented elsewhere
   in TRAIN.
5. Classifier probabilities come only from predict_proba()
   or an explicitly calibrated probability model.

STRICT CHRONOLOGICAL TRACK
--------------------------
Purpose:
Evaluate temporal generalization and identify attack
mechanisms that were not represented in earlier data.

Important:
A mechanism absent from chronological TRAIN cannot be
reported as an ordinary supervised classification result.

Those cases belong to anomaly/OOD analysis.

ANOMALY TRACK
-------------
Candidate C Isolation Forest remains responsible for
anomaly detection.

Its anomaly score is NOT converted into a probability.

DECISION SUPPORT
----------------
Known mechanism:
    anomaly
       ↓
    classifier
       ↓
    attack mechanism + probability

Unseen mechanism:
    anomaly
       ↓
    no learned mechanism claim
       ↓
    anomaly/unseen-mechanism condition

Combination attack:
    multiple attack labels
       ↓
    independent label probabilities
       ↓
    multi-label result

No artificial combined class is created.
"""
    )

    # -----------------------------------------------------
    # 13. Final audit table
    # -----------------------------------------------------

    print(
        "\n[11] Final design audit"
    )

    design = pd.DataFrame(
        [
            {
                "track": "primary_classifier",
                "purpose": "Known-mechanism scenario classification",
                "all_39_labels_in_train": True,
                "scenario_leakage": False,
                "combination_attacks_in_train": True,
                "probability_source": "classifier predict_proba/calibration",
            },
            {
                "track": "chronological",
                "purpose": "Temporal generalization analysis",
                "all_39_labels_in_train": (
                    len(chrono_train_codes) == 39
                ),
                "scenario_leakage": False,
                "combination_attacks_in_train": False,
                "probability_source": "not applicable",
            },
            {
                "track": "anomaly",
                "purpose": "Unseen/anomalous behavior detection",
                "all_39_labels_in_train": False,
                "scenario_leakage": False,
                "combination_attacks_in_train": True,
                "probability_source": "Isolation Forest score only",
            },
        ]
    )

    print(
        design.to_string(index=False)
    )

    # -----------------------------------------------------
    # 14. Save design report
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    design.to_csv(
        REPORT_PATH,
        index=False
    )

    print(
        "\nDesign report:",
        REPORT_PATH
    )

    # -----------------------------------------------------
    # 15. Final status
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("STEP 6.11 PRE-CLASSIFICATION DESIGN COMPLETE")
    print("=" * 100)

    print(
        """
No classifier was trained.

No existing model was modified.

No raw HAI data was modified.

The classifier will be evaluated as a multi-label
attack-mechanism classifier, while Candidate C remains
the anomaly detector.

Probability must come from the classifier, never from
the Isolation Forest anomaly score.
"""
    )


if __name__ == "__main__":
    main()