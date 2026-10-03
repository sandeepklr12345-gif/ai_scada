from pathlib import Path
import pandas as pd
import numpy as np

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix


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


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 100)
    print("HAI 23.05 EVENT-AWARE CLASSIFIER COVERAGE ANALYSIS")
    print("=" * 100)

    # -----------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------

    print("\n[1] Loading multi-label dataset")

    data = pd.read_csv(INPUT_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print(
        "Dataset shape:",
        data.shape
    )

    # -----------------------------------------------------
    # 2. Identify attack target columns
    # -----------------------------------------------------

    print("\n[2] Identifying attack-code targets")

    target_columns = [
        column
        for column in data.columns
        if column.startswith("AP")
        or column.startswith("AE")
    ]

    print(
        "Attack-code targets:",
        len(target_columns)
    )

    if len(target_columns) != 39:
        fail(
            f"Expected 39 attack codes, "
            f"found {len(target_columns)}."
        )

    # -----------------------------------------------------
    # 3. Build scenario-level representation
    # -----------------------------------------------------

    print(
        "\n[3] Building scenario-level representation"
    )

    attack_data = data[
        data["attack_label"] == 1
    ].copy()

    scenario_rows = []

    for scenario_id, group in attack_data.groupby(
        "scenario_id"
    ):

        active_codes = [
            code
            for code in target_columns
            if group[code].sum() > 0
        ]

        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "start_timestamp": group[
                    "timestamp"
                ].min(),
                "end_timestamp": group[
                    "timestamp"
                ].max(),
                "rows": len(group),
                "attack_codes": active_codes,
                "code_count": len(active_codes),
            }
        )

    scenarios = pd.DataFrame(
        scenario_rows
    ).sort_values(
        "start_timestamp"
    ).reset_index(drop=True)

    print(
        "Scenarios:",
        len(scenarios)
    )

    if len(scenarios) != 38:
        fail(
            f"Expected 38 scenarios, "
            f"found {len(scenarios)}."
        )

    print(
        scenarios[
            [
                "scenario_id",
                "start_timestamp",
                "end_timestamp",
                "rows",
                "attack_codes",
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # 4. Build scenario/code incidence matrix
    # -----------------------------------------------------

    print(
        "\n[4] Building scenario/code incidence matrix"
    )

    scenario_ids = scenarios[
        "scenario_id"
    ].tolist()

    matrix = np.zeros(
        (
            len(scenario_ids),
            len(target_columns),
        ),
        dtype=int,
    )

    for i, scenario in scenarios.iterrows():

        for code in scenario["attack_codes"]:

            j = target_columns.index(code)

            matrix[i, j] = 1

    incidence = pd.DataFrame(
        matrix,
        index=scenario_ids,
        columns=target_columns,
    )

    print(
        "Matrix shape:",
        incidence.shape
    )

    # -----------------------------------------------------
    # 5. Determine minimum scenario coverage
    # -----------------------------------------------------

    print(
        "\n[5] Checking which scenarios are mandatory"
    )

    mandatory_scenarios = []

    for code in target_columns:

        matching = incidence.index[
            incidence[code] > 0
        ].tolist()

        print(
            f"{code}: {matching}"
        )

        if len(matching) == 1:

            mandatory_scenarios.extend(
                matching
            )

    mandatory_scenarios = sorted(
        set(mandatory_scenarios)
    )

    print(
        "\nScenarios containing labels that "
        "occur nowhere else:"
    )

    print(
        mandatory_scenarios
    )

    print(
        "Mandatory scenario count:",
        len(mandatory_scenarios)
    )

    # -----------------------------------------------------
    # 6. Solve minimum training scenario set
    #
    # Every attack code must occur at least once
    # in TRAIN.
    #
    # No rows are split inside a scenario.
    # -----------------------------------------------------

    print(
        "\n[6] Solving minimum event-aware training coverage"
    )

    num_scenarios = len(scenario_ids)

    # Objective:
    # minimize number of scenarios assigned to training.
    objective = np.ones(
        num_scenarios,
        dtype=float,
    )

    # Every attack code must have at least one
    # training scenario.
    constraints_matrix = matrix.T

    lower_bounds = np.ones(
        len(target_columns),
        dtype=float,
    )

    upper_bounds = np.full(
        len(target_columns),
        np.inf,
        dtype=float,
    )

    constraints = LinearConstraint(
        csr_matrix(constraints_matrix),
        lower_bounds,
        upper_bounds,
    )

    result = milp(
        c=objective,
        integrality=np.ones(
            num_scenarios
        ),
        bounds=Bounds(
            np.zeros(num_scenarios),
            np.ones(num_scenarios),
        ),
        constraints=constraints,
        options={
            "time_limit": 30,
        },
    )

    if not result.success:

        fail(
            "Could not find a training scenario set "
            "covering all 39 attack codes."
        )

    selected_indices = [
        i
        for i, value in enumerate(result.x)
        if value > 0.5
    ]

    minimum_training_scenarios = [
        scenario_ids[i]
        for i in selected_indices
    ]

    print(
        "\nMinimum training scenario count:",
        len(minimum_training_scenarios)
    )

    print(
        "Candidate minimum TRAIN scenarios:"
    )

    print(
        sorted(minimum_training_scenarios)
    )

    # -----------------------------------------------------
    # 7. Verify complete training-code coverage
    # -----------------------------------------------------

    print(
        "\n[7] Verifying training-code coverage"
    )

    training_incidence = incidence.loc[
        minimum_training_scenarios
    ]

    learned_codes = set(
        training_incidence.columns[
            training_incidence.sum(axis=0) > 0
        ]
    )

    missing_codes = (
        set(target_columns)
        - learned_codes
    )

    print(
        "Learned attack codes:",
        len(learned_codes)
    )

    print(
        "Missing attack codes:",
        sorted(missing_codes)
    )

    if missing_codes:
        fail(
            "Minimum scenario solution does not "
            "cover all attack codes."
        )

    print(
        "PASS: All 39 attack codes are represented "
        "in candidate TRAIN."
    )

    # -----------------------------------------------------
    # 8. Remaining scenarios
    # -----------------------------------------------------

    print(
        "\n[8] Remaining evaluation scenarios"
    )

    remaining_scenarios = [
        scenario
        for scenario in scenario_ids
        if scenario not in minimum_training_scenarios
    ]

    print(
        "Remaining scenarios:",
        len(remaining_scenarios)
    )

    print(
        sorted(remaining_scenarios)
    )

    # -----------------------------------------------------
    # 9. Evaluate possible validation/test allocation
    #
    # We want completely separate scenarios.
    #
    # Use approximately half of remaining scenarios
    # for validation and half for test.
    # -----------------------------------------------------

    print(
        "\n[9] Evaluation scenario allocation"
    )

    remaining_sorted = (
        scenarios[
            scenarios["scenario_id"].isin(
                remaining_scenarios
            )
        ]
        .sort_values("start_timestamp")
        .reset_index(drop=True)
    )

    midpoint = len(remaining_sorted) // 2

    validation_scenarios = (
        remaining_sorted
        .iloc[:midpoint]["scenario_id"]
        .tolist()
    )

    test_scenarios = (
        remaining_sorted
        .iloc[midpoint:]["scenario_id"]
        .tolist()
    )

    print(
        "Candidate VALIDATION scenarios:"
    )

    print(
        validation_scenarios
    )

    print(
        "\nCandidate TEST scenarios:"
    )

    print(
        test_scenarios
    )

    # -----------------------------------------------------
    # 10. Scenario overlap
    # -----------------------------------------------------

    print(
        "\n[10] Scenario overlap validation"
    )

    train_set = set(
        minimum_training_scenarios
    )

    validation_set = set(
        validation_scenarios
    )

    test_set = set(
        test_scenarios
    )

    print(
        "TRAIN / VALIDATION:",
        train_set & validation_set
    )

    print(
        "TRAIN / TEST:",
        train_set & test_set
    )

    print(
        "VALIDATION / TEST:",
        validation_set & test_set
    )

    if (
        train_set & validation_set
        or train_set & test_set
        or validation_set & test_set
    ):

        fail(
            "Scenario overlap detected."
        )

    print(
        "PASS: No scenario overlap."
    )

    # -----------------------------------------------------
    # 11. Evaluation label coverage
    # -----------------------------------------------------

    print(
        "\n[11] Evaluation label coverage"
    )

    validation_incidence = incidence.loc[
        validation_scenarios
    ]

    test_incidence = incidence.loc[
        test_scenarios
    ]

    validation_codes = set(
        validation_incidence.columns[
            validation_incidence.sum(axis=0) > 0
        ]
    )

    test_codes = set(
        test_incidence.columns[
            test_incidence.sum(axis=0) > 0
        ]
    )

    print(
        "Validation attack codes:",
        len(validation_codes)
    )

    print(
        sorted(validation_codes)
    )

    print(
        "\nTest attack codes:",
        len(test_codes)
    )

    print(
        sorted(test_codes)
    )

    print(
        "\nTest codes unseen during TRAIN:"
    )

    print(
        sorted(
            test_codes
            - learned_codes
        )
    )

    # -----------------------------------------------------
    # 12. Combination attack coverage
    # -----------------------------------------------------

    print(
        "\n[12] Combination scenario coverage"
    )

    combination_scenarios = []

    for _, row in scenarios.iterrows():

        if len(row["attack_codes"]) >= 2:

            combination_scenarios.append(
                row["scenario_id"]
            )

    train_combinations = sorted(
        set(combination_scenarios)
        & train_set
    )

    validation_combinations = sorted(
        set(combination_scenarios)
        & validation_set
    )

    test_combinations = sorted(
        set(combination_scenarios)
        & test_set
    )

    print(
        "TRAIN combinations:",
        train_combinations
    )

    print(
        "VALIDATION combinations:",
        validation_combinations
    )

    print(
        "TEST combinations:",
        test_combinations
    )

    # -----------------------------------------------------
    # 13. Row counts
    # -----------------------------------------------------

    print(
        "\n[13] Approximate row distribution"
    )

    def rows_for_scenarios(
        scenario_list
    ):

        if not scenario_list:
            return 0

        return int(
            scenarios[
                scenarios["scenario_id"].isin(
                    scenario_list
                )
            ]["rows"].sum()
        )

    train_attack_rows = rows_for_scenarios(
        minimum_training_scenarios
    )

    validation_attack_rows = rows_for_scenarios(
        validation_scenarios
    )

    test_attack_rows = rows_for_scenarios(
        test_scenarios
    )

    print(
        "TRAIN attack rows:",
        train_attack_rows
    )

    print(
        "VALIDATION attack rows:",
        validation_attack_rows
    )

    print(
        "TEST attack rows:",
        test_attack_rows
    )

    # -----------------------------------------------------
    # 14. Final interpretation
    # -----------------------------------------------------

    print(
        "\n[14] Interpretation"
    )

    print(
        """
This is an analysis-only result.

The optimizer identifies the smallest set of complete
attack scenarios needed for TRAIN to contain every
documented attack-code label.

No individual scenario is divided between splits.

The resulting evaluation scenarios remain completely
separate from TRAIN.

However, if a test attack code is also present in TRAIN,
the test measures generalization to a new scenario of a
known attack mechanism.

If a test attack code is absent from TRAIN, that attack
mechanism is genuinely unseen and cannot be evaluated
as ordinary supervised classification.
"""
    )

    print("=" * 100)
    print("EVENT-AWARE CLASSIFIER COVERAGE ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()