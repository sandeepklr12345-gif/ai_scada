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
    / "hai_test2_attack_classifier_temporal_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "splits"
    / "temporal_gpu"
)

TRAIN_PATH = OUTPUT_DIR / "train.csv"
EVAL_PATH = OUTPUT_DIR / "evaluation.csv"

GUARD_MINUTES = 5


TRAIN_SCENARIOS = [
    "A202", "A204", "A205", "A206", "A207",
    "A208", "A210", "A211", "A212", "A215",
    "A216", "A217", "A218", "A219", "A220",
    "A221", "A222", "A223", "A224", "A225",
    "A226", "A227", "A228", "A229", "A230",
    "A231", "A232", "A233", "A234", "A237",
    "A238",
]

EVAL_SCENARIOS = [
    "A201",
    "A203",
    "A209",
    "A213",
    "A214",
    "A235",
    "A236",
]


def fail(message):
    raise ValueError(message)


def main():

    print("=" * 100)
    print("HAI 23.05 TEMPORAL GPU EVENT-LEVEL SPLIT")
    print("=" * 100)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("\n[1] Loading temporal dataset")

    data = pd.read_csv(INPUT_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print("Dataset shape:", data.shape)

    # ---------------------------------------------------------
    # 2. Validate scenario sets
    # ---------------------------------------------------------

    print("\n[2] Validating scenarios")

    attack_scenarios = set(
        data.loc[
            data["attack_label"] == 1,
            "scenario_id"
        ].dropna()
    )

    train_set = set(TRAIN_SCENARIOS)
    eval_set = set(EVAL_SCENARIOS)

    if train_set & eval_set:
        fail("Training/evaluation scenario overlap detected.")

    missing_train = train_set - attack_scenarios
    missing_eval = eval_set - attack_scenarios

    if missing_train:
        fail(
            f"Training scenarios missing: {sorted(missing_train)}"
        )

    if missing_eval:
        fail(
            f"Evaluation scenarios missing: {sorted(missing_eval)}"
        )

    print(
        "Training scenarios:",
        len(train_set)
    )

    print(
        "Evaluation scenarios:",
        len(eval_set)
    )

    # ---------------------------------------------------------
    # 3. Build evaluation attack rows
    # ---------------------------------------------------------

    print("\n[3] Selecting held-out attack events")

    eval_attack = data[
        data["scenario_id"].isin(
            EVAL_SCENARIOS
        )
    ].copy()

    if eval_attack.empty:
        fail("No evaluation attack rows found.")

    # ---------------------------------------------------------
    # 4. Build ±5 minute normal guard windows
    # ---------------------------------------------------------

    print(
        "\n[4] Building normal guard windows"
    )

    guard_windows = []

    for scenario in EVAL_SCENARIOS:

        scenario_rows = eval_attack[
            eval_attack["scenario_id"] == scenario
        ]

        start = (
            scenario_rows["timestamp"].min()
            - pd.Timedelta(
                minutes=GUARD_MINUTES
            )
        )

        end = (
            scenario_rows["timestamp"].max()
            + pd.Timedelta(
                minutes=GUARD_MINUTES
            )
        )

        guard_windows.append(
            {
                "scenario": scenario,
                "start": start,
                "end": end,
            }
        )

        print(
            f"{scenario}: "
            f"{start} -> {end}"
        )

    # ---------------------------------------------------------
    # 5. Construct evaluation mask
    # ---------------------------------------------------------

    print(
        "\n[5] Constructing event-level evaluation set"
    )

    evaluation_mask = (
        data["scenario_id"].isin(
            EVAL_SCENARIOS
        )
    )

    normal_mask = (
        data["attack_label"] == 0
    )

    normal_guard_mask = pd.Series(
        False,
        index=data.index
    )

    for window in guard_windows:

        normal_guard_mask |= (
            normal_mask
            &
            (data["timestamp"] >= window["start"])
            &
            (data["timestamp"] <= window["end"])
        )

    evaluation_mask |= normal_guard_mask

    evaluation = data[
        evaluation_mask
    ].copy()

    train = data[
        ~evaluation_mask
    ].copy()

    # ---------------------------------------------------------
    # 6. Validate no held-out attack enters training
    # ---------------------------------------------------------

    print(
        "\n[6] Validating scenario separation"
    )

    train_attack_scenarios = set(
        train.loc[
            train["attack_label"] == 1,
            "scenario_id"
        ].dropna()
    )

    eval_attack_scenarios = set(
        evaluation.loc[
            evaluation["attack_label"] == 1,
            "scenario_id"
        ].dropna()
    )

    overlap = (
        train_attack_scenarios
        &
        eval_attack_scenarios
    )

    print(
        "Training attack scenarios:",
        sorted(train_attack_scenarios)
    )

    print(
        "Evaluation attack scenarios:",
        sorted(eval_attack_scenarios)
    )

    print(
        "Scenario overlap:",
        overlap
    )

    if overlap:
        fail(
            "Scenario leakage detected."
        )

    if eval_attack_scenarios != eval_set:
        fail(
            "Evaluation scenario set does not match "
            "the approved seven held-out scenarios."
        )

    print(
        "PASS: No held-out scenario leakage."
    )

    # ---------------------------------------------------------
    # 7. Validate training scenario coverage
    # ---------------------------------------------------------

    print(
        "\n[7] Validating training scenario coverage"
    )

    if not train_set.issubset(
        train_attack_scenarios
    ):
        missing = (
            train_set
            - train_attack_scenarios
        )

        fail(
            f"Training scenarios missing after split: "
            f"{sorted(missing)}"
        )

    print(
        "PASS: All 31 training scenarios retained."
    )

    # ---------------------------------------------------------
    # 8. Check label coverage
    # ---------------------------------------------------------

    print(
        "\n[8] Checking label coverage"
    )

    target_columns = [
        column
        for column in data.columns
        if column.startswith("AP")
        or column.startswith("AE")
    ]

    if len(target_columns) != 39:
        fail(
            f"Expected 39 targets, found {len(target_columns)}."
        )

    missing_training_labels = [
        code
        for code in target_columns
        if train[code].sum() == 0
    ]

    if missing_training_labels:
        fail(
            "Training split is missing labels: "
            + str(missing_training_labels)
        )

    print(
        "PASS: All 39 attack mechanisms represented in training."
    )

    # ---------------------------------------------------------
    # 9. Combination coverage
    # ---------------------------------------------------------

    train_combinations = int(
        (
            train[target_columns].sum(axis=1) > 1
        ).sum()
    )

    eval_combinations = int(
        (
            evaluation[target_columns].sum(axis=1) > 1
        ).sum()
    )

    print(
        "\nTraining combination rows:",
        train_combinations
    )

    print(
        "Evaluation combination rows:",
        eval_combinations
    )

    if train_combinations == 0:
        fail(
            "No combination attacks in training."
        )

    # ---------------------------------------------------------
    # 10. Dataset sizes
    # ---------------------------------------------------------

    print(
        "\n[10] Final split sizes"
    )

    print(
        "Training:",
        train.shape
    )

    print(
        "Evaluation:",
        evaluation.shape
    )

    print(
        "Training attack rows:",
        int(
            (train["attack_label"] == 1).sum()
        )
    )

    print(
        "Training normal rows:",
        int(
            (train["attack_label"] == 0).sum()
        )
    )

    print(
        "Evaluation attack rows:",
        int(
            (evaluation["attack_label"] == 1).sum()
        )
    )

    print(
        "Evaluation normal rows:",
        int(
            (evaluation["attack_label"] == 0).sum()
        )
    )

    # ---------------------------------------------------------
    # 11. Save
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_PATH,
        index=False
    )

    evaluation.to_csv(
        EVAL_PATH,
        index=False
    )

    print(
        "\nSaved training split:",
        TRAIN_PATH
    )

    print(
        "Saved evaluation split:",
        EVAL_PATH
    )

    print("\n" + "=" * 100)
    print("STEP 6.16A COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()