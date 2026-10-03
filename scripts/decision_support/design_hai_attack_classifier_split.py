from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

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

    print("=" * 90)
    print("HAI 23.05 ATTACK CLASSIFIER TEMPORAL SPLIT ANALYSIS")
    print("=" * 90)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print("\n[1] Loading multi-label dataset...")

    data = pd.read_csv(INPUT_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    print(
        "Dataset shape:",
        data.shape
    )

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    print("\n[2] Basic dataset validation")

    if data["timestamp"].duplicated().any():
        fail(
            "Duplicate timestamps detected."
        )

    if not data["timestamp"].is_monotonic_increasing:
        print(
            "WARNING: timestamps are not sorted. "
            "Sorting for analysis."
        )

        data = data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    print(
        "Start:",
        data["timestamp"].min()
    )

    print(
        "End:",
        data["timestamp"].max()
    )

    print(
        "Rows:",
        len(data)
    )

    print(
        "PASS: Timestamp structure validated."
    )

    # -----------------------------------------------------
    # Identify attack and normal rows
    # -----------------------------------------------------

    print("\n[3] Attack / normal temporal distribution")

    attack = data[
        data["attack_label"] == 1
    ].copy()

    normal = data[
        data["attack_label"] == 0
    ].copy()

    print(
        "Normal rows:",
        len(normal)
    )

    print(
        "Attack rows:",
        len(attack)
    )

    print(
        "Attack start:",
        attack["timestamp"].min()
    )

    print(
        "Attack end:",
        attack["timestamp"].max()
    )

    # -----------------------------------------------------
    # Scenario-level temporal blocks
    # -----------------------------------------------------

    print("\n[4] Scenario temporal blocks")

    scenario_blocks = (
        attack
        .groupby("scenario_id")
        .agg(
            start_timestamp=(
                "timestamp",
                "min",
            ),
            end_timestamp=(
                "timestamp",
                "max",
            ),
            rows=(
                "timestamp",
                "size",
            ),
            attack_codes=(
                "attack_codes",
                "first",
            ),
        )
        .reset_index()
        .sort_values("start_timestamp")
    )

    scenario_blocks[
        "duration_minutes"
    ] = (
        (
            scenario_blocks["end_timestamp"]
            - scenario_blocks["start_timestamp"]
        )
        / pd.Timedelta(minutes=1)
    ) + 1

    print(
        scenario_blocks.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Check that each scenario forms a contiguous block
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Check scenario continuity at minute level
    # -----------------------------------------------------

    print(
        "\n[5] Scenario continuity analysis"
    )

    continuity_failures = []

    for scenario_id, group in attack.groupby(
        "scenario_id"
    ):

        minute_timestamps = (
            group["timestamp"]
            .dt.floor("min")
            .drop_duplicates()
            .sort_values()
        )

        differences = (
            minute_timestamps.diff()
            .dropna()
        )

        invalid_intervals = differences[
            differences
            != pd.Timedelta(minutes=1)
        ]

        if not invalid_intervals.empty:

            continuity_failures.append(
                (
                    scenario_id,
                    len(invalid_intervals),
                )
            )

    if continuity_failures:

        print(
            "Scenarios with non-contiguous minute blocks:"
        )

        for scenario_id, count in continuity_failures:
            print(
                f"  {scenario_id}: "
                f"{count} invalid minute intervals"
            )

        fail(
            "One or more attack scenarios are not "
            "temporally contiguous at minute level."
        )

    print(
        "PASS: All 38 attack scenarios form "
        "contiguous minute-level blocks."
    )
    # -----------------------------------------------------
    # Scenario gap analysis
    # -----------------------------------------------------

    print("\n[6] Scenario separation analysis")

    scenario_blocks["previous_end"] = (
        scenario_blocks["end_timestamp"]
        .shift(1)
    )

    scenario_blocks["gap_minutes"] = (
        (
            scenario_blocks["start_timestamp"]
            - scenario_blocks["previous_end"]
        )
        / pd.Timedelta(minutes=1)
    ) - 1

    print(
        scenario_blocks[
            [
                "scenario_id",
                "start_timestamp",
                "end_timestamp",
                "gap_minutes",
            ]
        ].to_string(index=False)
    )

    minimum_gap = (
        scenario_blocks["gap_minutes"]
        .iloc[1:]
        .min()
    )

    print(
        "\nMinimum gap between attack scenarios:",
        minimum_gap,
        "minutes"
    )

    # -----------------------------------------------------
    # Candidate chronological splits
    #
    # These are analysis candidates only.
    # No split is saved or used for training.
    # -----------------------------------------------------

    print("\n[7] Candidate chronological splits")

    start_time = data["timestamp"].min()
    end_time = data["timestamp"].max()

    total_duration = (
        end_time - start_time
    )

    print(
        "Total time span:",
        total_duration
    )

    # 60 / 20 / 20 chronological split
    train_end = (
        start_time
        + total_duration * 0.60
    )

    validation_end = (
        start_time
        + total_duration * 0.80
    )

    print("\nCandidate A: 60 / 20 / 20")

    print(
        "Training:",
        start_time,
        "→",
        train_end
    )

    print(
        "Validation:",
        train_end,
        "→",
        validation_end
    )

    print(
        "Test:",
        validation_end,
        "→",
        end_time
    )

    # -----------------------------------------------------
    # Count attack scenarios in candidate splits
    # -----------------------------------------------------

    def summarize_split(
        name,
        start,
        end,
    ):

        mask = (
            data["timestamp"] >= start
        ) & (
            data["timestamp"] < end
        )

        subset = data.loc[mask]

        attack_subset = subset[
            subset["attack_label"] == 1
        ]

        scenarios = sorted(
            attack_subset[
                "scenario_id"
            ]
            .unique()
            .tolist()
        )

        print(
            f"\n{name}"
        )

        print(
            "Rows:",
            len(subset)
        )

        print(
            "Attack rows:",
            len(attack_subset)
        )

        print(
            "Attack scenarios:",
            len(scenarios)
        )

        print(
            scenarios
        )

        return set(scenarios)

    train_scenarios = summarize_split(
        "TRAIN",
        start_time,
        train_end,
    )

    validation_scenarios = summarize_split(
        "VALIDATION",
        train_end,
        validation_end,
    )

    test_scenarios = summarize_split(
        "TEST",
        validation_end,
        end_time
        + pd.Timedelta(minutes=1),
    )

    # -----------------------------------------------------
    # Check scenario leakage
    # -----------------------------------------------------

    print("\n[8] Scenario leakage analysis")

    train_validation_overlap = (
        train_scenarios
        & validation_scenarios
    )

    train_test_overlap = (
        train_scenarios
        & test_scenarios
    )

    validation_test_overlap = (
        validation_scenarios
        & test_scenarios
    )

    print(
        "Train / Validation overlap:",
        train_validation_overlap
    )

    print(
        "Train / Test overlap:",
        train_test_overlap
    )

    print(
        "Validation / Test overlap:",
        validation_test_overlap
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):

        print(
            "\nWARNING: Candidate chronological split "
            "contains scenario overlap."
        )

    else:

        print(
            "PASS: No scenario crosses the "
            "candidate split boundaries."
        )

    # -----------------------------------------------------
    # Find scenarios touching candidate boundaries
    # -----------------------------------------------------

    print(
        "\n[9] Boundary-crossing scenario analysis"
    )

    candidate_boundaries = {
        "train_validation": train_end,
        "validation_test": validation_end,
    }

    for boundary_name, boundary in (
        candidate_boundaries.items()
    ):

        crossing = scenario_blocks[
            (
                scenario_blocks["start_timestamp"]
                < boundary
            )
            & (
                scenario_blocks["end_timestamp"]
                >= boundary
            )
        ]

        print(
            f"\n{boundary_name} boundary:",
            boundary
        )

        if crossing.empty:

            print(
                "No attack scenario crosses boundary."
            )

        else:

            print(
                "Scenarios crossing boundary:"
            )

            print(
                crossing[
                    [
                        "scenario_id",
                        "start_timestamp",
                        "end_timestamp",
                    ]
                ].to_string(index=False)
            )

    # -----------------------------------------------------
    # Evaluate attack coverage
    # -----------------------------------------------------

    print(
        "\n[10] Attack coverage analysis"
    )

    print(
        "\nTotal attack scenarios:",
        len(scenario_blocks)
    )

    print(
        "Training scenarios:",
        len(train_scenarios)
    )

    print(
        "Validation scenarios:",
        len(validation_scenarios)
    )

    print(
        "Test scenarios:",
        len(test_scenarios)
    )

    # -----------------------------------------------------
    # Event-aware split recommendation
    #
    # We do not automatically select a split.
    # -----------------------------------------------------

    print(
        "\n[11] Split-design conclusion"
    )

    print(
        """
The dataset contains contiguous attack events.
Therefore, a random row-level train/test split
should NOT be used as the primary evaluation design.

A chronological/event-aware split is preferred.

However, the simple 60/20/20 chronological split
must be reviewed based on its attack-scenario coverage
before being selected for classifier training.
"""
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("=" * 90)
    print("TEMPORAL SPLIT ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()