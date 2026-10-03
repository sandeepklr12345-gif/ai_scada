from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAPPING_INPUT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test1_label_mapping.csv"
)

SUMMARY_INPUT = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
    / "hai_test1_label_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "hai_test1_attack_scenario_mapping.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "hai_test1_attack_class_summary.csv"


# ------------------------------------------------------------------
# HAI 23.05 TEST 1 DOCUMENTED ATTACK SCENARIOS
# ------------------------------------------------------------------
#
# These are the scenarios that actually occur in hai-test1.
#
# A201 is intentionally NOT included because its documented start
# date is Aug. 17, 2022, while hai-test1 ends on Aug. 13, 2022.
#
# ------------------------------------------------------------------

SCENARIOS = [
    {
        "scenario_id": "A101",
        "attack_primitive": "AP01",
        "target_controller": "P1-PC-SP1",
        "target_points": "P1_B2016",
        "start": "2022-08-12 16:25:04",
        "duration_seconds": 237,
    },
    {
        "scenario_id": "A102",
        "attack_primitive": "AP02",
        "target_controller": "P1-PC-SP1PV1",
        "target_points": "P1_B2016, P1_PIT01",
        "start": "2022-08-12 17:35:01",
        "duration_seconds": 198,
    },
    {
        "scenario_id": "A103",
        "attack_primitive": "AP04",
        "target_controller": "P1-PC-CO1",
        "target_points": "P1_PCV01D",
        "start": "2022-08-12 18:32:17",
        "duration_seconds": 156,
    },
    {
        "scenario_id": "A104",
        "attack_primitive": "AP05",
        "target_controller": "P1-PC-CO1PV1",
        "target_points": "P1_PCV01D, P1_PIT01",
        "start": "2022-08-12 19:21:04",
        "duration_seconds": 164,
    },
    {
        "scenario_id": "A105",
        "attack_primitive": "AP07",
        "target_controller": "P1-PC-CO1-ST",
        "target_points": "P1_PCV01D",
        "start": "2022-08-12 20:43:01",
        "duration_seconds": 161,
    },
    {
        "scenario_id": "A106",
        "attack_primitive": "AP03",
        "target_controller": "P1-PC-SP1PV1PV2",
        "target_points": "P1_B2016, P1_PIT01",
        "start": "2022-08-12 21:36:01",
        "duration_seconds": 197,
    },
    {
        "scenario_id": "A107",
        "attack_primitive": "AP40",
        "target_controller": "P1-PC-SP1-LT",
        "target_points": "P1_B2016",
        "start": "2022-08-12 22:47:01",
        "duration_seconds": 604,
    },
    {
        "scenario_id": "A108",
        "attack_primitive": "AP08",
        "target_controller": "P1-FC-SP1",
        "target_points": "P1_B3005",
        "start": "2022-08-12 23:35:15",
        "duration_seconds": 96,
    },
    {
        "scenario_id": "A109",
        "attack_primitive": "AP09",
        "target_controller": "P1-FC-SP1PV1",
        "target_points": "P1_B3005, P1_FT03",
        "start": "2022-08-13 00:25:04",
        "duration_seconds": 130,
    },
    {
        "scenario_id": "A110",
        "attack_primitive": "AP11",
        "target_controller": "P1-FC-CO1",
        "target_points": "P1_FCV03D",
        "start": "2022-08-13 01:34:09",
        "duration_seconds": 55,
    },
    {
        "scenario_id": "A111",
        "attack_primitive": "AP12",
        "target_controller": "P1-FC-CO1PV1",
        "target_points": "P1_FCV03D, P1_FT03",
        "start": "2022-08-13 02:21:04",
        "duration_seconds": 131,
    },
    {
        "scenario_id": "A112",
        "attack_primitive": "AP13",
        "target_controller": "P1-FC-CO1-ST",
        "target_points": "P1_FCV03D",
        "start": "2022-08-13 03:26:03",
        "duration_seconds": 78,
    },
    {
        "scenario_id": "A113",
        "attack_primitive": "AP10",
        "target_controller": "P1-FC-SP1PV1PV2",
        "target_points": "P1_B3005, P1_FT03, P1_LIT01",
        "start": "2022-08-13 04:43:02",
        "duration_seconds": 133,
    },
    {
        "scenario_id": "A114",
        "attack_primitive": "AP41",
        "target_controller": "P1-FC-SP1-LT",
        "target_points": "P1_B3005",
        "start": "2022-08-13 05:40:08",
        "duration_seconds": 627,
    },
]


def main():

    print("=" * 80)
    print("HAI 23.05 TEST 1 ATTACK SCENARIO MAPPING")
    print("=" * 80)

    mapping = pd.read_csv(MAPPING_INPUT)
    summary = pd.read_csv(SUMMARY_INPUT)

    mapping["timestamp"] = pd.to_datetime(mapping["timestamp"])

    print(f"\nInput mapping shape : {mapping.shape}")
    print(f"Input summary shape : {summary.shape}")

    positive = mapping[mapping["label"] == 1].copy()

    print(f"Attack rows found   : {len(positive)}")

    # --------------------------------------------------------------
    # MAP EACH DOCUMENTED SCENARIO
    # --------------------------------------------------------------

    mapped_parts = []

    for scenario in SCENARIOS:

        start = pd.Timestamp(scenario["start"])

        expected_end = (
            start
            + pd.Timedelta(seconds=scenario["duration_seconds"])
        )

        rows = positive[
            (positive["timestamp"] >= start)
            & (positive["timestamp"] <= expected_end)
        ].copy()

        if rows.empty:
            raise ValueError(
                f"No label rows found for {scenario['scenario_id']} "
                f"starting at {start}"
            )

        expected_rows = scenario["duration_seconds"] + 1

        if len(rows) != expected_rows:
            raise ValueError(
                f"{scenario['scenario_id']} row-count mismatch: "
                f"expected {expected_rows}, found {len(rows)}"
            )

        if rows["timestamp"].iloc[0] != start:
            raise ValueError(
                f"{scenario['scenario_id']} start timestamp mismatch."
            )

        if rows["timestamp"].iloc[-1] != expected_end:
            raise ValueError(
                f"{scenario['scenario_id']} end timestamp mismatch."
            )

        rows["scenario_id"] = scenario["scenario_id"]
        rows["attack_primitive"] = scenario["attack_primitive"]
        rows["target_controller"] = scenario["target_controller"]
        rows["target_points"] = scenario["target_points"]

        mapped_parts.append(rows)

        print(
            f"{scenario['scenario_id']} | "
            f"{scenario['attack_primitive']} | "
            f"{len(rows):4d} rows | "
            f"{start} -> {expected_end}"
        )

    mapped_attacks = pd.concat(mapped_parts, ignore_index=True)

    # --------------------------------------------------------------
    # VERIFY COMPLETE ATTACK COVERAGE
    # --------------------------------------------------------------

    if len(mapped_attacks) != len(positive):
        raise ValueError(
            "Mapped attack rows do not equal all positive-label rows."
        )

    if mapped_attacks["timestamp"].duplicated().any():
        raise ValueError(
            "A timestamp was mapped to more than one attack scenario."
        )

    print("\nAttack coverage: PASS")

    # --------------------------------------------------------------
    # SAVE COMPLETE DATASET
    # --------------------------------------------------------------

    normal = mapping[mapping["label"] == 0].copy()

    normal["scenario_id"] = "NORMAL"
    normal["attack_primitive"] = "NORMAL"
    normal["target_controller"] = "NONE"
    normal["target_points"] = "NONE"

    final = pd.concat(
        [normal, mapped_attacks],
        ignore_index=True,
    )

    final = final.sort_values("timestamp").reset_index(drop=True)

    final["classification_target"] = final[
        "attack_primitive"
    ]

    # --------------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------------

    expected_rows = len(mapping)

    if len(final) != expected_rows:
        raise ValueError(
            f"Final row count mismatch: "
            f"expected {expected_rows}, found {len(final)}"
        )

    if final["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found in final dataset."
        )

    expected_classes = (
        ["NORMAL"]
        + [scenario["attack_primitive"] for scenario in SCENARIOS]
    )

    actual_classes = sorted(
        final["classification_target"].unique()
    )

    print("\nClassification classes:")
    for class_name in actual_classes:
        count = (
            final["classification_target"]
            .eq(class_name)
            .sum()
        )
        print(f"{class_name:10s} : {count:6d}")

    # --------------------------------------------------------------
    # CLASS SUMMARY
    # --------------------------------------------------------------

    class_summary = (
        final
        .groupby(
            [
                "classification_target",
                "scenario_id",
                "attack_primitive",
                "target_controller",
                "target_points",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
    )

    class_summary["percentage"] = (
        class_summary["rows"]
        / len(final)
        * 100
    )

    # --------------------------------------------------------------
    # SAVE
    # --------------------------------------------------------------

    final.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    class_summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\nClass summary:")
    print(class_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("SCENARIO MAPPING: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()