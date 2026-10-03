from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST2_SCADA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
    / "hai-test2.csv"
)

TEST2_LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
    / "label-test2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "hai"
    / "hai-23.05"
    / "fault_classification"
)

OUTPUT_PATH = OUTPUT_DIR / "hai_test2_classifier_targets.csv"


# ---------------------------------------------------------
# Official Test 2 scenario catalog
# ---------------------------------------------------------

TEST2_SCENARIOS = [
    ("A201", ["AP14"], "2022-08-17 01:27:00", 132),
    ("A202", ["AP15"], "2022-08-17 03:37:00", 131),
    ("A203", ["AP16"], "2022-08-17 04:21:00", 68),
    ("A204", ["AP17"], "2022-08-17 05:46:00", 122),
    ("A205", ["AP18"], "2022-08-17 06:21:00", 85),
    ("A206", ["AP03"], "2022-08-17 08:36:00", 196),
    ("A207", ["AP43"], "2022-08-17 09:42:00", 614),
    ("A208", ["AP42"], "2022-08-17 10:36:00", 133),
    ("A209", ["AP23"], "2022-08-17 11:35:00", 85),
    ("A210", ["AP23"], "2022-08-17 12:25:00", 88),
    ("A211", ["AP46"], "2022-08-17 13:47:00", 204),
    ("A212", ["AP24"], "2022-08-17 14:25:00", 127),
    ("A213", ["AP25"], "2022-08-17 15:13:00", 539),
    ("A214", ["AP19"], "2022-08-17 17:34:00", 61),
    ("A215", ["AP20"], "2022-08-17 18:16:00", 147),
    ("A216", ["AP21"], "2022-08-17 19:40:00", 95),
    ("A217", ["AP22"], "2022-08-17 20:12:00", 505),
    ("A218", ["AP07"], "2022-08-17 22:41:00", 214),
    ("A219", ["AP13"], "2022-08-17 23:38:00", 131),

    ("A220", ["AE03"], "2022-08-18 13:48:00", 131),
    ("A221", ["AE08"], "2022-08-18 14:58:00", 82),
    ("A222", ["AE01"], "2022-08-18 16:20:00", 211),
    ("A223", ["AE07"], "2022-08-18 17:38:00", 79),

    ("A224", ["AP14", "AP26"], "2022-08-18 18:45:00", 107),
    ("A225", ["AP16", "AP32"], "2022-08-18 19:21:00", 60),
    ("A226", ["AP04", "AP11"], "2022-08-18 20:32:00", 118),
    ("A227", ["AP09", "AP14"], "2022-08-18 21:41:00", 132),
    ("A228", ["AP05", "AP30"], "2022-08-18 23:15:00", 155),
    ("A229", ["AP45", "AP01"], "2022-08-19 01:23:00", 115),
    ("A230", ["AP19", "AP02"], "2022-08-19 02:43:00", 154),
    ("A231", ["AP08", "AP35"], "2022-08-19 04:34:00", 95),
    ("A232", ["AP45", "AP27"], "2022-08-19 05:14:00", 153),
    ("A233", ["AP44", "AP47"], "2022-08-19 06:46:00", 2051),

    ("A234", ["AP25"], "2022-08-19 08:24:00", 529),
    ("A235", ["AE05"], "2022-08-19 09:27:00", 86),
    ("A236", ["AE06"], "2022-08-19 10:34:00", 119),
    ("A237", ["AE05"], "2022-08-19 14:18:00", 189),
    ("A238", ["AE06"], "2022-08-19 14:51:00", 122),
]


# ---------------------------------------------------------
# Verified boundary reconciliation
#
# The released Test 2 label file contains four positive
# minute buckets that extend one minute beyond the
# documented second-level scenario duration.
#
# These timestamps were inspected directly in the released
# label data and are retained as positive labels.
# ---------------------------------------------------------

BOUNDARY_RECONCILIATION = {
    pd.Timestamp("2022-08-17 15:22:00"): "A213",
    pd.Timestamp("2022-08-18 15:00:00"): "A221",
    pd.Timestamp("2022-08-18 19:22:00"): "A225",
    pd.Timestamp("2022-08-18 20:34:00"): "A226",
}


def main():

    print("=" * 80)
    print("HAI 23.05 TEST 2 CLASSIFIER TARGET CONSTRUCTION")
    print("=" * 80)

    # -----------------------------------------------------
    # Load SCADA
    # -----------------------------------------------------

    print("\nLoading SCADA data...")

    scada = pd.read_csv(TEST2_SCADA_PATH)
    scada["timestamp"] = pd.to_datetime(scada["timestamp"])

    print("SCADA rows:", len(scada))

    # -----------------------------------------------------
    # Validate SCADA timestamps
    # -----------------------------------------------------

    if scada["timestamp"].isna().any():
        raise ValueError(
            "SCADA timestamp column contains invalid timestamps."
        )

    if scada["timestamp"].duplicated().any():
        raise ValueError(
            "SCADA timestamps are not unique."
        )

    # -----------------------------------------------------
    # Load labels
    # -----------------------------------------------------

    print("\nLoading labels...")

    labels = pd.read_csv(TEST2_LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])

    print("Label rows:", len(labels))

    if labels["timestamp"].isna().any():
        raise ValueError(
            "Label timestamp column contains invalid timestamps."
        )

    # -----------------------------------------------------
    # Collapse repeated label timestamps.
    #
    # Test 2 labels contain repeated rows for the same
    # minute-level timestamp.
    #
    # max() preserves an attack if any repeated label row
    # for that timestamp is positive.
    # -----------------------------------------------------

    minute_labels = (
        labels
        .groupby("timestamp")["label"]
        .max()
        .sort_index()
    )

    print(
        "Unique label timestamps:",
        len(minute_labels)
    )

    actual_positive_minutes = set(
        minute_labels[minute_labels == 1].index
    )

    print(
        "Positive label minutes:",
        len(actual_positive_minutes)
    )

    # -----------------------------------------------------
    # Expected source-label validation
    # -----------------------------------------------------

    EXPECTED_POSITIVE_MINUTES = 165

    if len(actual_positive_minutes) != EXPECTED_POSITIVE_MINUTES:
        raise ValueError(
            f"Expected {EXPECTED_POSITIVE_MINUTES} positive "
            f"label minutes, but found "
            f"{len(actual_positive_minutes)}."
        )

    print(
        "PASS: Released label file contains "
        f"{EXPECTED_POSITIVE_MINUTES} positive minute buckets."
    )

    # -----------------------------------------------------
    # Create target dataframe
    # -----------------------------------------------------

    target = pd.DataFrame({
        "timestamp": scada["timestamp"],
        "attack_label": 0,
        "scenario_id": "NORMAL",
        "attack_codes": "NORMAL",
    })

    # -----------------------------------------------------
    # Map positive minute blocks to scenarios
    #
    # First use documented duration-derived windows.
    # -----------------------------------------------------

    for scenario_id, attack_codes, start_time, duration in TEST2_SCENARIOS:

        start = pd.Timestamp(start_time)

        end = (
            start
            + pd.Timedelta(seconds=duration - 1)
        )

        minute_range = pd.date_range(
            start=start.floor("min"),
            end=end.floor("min"),
            freq="min",
        )

        scenario_minutes = [
            minute
            for minute in minute_range
            if minute in actual_positive_minutes
        ]

        if not scenario_minutes:
            print(
                f"WARNING: {scenario_id} has no matching "
                "positive label minutes."
            )
            continue

        mask = target["timestamp"].dt.floor("min").isin(
            scenario_minutes
        )

        target.loc[mask, "attack_label"] = 1

        target.loc[mask, "scenario_id"] = scenario_id

        target.loc[mask, "attack_codes"] = "+".join(
            attack_codes
        )

    # -----------------------------------------------------
    # Boundary reconciliation
    #
    # Four released positive minute buckets are not included
    # by the second-level duration-derived windows.
    #
    # They are explicitly assigned to their corresponding
    # documented scenarios.
    # -----------------------------------------------------

    print("\nApplying verified boundary reconciliation...")

    for timestamp, scenario_id in BOUNDARY_RECONCILIATION.items():

        scenario_lookup = {
            item[0]: item
            for item in TEST2_SCENARIOS
        }

        if scenario_id not in scenario_lookup:
            raise ValueError(
                f"Boundary scenario {scenario_id} "
                "does not exist in scenario catalog."
            )

        _, attack_codes, _, _ = scenario_lookup[scenario_id]

        mask = (
            target["timestamp"].dt.floor("min")
            == timestamp
        )

        matched_rows = int(mask.sum())

        if matched_rows == 0:
            raise ValueError(
                f"Boundary timestamp {timestamp} "
                "does not exist in SCADA data."
            )

        target.loc[mask, "attack_label"] = 1
        target.loc[mask, "scenario_id"] = scenario_id
        target.loc[mask, "attack_codes"] = "+".join(
            attack_codes
        )

        print(
            f"  {timestamp} -> {scenario_id} "
            f"({'+'.join(attack_codes)}) "
            f"[{matched_rows} SCADA rows]"
        )

    # -----------------------------------------------------
    # VALIDATION 1
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("TARGET VALIDATION")
    print("=" * 80)

    target_positive_minutes = set(
        target.loc[
            target["attack_label"] == 1,
            "timestamp"
        ].dt.floor("min")
    )

    print(
        "\nReleased positive minutes :",
        len(actual_positive_minutes)
    )

    print(
        "Target positive minutes   :",
        len(target_positive_minutes)
    )

    if actual_positive_minutes != target_positive_minutes:

        missing_from_target = sorted(
            actual_positive_minutes
            - target_positive_minutes
        )

        unexpected_in_target = sorted(
            target_positive_minutes
            - actual_positive_minutes
        )

        print("\nMissing from target:")
        for timestamp in missing_from_target:
            print(" ", timestamp)

        print("\nUnexpected target positives:")
        for timestamp in unexpected_in_target:
            print(" ", timestamp)

        raise ValueError(
            "Target positive-minute set does not exactly "
            "match released positive-minute set."
        )

    print(
        "PASS: Target positive-minute set exactly "
        "matches released labels."
    )

    # -----------------------------------------------------
    # VALIDATION 2
    # -----------------------------------------------------

    expected_attack_rows = (
        len(actual_positive_minutes) * 60
    )

    actual_attack_rows = int(
        target["attack_label"].sum()
    )

    expected_normal_rows = (
        len(target) - expected_attack_rows
    )

    actual_normal_rows = int(
        (target["attack_label"] == 0).sum()
    )

    print(
        "\nExpected attack rows:",
        expected_attack_rows
    )

    print(
        "Actual attack rows  :",
        actual_attack_rows
    )

    print(
        "Expected normal rows:",
        expected_normal_rows
    )

    print(
        "Actual normal rows  :",
        actual_normal_rows
    )

    if actual_attack_rows != expected_attack_rows:
        raise ValueError(
            "Attack row count validation failed."
        )

    if actual_normal_rows != expected_normal_rows:
        raise ValueError(
            "Normal row count validation failed."
        )

    print(
        "PASS: Attack and normal row counts are correct."
    )

    # -----------------------------------------------------
    # VALIDATION 3
    # -----------------------------------------------------
    # Verify the four reconciled boundary minutes.
    # -----------------------------------------------------

    print("\nValidating boundary reconciliations...")

    for timestamp, scenario_id in BOUNDARY_RECONCILIATION.items():

        minute_mask = (
            target["timestamp"].dt.floor("min")
            == timestamp
        )

        boundary_rows = target.loc[minute_mask]

        if len(boundary_rows) != 60:
            raise ValueError(
                f"{timestamp}: expected 60 SCADA rows, "
                f"found {len(boundary_rows)}."
            )

        if not boundary_rows["attack_label"].eq(1).all():
            raise ValueError(
                f"{timestamp}: not all rows are attack-labelled."
            )

        if not boundary_rows["scenario_id"].eq(
            scenario_id
        ).all():
            raise ValueError(
                f"{timestamp}: incorrect scenario assignment."
            )

    print(
        "PASS: All four boundary minutes reconciled."
    )

    # -----------------------------------------------------
    # VALIDATION 4
    # -----------------------------------------------------
    # Verify every documented scenario is represented.
    # -----------------------------------------------------

    scenario_ids = {
        scenario[0]
        for scenario in TEST2_SCENARIOS
    }

    assigned_scenarios = set(
        target.loc[
            target["attack_label"] == 1,
            "scenario_id"
        ]
    )

    missing_scenarios = (
        scenario_ids - assigned_scenarios
    )

    if missing_scenarios:
        raise ValueError(
            "Documented scenarios missing from target: "
            + ", ".join(sorted(missing_scenarios))
        )

    print(
        "PASS: All 38 documented scenarios represented."
    )

    # -----------------------------------------------------
    # VALIDATION 5
    # -----------------------------------------------------
    # Verify scenario attack codes against catalog.
    # -----------------------------------------------------

    scenario_catalog = {
        scenario_id: "+".join(attack_codes)
        for scenario_id, attack_codes, _, _
        in TEST2_SCENARIOS
    }

    for scenario_id, expected_codes in scenario_catalog.items():

        rows = target.loc[
            target["scenario_id"] == scenario_id
        ]

        if rows.empty:
            raise ValueError(
                f"{scenario_id}: no target rows found."
            )

        actual_codes = set(
            rows["attack_codes"]
        )

        if actual_codes != {expected_codes}:
            raise ValueError(
                f"{scenario_id}: attack-code mismatch. "
                f"Expected {expected_codes}, "
                f"found {actual_codes}"
            )

    print(
        "PASS: Scenario attack-code mappings validated."
    )

    # -----------------------------------------------------
    # VALIDATION 6
    # -----------------------------------------------------
    # Verify combination scenarios remain multi-label.
    # -----------------------------------------------------

    expected_combinations = {
        scenario_id: "+".join(attack_codes)
        for scenario_id, attack_codes, _, _
        in TEST2_SCENARIOS
        if len(attack_codes) > 1
    }

    actual_combinations = {
        scenario_id: codes
        for scenario_id, codes
        in zip(
            target["scenario_id"],
            target["attack_codes"]
        )
        if "+" in codes
    }

    for scenario_id, expected_codes in expected_combinations.items():

        if scenario_id not in actual_combinations:
            raise ValueError(
                f"Combination scenario {scenario_id} "
                "was not preserved."
            )

        if actual_combinations[scenario_id] != expected_codes:
            raise ValueError(
                f"Combination scenario {scenario_id}: "
                "attack codes were modified."
            )

    if len(expected_combinations) != 10:
        raise ValueError(
            "Expected exactly 10 combination scenarios."
        )

    print(
        "PASS: All 10 combination scenarios preserved."
    )

    # -----------------------------------------------------
    # VALIDATION 7
    # -----------------------------------------------------
    # No NORMAL row should have attack codes.
    # -----------------------------------------------------

    normal_rows = target[
        target["attack_label"] == 0
    ]

    if not normal_rows["scenario_id"].eq(
        "NORMAL"
    ).all():
        raise ValueError(
            "Some normal rows have a scenario assignment."
        )

    if not normal_rows["attack_codes"].eq(
        "NORMAL"
    ).all():
        raise ValueError(
            "Some normal rows have attack codes."
        )

    print(
        "PASS: Normal rows contain no scenario or "
        "attack-code assignments."
    )

    # -----------------------------------------------------
    # VALIDATION 8
    # -----------------------------------------------------
    # No attack row should remain NORMAL.
    # -----------------------------------------------------

    attack_rows = target[
        target["attack_label"] == 1
    ]

    if attack_rows["scenario_id"].eq(
        "NORMAL"
    ).any():
        raise ValueError(
            "Attack rows contain NORMAL scenario assignments."
        )

    if attack_rows["attack_codes"].eq(
        "NORMAL"
    ).any():
        raise ValueError(
            "Attack rows contain NORMAL attack codes."
        )

    print(
        "PASS: All attack rows have scenario assignments."
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    target.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL TARGET SUMMARY")
    print("=" * 80)

    print("\nTarget distribution:")
    print(
        target["attack_label"]
        .value_counts()
        .sort_index()
    )

    print("\nScenario distribution:")
    print(
        target["scenario_id"]
        .value_counts()
    )

    print("\nCombination scenarios:")

    combinations = target[
        target["attack_codes"].str.contains(
            "+",
            regex=False
        )
    ]

    print(
        combinations[
            [
                "timestamp",
                "scenario_id",
                "attack_codes"
            ]
        ]
        .drop_duplicates("scenario_id")
        .sort_values("scenario_id")
        .to_string(index=False)
    )

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 80)
    print("ALL TARGET VALIDATIONS PASSED")
    print("TARGET CONSTRUCTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()