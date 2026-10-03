from hai_scenario_mapping import HAI_SCENARIO_MAPPING


TEST2_SCENARIOS = {
    "A201": ["AP14"],
    "A202": ["AP15"],
    "A203": ["AP16"],
    "A204": ["AP17"],
    "A205": ["AP18"],
    "A206": ["AP03"],
    "A207": ["AP43"],
    "A208": ["AP42"],
    "A209": ["AP23"],
    "A210": ["AP23"],
    "A211": ["AP46"],
    "A212": ["AP24"],
    "A213": ["AP25"],
    "A214": ["AP19"],
    "A215": ["AP20"],
    "A216": ["AP21"],
    "A217": ["AP22"],
    "A218": ["AP07"],
    "A219": ["AP13"],

    "A220": ["AE03"],
    "A221": ["AE08"],
    "A222": ["AE01"],
    "A223": ["AE07"],

    "A224": ["AP14", "AP26"],
    "A225": ["AP16", "AP32"],
    "A226": ["AP04", "AP11"],
    "A227": ["AP09", "AP14"],
    "A228": ["AP05", "AP30"],
    "A229": ["AP45", "AP01"],
    "A230": ["AP19", "AP02"],
    "A231": ["AP08", "AP35"],
    "A232": ["AP45", "AP27"],
    "A233": ["AP44", "AP47"],

    "A234": ["AP25"],
    "A235": ["AE05"],
    "A236": ["AE06"],
    "A237": ["AE05"],
    "A238": ["AE06"],
}


def main():

    print("=" * 80)
    print("HAI 23.05 TEST 2 SCENARIO MAPPING VALIDATION")
    print("=" * 80)

    print(f"\nDocumented scenarios : {len(TEST2_SCENARIOS)}")
    print(f"Mapped attack codes  : {len(HAI_SCENARIO_MAPPING)}")

    missing_codes = set()
    scenario_failures = []

    print("\nScenario resolution:")
    print("-" * 80)

    for scenario_id, attack_codes in TEST2_SCENARIOS.items():

        names = []

        for attack_code in attack_codes:

            mapping = HAI_SCENARIO_MAPPING.get(attack_code)

            if mapping is None:
                missing_codes.add(attack_code)
            else:
                names.append(mapping["name"])

        if len(names) != len(attack_codes):

            scenario_failures.append(scenario_id)

            print(
                f"{scenario_id} | "
                f"FAIL | "
                f"{', '.join(attack_codes)}"
            )

        else:

            print(
                f"{scenario_id} | "
                f"PASS | "
                f"{' + '.join(names)}"
            )

    print("\n" + "-" * 80)

    if missing_codes:

        print("\nMissing attack-code mappings:")

        for code in sorted(missing_codes):
            print(f"  - {code}")

    else:
        print("\nMissing attack-code mappings: NONE")

    print(f"\nScenario failures: {len(scenario_failures)}")

    if not missing_codes and not scenario_failures:

        print("\nPASS")
        print("All Test 2 scenarios have meaningful attack mappings.")

    else:

        print("\nFAIL")
        print(
            "Some Test 2 attack codes are not yet represented "
            "in HAI_SCENARIO_MAPPING."
        )

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()