HAI_SCENARIO_MAPPING = {

    # ---------------------------------------------------------
    # P1-LC
    # ---------------------------------------------------------

    "AP14": {
        "name": "Water-Level Setpoint Manipulation",
        "controller": "P1-LC",
        "target": "P1_B3004",
    },

    "AP15": {
        "name": "Water-Level Setpoint + Sensor Manipulation",
        "controller": "P1-LC",
        "target": "P1_B3004, P1_LIT01",
    },

    "AP16": {
        "name": "Water-Level Control Output Manipulation",
        "controller": "P1-LC",
        "target": "P1_LCV01D",
    },

    "AP17": {
        "name": "Water-Level Control + Sensor Manipulation",
        "controller": "P1-LC",
        "target": "P1_LCV01D, P1_LIT01",
    },

    "AP18": {
        "name": "Short-Term Water-Level Control Manipulation",
        "controller": "P1-LC",
        "target": "P1_LCV01D",
    },

    "AP19": {
        "name": "Temperature Control Output Manipulation",
        "controller": "P1-TC",
        "target": "P1_FCV01D",
    },

    "AP20": {
        "name": "Temperature Control + Sensor Manipulation",
        "controller": "P1-TC",
        "target": "P1_FCV01D, P1_TIT01",
    },

    "AP21": {
        "name": "Short-Term Temperature Control Manipulation",
        "controller": "P1-TC",
        "target": "P1_FCV01D",
    },

    "AP22": {
        "name": "Long-Term Temperature Setpoint Manipulation",
        "controller": "P1-TC",
        "target": "P1_B4002",
    },

    # ---------------------------------------------------------
    # P1-PC / P1-FC
    # ---------------------------------------------------------

    "AP03": {
        "name": "Pressure Setpoint + Multi-Sensor Manipulation",
        "controller": "P1-PC",
        "target": "P1_B2016, P1_PIT01, P1_FIT01",
    },

    "AP04": {
        "name": "Pressure Control Output Manipulation",
        "controller": "P1-PC",
        "target": "P1_PCV01D",
    },

    "AP05": {
        "name": "Pressure Control + Sensor Manipulation",
        "controller": "P1-PC",
        "target": "P1_PCV01D, P1_PIT01",
    },

    "AP07": {
        "name": "Short-Term Pressure Control Manipulation",
        "controller": "P1-PC",
        "target": "P1_PCV01D",
    },

    "AP09": {
        "name": "Flow Setpoint + Sensor Manipulation",
        "controller": "P1-FC",
        "target": "P1_B3005, P1_FT03",
    },

    "AP11": {
        "name": "Flow Control Output Manipulation",
        "controller": "P1-FC",
        "target": "P1_FCV03D",
    },

    "AP13": {
        "name": "Short-Term Flow Control Manipulation",
        "controller": "P1-FC",
        "target": "P1_FCV03D",
    },

    "AP23": {
        "name": "Control Output Manipulation",
        "controller": "P1-CC",
        "target": "P1_PP04",
    },

    "AP24": {
        "name": "Short-Term Control Output Manipulation",
        "controller": "P1-CC",
        "target": "P1_PP04",
    },

    "AP25": {
        "name": "Long-Term Control Setpoint Manipulation",
        "controller": "P1-CC",
        "target": "P1_PP04_SP",
    },

    # ---------------------------------------------------------
    # P2
    # ---------------------------------------------------------

    "AP26": {
        "name": "Safety Control Setpoint Manipulation",
        "controller": "P2-SC",
        "target": "P2_AutoSD",
    },

    "AP27": {
        "name": "Safety Setpoint + Sensor Manipulation",
        "controller": "P2-SC",
        "target": "P2_AutoSD, P2_SIT01",
    },

    "AP30": {
        "name": "Safety Control + Sensor Manipulation",
        "controller": "P2-SC",
        "target": "P2_SCO, P2_SIT01",
    },

    "AP32": {
        "name": "Turbine Temperature Setpoint Manipulation",
        "controller": "P2-TC",
        "target": "P2_VTR01",
    },

    "AP35": {
        "name": "Level Control Sensor Manipulation",
        "controller": "P3-LC",
        "target": "P3_LCP01D",
    },

    # ---------------------------------------------------------
    # Long-term / advanced scenarios
    # ---------------------------------------------------------

    "AP42": {
        "name": "Water-Level Control + Multi-Sensor Manipulation",
        "controller": "P1-LC",
        "target": "P1_LCV01D, P1_LIT01, P1_FT03",
    },

    "AP43": {
        "name": "Long-Term Water-Level Control Manipulation",
        "controller": "P1-LC",
        "target": "P1_LCV01D",
    },

    "AP44": {
        "name": "Long-Term Water-Level Control + Sensor Manipulation",
        "controller": "P1-LC",
        "target": "P1_LCV01D, P1_LIT01",
    },

    "AP45": {
        "name": "Temperature Setpoint Manipulation",
        "controller": "P1-TC",
        "target": "P1_B4002",
    },

    "AP46": {
        "name": "Control Output + Temperature Sensor Manipulation",
        "controller": "P1-CC",
        "target": "P1_PP04, P1_TIT03",
    },

    "AP47": {
        "name": "Long-Term Turbine Temperature Setpoint Manipulation",
        "controller": "P2-TC",
        "target": "P2_VTR02",
    },
    "AP01": {
    "name": "Pressure Setpoint Manipulation",
    "controller": "P1-PC",
    "target": "P1_B2016",
    },
    "AP02": {
    "name": "Pressure Setpoint + Sensor Manipulation",
    "controller": "P1-PC",
    "target": "P1_B2016, P1_PIT01",
    },
    "AP08": {
    "name": "Flow Setpoint Manipulation",
    "controller": "P1-FC",
    "target": "P1_B3005",
    },

    # ---------------------------------------------------------
    # Internal-point attacks
    # ---------------------------------------------------------

    "AE01": {
        "name": "Pressure Control Internal Initialization Manipulation",
        "controller": "P1-PC",
        "target": "1001.09-OUT",
    },

    "AE03": {
        "name": "Water-Level Control Internal Initialization Manipulation",
        "controller": "P1-LC",
        "target": "1002.14-OUT",
    },

    "AE05": {
        "name": "Temperature Control Calibration Manipulation",
        "controller": "P1-TC",
        "target": "1003.05-OUT",
    },

    "AE06": {
        "name": "Boiler Flow Calibration Manipulation",
        "controller": "P1-TC",
        "target": "DM-FT02Z",
    },

    "AE07": {
        "name": "Control Command Calibration Manipulation",
        "controller": "P1-CC",
        "target": "1020.15-OUT",
    },

    "AE08": {
        "name": "Heater Threshold Manipulation",
        "controller": "P1-HC",
        "target": "1004.21-OUT",
    },
}


def get_scenario_name(attack_code: str) -> str:
    """Return the human-readable name for an HAI attack code."""

    attack = HAI_SCENARIO_MAPPING.get(attack_code)

    if attack is None:
        return "Unmapped HAI Attack Scenario"

    return attack["name"]


if __name__ == "__main__":

    print("=" * 70)
    print("HAI 23.05 SCENARIO MAPPING")
    print("=" * 70)

    print(f"\nTotal mapped attack codes: {len(HAI_SCENARIO_MAPPING)}")

    print("\nSample mappings:")
    print("-" * 70)

    for attack_code in ["AP14", "AP15", "AP16", "AP22", "AP26", "AP30", "AE01", "AE05"]:
        attack = HAI_SCENARIO_MAPPING[attack_code]

        print(
            f"{attack_code:5} | "
            f"{attack['name']}"
        )

    print("\nFunction test:")
    print("-" * 70)

    print("AP14 ->", get_scenario_name("AP14"))
    print("AE01 ->", get_scenario_name("AE01"))

    print("\n" + "=" * 70)
    print("SCENARIO MAPPING SELF-TEST COMPLETE")
    print("=" * 70)