from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCADA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
    / "hai-test2.csv"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
    / "label-test2.csv"
)


# Documented HAI 23.05 / HAIEnd 23.05 attack schedule
# corresponding to Test 2.
#
# IMPORTANT:
# These are documented attack intervals, not inferred labels.

SCENARIOS = [
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


def main():

    print("=" * 80)
    print("HAI 23.05 TEST 2 ATTACK SCENARIO ANALYSIS")
    print("=" * 80)

    scada = pd.read_csv(SCADA_PATH)
    labels = pd.read_csv(LABEL_PATH)

    scada["timestamp"] = pd.to_datetime(scada["timestamp"])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])

    print("\nSCADA shape :", scada.shape)
    print("Label shape :", labels.shape)

    print("\nSCADA time range:")
    print(scada["timestamp"].min())
    print(scada["timestamp"].max())

    print("\nUnique SCADA timestamps:")
    print(scada["timestamp"].nunique())

    print("\nUnique label timestamps:")
    print(labels["timestamp"].nunique())

    print("\nLabel value distribution:")
    print(labels["label"].value_counts(dropna=False))

    print("\nDocumented scenarios:")
    print(len(SCENARIOS))

    print("\nScenario schedule:")

    for scenario_id, primitives, start_text, duration in SCENARIOS:

        start = pd.Timestamp(start_text)

        end = start + pd.Timedelta(seconds=duration)

        print(
            f"{scenario_id:5s} | "
            f"{','.join(primitives):12s} | "
            f"{start} -> {end} | "
            f"{duration:4d} sec"
        )

    print("\n" + "=" * 80)
    print("TEST 2 LABEL ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()