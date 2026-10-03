from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hai"
    / "hai-23.05"
    / "label-test2.csv"
)


TEST2_SCENARIOS = [
    ("A201", "2022-08-17 01:27:00", 132),
    ("A202", "2022-08-17 03:37:00", 131),
    ("A203", "2022-08-17 04:21:00", 68),
    ("A204", "2022-08-17 05:46:00", 122),
    ("A205", "2022-08-17 06:21:00", 85),
    ("A206", "2022-08-17 08:36:00", 196),
    ("A207", "2022-08-17 09:42:00", 614),
    ("A208", "2022-08-17 10:36:00", 133),
    ("A209", "2022-08-17 11:35:00", 85),
    ("A210", "2022-08-17 12:25:00", 88),
    ("A211", "2022-08-17 13:47:00", 204),
    ("A212", "2022-08-17 14:25:00", 127),
    ("A213", "2022-08-17 15:13:00", 539),
    ("A214", "2022-08-17 17:34:00", 61),
    ("A215", "2022-08-17 18:16:00", 147),
    ("A216", "2022-08-17 19:40:00", 95),
    ("A217", "2022-08-17 20:12:00", 505),
    ("A218", "2022-08-17 22:41:00", 214),
    ("A219", "2022-08-17 23:38:00", 131),

    ("A220", "2022-08-18 13:48:00", 131),
    ("A221", "2022-08-18 14:58:00", 82),
    ("A222", "2022-08-18 16:20:00", 211),
    ("A223", "2022-08-18 17:38:00", 79),

    ("A224", "2022-08-18 18:45:00", 107),
    ("A225", "2022-08-18 19:21:00", 60),
    ("A226", "2022-08-18 20:32:00", 118),
    ("A227", "2022-08-18 21:41:00", 132),
    ("A228", "2022-08-18 23:15:00", 155),
    ("A229", "2022-08-19 01:23:00", 115),
    ("A230", "2022-08-19 02:43:00", 154),
    ("A231", "2022-08-19 04:34:00", 95),
    ("A232", "2022-08-19 05:14:00", 153),
    ("A233", "2022-08-19 06:46:00", 2051),

    ("A234", "2022-08-19 08:24:00", 529),
    ("A235", "2022-08-19 09:27:00", 86),
    ("A236", "2022-08-19 10:34:00", 119),
    ("A237", "2022-08-19 14:18:00", 189),
    ("A238", "2022-08-19 14:51:00", 122),
]


def main():

    print("=" * 90)
    print("HAI 23.05 TEST 2 UNMATCHED POSITIVE MINUTES")
    print("=" * 90)

    labels = pd.read_csv(LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])

    minute_labels = (
        labels
        .groupby("timestamp")["label"]
        .max()
        .sort_index()
    )

    actual_positive_minutes = set(
        minute_labels[minute_labels == 1].index
    )

    expected_positive_minutes = set()

    for scenario_id, start_time, duration in TEST2_SCENARIOS:

        start = pd.Timestamp(start_time)
        end = start + pd.Timedelta(seconds=duration - 1)

        minute_range = pd.date_range(
            start=start.floor("min"),
            end=end.floor("min"),
            freq="min",
        )

        expected_positive_minutes.update(minute_range)

    unmatched = sorted(
        actual_positive_minutes - expected_positive_minutes
    )

    print("\nActual positive minutes   :", len(actual_positive_minutes))
    print("Expected scenario minutes :", len(expected_positive_minutes))
    print("Unmatched positive minutes:", len(unmatched))

    print("\nUnmatched minutes:")
    print("-" * 90)

    for timestamp in unmatched:
        print(timestamp)

    print("\n" + "-" * 90)

    if len(unmatched) == 0:
        print("PASS: Every positive label minute is explained by a scenario.")

    else:
        print(
            "REVIEW REQUIRED: These positive label minutes are present "
            "in the released labels but are not covered by the documented "
            "duration-derived scenario windows."
        )

    print("\n" + "=" * 90)
    print("INSPECTION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()