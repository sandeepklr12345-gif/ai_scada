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


def main():

    print("=" * 90)
    print("HAI 23.05 TEST 2 ACTUAL POSITIVE LABEL BLOCKS")
    print("=" * 90)

    labels = pd.read_csv(LABEL_PATH)

    labels["timestamp"] = pd.to_datetime(labels["timestamp"])

    # The file contains repeated rows for the same minute.
    # Keep one label value per unique timestamp.
    minute_labels = (
        labels
        .groupby("timestamp")["label"]
        .max()
        .sort_index()
    )

    positive = minute_labels[minute_labels == 1]

    print("\nUnique timestamps :", len(minute_labels))
    print("Positive timestamps:", len(positive))

    # ------------------------------------------------------------
    # FIND CONTIGUOUS POSITIVE MINUTE BLOCKS
    # ------------------------------------------------------------

    positive_times = positive.index

    blocks = []

    if len(positive_times) > 0:

        block_start = positive_times[0]
        previous = positive_times[0]

        for current in positive_times[1:]:

            difference = current - previous

            if difference != pd.Timedelta(minutes=1):

                blocks.append(
                    (
                        block_start,
                        previous,
                        int(
                            (
                                previous - block_start
                            ).total_seconds()
                            / 60
                        )
                        + 1,
                    )
                )

                block_start = current

            previous = current

        blocks.append(
            (
                block_start,
                previous,
                int(
                    (
                        previous - block_start
                    ).total_seconds()
                    / 60
                )
                + 1,
            )
        )

    print("\nPositive minute blocks:")
    print("-" * 90)

    for index, (start, end, minutes) in enumerate(
        blocks,
        start=1,
    ):

        print(
            f"{index:02d} | "
            f"{start} -> {end} | "
            f"{minutes:3d} minutes"
        )

    print("\n" + "-" * 90)

    print("Number of positive blocks:", len(blocks))
    print("Total positive minutes  :", len(positive))

    # ------------------------------------------------------------
    # SHOW RAW POSITIVE LABELS AROUND FIRST ATTACK
    # ------------------------------------------------------------

    print("\nFirst positive block raw timestamps:")

    if blocks:

        start, end, _ = blocks[0]

        sample = labels[
            (labels["timestamp"] >= start)
            & (
                labels["timestamp"]
                <= end + pd.Timedelta(minutes=1)
            )
        ]

        print(
            sample[
                ["timestamp", "label"]
            ].drop_duplicates()
            .to_string(index=False)
        )

    print("\n" + "=" * 90)
    print("INSPECTION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()