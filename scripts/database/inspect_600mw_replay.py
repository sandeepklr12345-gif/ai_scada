import pandas as pd


REPLAY_PATH = "data/integration/runtime_simulation/600mw_scada_replay.csv"

METADATA_COLUMNS = {
    "timestamp",
    "replay_source",
    "replay_step",
}

POWER_COLUMN = "Power output\n（MW）"


def main():
    df = pd.read_csv(REPLAY_PATH, nrows=1)

    features = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]

    print("=" * 70)
    print("600 MW REPLAY INSPECTION")
    print("=" * 70)

    print(f"\nTotal model features: {len(features)}")

    print("\nFirst 15 model features:")
    print("-" * 70)

    for index, feature in enumerate(features[:15], start=1):
        print(f"{index:2}. {feature!r}")

    power_value = df[POWER_COLUMN].iloc[0]

    print("\nCurrent power output:")
    print("-" * 70)
    print(f"Column : {POWER_COLUMN!r}")
    print(f"Value  : {power_value} MW")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()