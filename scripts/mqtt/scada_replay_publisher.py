from pathlib import Path
import json
import time

import pandas as pd
import paho.mqtt.client as mqtt


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPLAY_FILE = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_scada_replay.csv"
)


# ============================================================
# MQTT CONFIGURATION
# ============================================================

MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883

MQTT_TOPIC = "ai_scada/scada/600mw"

REPLAY_DELAY_SECONDS = 1.0


# ============================================================
# LOAD REPLAY DATA
# ============================================================

def load_replay():

    if not REPLAY_FILE.exists():
        raise FileNotFoundError(
            f"Replay file not found: {REPLAY_FILE}"
        )

    df = pd.read_csv(REPLAY_FILE)

    required_columns = {
        "timestamp",
        "replay_source",
        "replay_step",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Replay dataset is missing required columns: {sorted(missing)}"
        )

    return df


# ============================================================
# BUILD MQTT MESSAGE
# ============================================================

def build_message(row):

    excluded_columns = {
        "timestamp",
        "replay_source",
        "replay_step",
    }

    features = {
        column: float(row[column])
        for column in row.index
        if column not in excluded_columns
    }

    return {
        "timestamp": str(row["timestamp"]),
        "source": str(row["replay_source"]),
        "step": int(row["replay_step"]),
        "features": features,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("600 MW SCADA MQTT REPLAY PUBLISHER")
    print("=" * 70)

    print(f"\nReplay file : {REPLAY_FILE}")
    print(f"MQTT broker : {MQTT_HOST}:{MQTT_PORT}")
    print(f"MQTT topic  : {MQTT_TOPIC}")
    print(f"Replay delay: {REPLAY_DELAY_SECONDS} seconds")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_replay()

    print(f"\nRows loaded: {len(df)}")
    print(f"Columns    : {len(df.columns)}")

    # --------------------------------------------------------
    # Create MQTT client
    # --------------------------------------------------------

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="ai_scada_600mw_replay",
    )

    print("\nConnecting to MQTT broker...")

    client.connect(
        MQTT_HOST,
        MQTT_PORT,
        keepalive=60,
    )

    print("MQTT connection established.")

    client.loop_start()

    try:

        # ----------------------------------------------------
        # Replay rows
        # ----------------------------------------------------

        for _, row in df.iterrows():

            message = build_message(row)

            payload = json.dumps(message)

            result = client.publish(
                MQTT_TOPIC,
                payload,
                qos=0,
            )

            result.wait_for_publish()

            print(
                f"Published step={message['step']} "
                f"timestamp={message['timestamp']}"
            )

            time.sleep(REPLAY_DELAY_SECONDS)

    except KeyboardInterrupt:

        print("\nReplay interrupted by user.")

    finally:

        client.loop_stop()
        client.disconnect()

        print("\nMQTT connection closed.")

    print("\n" + "=" * 70)
    print("600 MW SCADA MQTT REPLAY: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()