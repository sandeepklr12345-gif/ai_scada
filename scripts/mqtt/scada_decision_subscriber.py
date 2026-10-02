import json

import paho.mqtt.client as mqtt
import requests


# ============================================================
# MQTT CONFIGURATION
# ============================================================

MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883

MQTT_TOPIC = "ai_scada/scada/600mw"


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

FASTAPI_URL = "http://127.0.0.1:8000/decision"


# ============================================================
# DECISION STATE
# ============================================================

persistent_anomaly_count = 0


# ============================================================
# CALLBACKS
# ============================================================

def on_connect(client, userdata, flags, reason_code, properties):

    print("Connected to MQTT broker.")

    client.subscribe(MQTT_TOPIC)

    print(f"Subscribed to: {MQTT_TOPIC}")


def on_message(client, userdata, message):

    global persistent_anomaly_count

    try:

        # ----------------------------------------------------
        # Parse MQTT message
        # ----------------------------------------------------

        payload = json.loads(
            message.payload.decode("utf-8")
        )

        timestamp = payload["timestamp"]
        source = payload["source"]
        step = payload["step"]
        features = payload["features"]

        print()
        print("=" * 70)
        print(
            f"Received step={step} "
            f"timestamp={timestamp} "
            f"source={source}"
        )
        print(f"Features: {len(features)}")

        # ----------------------------------------------------
        # Send features to FastAPI Decision endpoint
        # ----------------------------------------------------

        response = requests.post(
            FASTAPI_URL,
            json={
                "features": features,
                "persistent_anomaly_count": (
                    persistent_anomaly_count
                ),
            },
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        decision = result["decision"]

        # ----------------------------------------------------
        # Update persistence state
        # ----------------------------------------------------

        if decision["anomaly_state"] == "ANOMALY":

            persistent_anomaly_count += 1

        else:

            persistent_anomaly_count = 0

        # ----------------------------------------------------
        # Display Decision Support result
        # ----------------------------------------------------

        print("\nDecision Support:")

        print(
            f"  Decision Level       : "
            f"{decision['decision_level']}"
        )

        print(
            f"  Action               : "
            f"{decision['action']}"
        )

        print(
            f"  Anomaly State        : "
            f"{decision['anomaly_state']}"
        )

        print(
            f"  Anomaly Score        : "
            f"{decision['anomaly_score']}"
        )

        print(
            f"  Predicted Mechanism  : "
            f"{decision['predicted_attack_mechanism']}"
        )

        print(
            f"  Classifier Score     : "
            f"{decision['classifier_score']}"
        )

        print(
            f"  Persistent Anomalies : "
            f"{persistent_anomaly_count}"
        )

        print(
            f"  Reason               : "
            f"{decision['reason']}"
        )

    except requests.RequestException as exc:

        print(
            f"FastAPI request failed: {exc}"
        )

    except Exception as exc:

        print(
            f"Failed to process MQTT message: {exc}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI_SCADA MQTT → FASTAPI DECISION SUBSCRIBER")
    print("=" * 70)

    print(
        f"\nMQTT broker : "
        f"{MQTT_HOST}:{MQTT_PORT}"
    )

    print(
        f"MQTT topic  : "
        f"{MQTT_TOPIC}"
    )

    print(
        f"FastAPI URL : "
        f"{FASTAPI_URL}"
    )

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="ai_scada_decision_subscriber",
    )

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(
        MQTT_HOST,
        MQTT_PORT,
        keepalive=60,
    )

    try:

        client.loop_forever()

    except KeyboardInterrupt:

        print("\nSubscriber stopped by user.")

    finally:

        client.disconnect()

        print("MQTT connection closed.")


if __name__ == "__main__":
    main()