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

FASTAPI_URL = "http://127.0.0.1:8000/forecast"


# ============================================================
# CALLBACKS
# ============================================================

def on_connect(client, userdata, flags, reason_code, properties):

    print("Connected to MQTT broker.")

    client.subscribe(MQTT_TOPIC)

    print(f"Subscribed to: {MQTT_TOPIC}")


def on_message(client, userdata, message):

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

        print(
            f"\nReceived step={step} "
            f"timestamp={timestamp} "
            f"features={len(features)}"
        )

        # ----------------------------------------------------
        # Send features to FastAPI
        # ----------------------------------------------------

        response = requests.post(
            FASTAPI_URL,
            json={
                "features": features
            },
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        # ----------------------------------------------------
        # Display forecast
        # ----------------------------------------------------

        forecast = result["forecast"]

        print(
            "Forecast:"
        )

        print(
            f"  2 min  : {forecast['power_2min']:.6f}"
        )

        print(
            f"  10 min : {forecast['power_10min']:.6f}"
        )

        print(
            f"  30 min : {forecast['power_30min']:.6f}"
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
    print("AI_SCADA MQTT → FASTAPI SUBSCRIBER")
    print("=" * 70)

    print(
        f"\nMQTT broker : {MQTT_HOST}:{MQTT_PORT}"
    )

    print(
        f"MQTT topic  : {MQTT_TOPIC}"
    )

    print(
        f"FastAPI URL : {FASTAPI_URL}"
    )

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="ai_scada_ai_subscriber",
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