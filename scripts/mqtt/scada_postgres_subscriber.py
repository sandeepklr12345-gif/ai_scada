import json
from datetime import datetime

import paho.mqtt.client as mqtt
import requests

from scripts.database.postgres_storage import PostgreSQLStorage


MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "ai_scada/scada/600mw"

FASTAPI_URL = "http://127.0.0.1:8000/forecast"

SOURCE_ID = 1
EQUIPMENT_ID = 3
POWER_PARAMETER_ID = 1

MAX_MESSAGES = 10

storage = PostgreSQLStorage()
message_count = 0


def on_connect(client, userdata, flags, reason_code, properties):

    print("=" * 70)
    print("POSTGRESQL MQTT INTEGRATION TEST")
    print("=" * 70)

    print(f"\nConnected to MQTT broker: {MQTT_HOST}:{MQTT_PORT}")
    print(f"Subscribed topic       : {MQTT_TOPIC}")

    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, message):

    global message_count

    try:
        # ------------------------------------------------------------
        # 1. Decode MQTT message
        # ------------------------------------------------------------

        payload = json.loads(
            message.payload.decode("utf-8")
        )

        timestamp = datetime.fromisoformat(
            payload["timestamp"]
        )

        features = payload["features"]

        power_value = features["Power output\n（MW）"]

        print("\nReceived MQTT message")
        print("-" * 70)
        print(f"Timestamp       : {timestamp}")
        print(f"Source          : {payload.get('source')}")
        print(f"Replay step     : {payload.get('step')}")
        print(f"Feature count   : {len(features)}")
        print(f"Power output    : {power_value} MW")

        # ------------------------------------------------------------
        # 2. Store current power measurement
        # ------------------------------------------------------------

        measurement_id = storage.save_measurement(
            timestamp=timestamp,
            source_id=SOURCE_ID,
            equipment_id=EQUIPMENT_ID,
            parameter_id=POWER_PARAMETER_ID,
            value=power_value,
            quality_status="GOOD",
        )

        print(f"\nMeasurement stored: ID={measurement_id}")

        # ------------------------------------------------------------
        # 3. Send model features to FastAPI
        # ------------------------------------------------------------

        response = requests.post(
            FASTAPI_URL,
            json={"features": features},
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        forecasts = result["forecast"]

        print("\nFastAPI forecast received")
        print("-" * 70)
        print(f"2 minute  : {forecasts['power_2min']:.6f} MW")
        print(f"10 minute : {forecasts['power_10min']:.6f} MW")
        print(f"30 minute : {forecasts['power_30min']:.6f} MW")

        # ------------------------------------------------------------
        # 4. Store the three forecast outputs
        # ------------------------------------------------------------

        forecast_horizons = {
            "power_2min": 2,
            "power_10min": 10,
            "power_30min": 30,
        }

        prediction_ids = []

        for prediction_name, horizon in forecast_horizons.items():

            prediction_id = storage.save_prediction(
                timestamp=timestamp,
                source_id=SOURCE_ID,
                equipment_id=EQUIPMENT_ID,
                model_name="600MW Power Output Forecasting",
                prediction_type="power_forecast",
                predicted_value=float(
                    forecasts[prediction_name]
                ),
                confidence=None,
                risk_level=None,
                prediction_horizon_minutes=horizon,
                description="MQTT SCADA replay forecast",
            )

            prediction_ids.append(prediction_id)

        print("\nForecasts stored")
        print("-" * 70)

        for prediction_id in prediction_ids:
            print(f"Prediction ID: {prediction_id}")

        # ------------------------------------------------------------
        # 5. Complete one-message test
        # ------------------------------------------------------------

        message_count += 1

        print(f"\nMessages processed: {message_count}/{MAX_MESSAGES}")

        if message_count >= MAX_MESSAGES:

            print("\n" + "=" * 70)
            print("MULTI-MESSAGE INTEGRATION TEST: PASS")
            print("=" * 70)

            print(f"\nProcessed {message_count} MQTT messages.")
            print("Stopping subscriber...")

            client.disconnect()

    except Exception as exc:

        print("\n" + "=" * 70)
        print("INTEGRATION TEST: FAILED")
        print("=" * 70)

        print(f"\nError: {exc}")

        client.disconnect()


def main():

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="ai_scada_postgres_test",
    )

    client.on_connect = on_connect
    client.on_message = on_message

    print("Connecting to MQTT broker...")

    client.connect(
        MQTT_HOST,
        MQTT_PORT,
        keepalive=60,
    )

    client.loop_forever()


if __name__ == "__main__":
    main()