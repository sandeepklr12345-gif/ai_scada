"""Production MQTT subscriber for historical 600 MW replay forecasts.

The runtime adapter keeps only the 71 RAW_SOURCE fields and builds the exact
119-feature model row before this subscriber calls the existing FastAPI
forecast endpoint. Replay-provided lag/time fields are never sent to the API.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
INTEGRATION_DIR = PROJECT_ROOT / "scripts" / "integration"
if str(INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_DIR))

from scripts.integration.replay_runtime_feature_adapter_600mw import (
    ReplayRuntimeFeatureAdapter600MW,
)
from scripts.database.postgres_storage import PostgreSQLStorage


# Preserve the existing local MQTT and FastAPI configuration.
MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "ai_scada/scada/600mw"
FASTAPI_URL = "http://127.0.0.1:8000/forecast"
HTTP_TIMEOUT_SECONDS = 10
EXPECTED_FEATURE_COUNT = 119
FORECAST_KEYS = ("power_2min", "power_10min", "power_30min")
FORECAST_HORIZONS = {
    "power_2min": 2,
    "power_10min": 10,
    "power_30min": 30,
}
SOURCE_ID = 1
EQUIPMENT_ID = 3
POWER_PARAMETER_ID = 1
MODEL_NAME = "600MW Power Output Forecasting"
PREDICTION_TYPE = "power_forecast"
PREDICTION_DESCRIPTION = "MQTT SCADA replay forecast"


class ScadaAIMqttSubscriber:
    def __init__(self, max_messages: int | None = None) -> None:
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages must be at least 1")

        self.max_messages = max_messages
        self.adapter = ReplayRuntimeFeatureAdapter600MW()
        self.expected_features = list(self.adapter.expected_features)
        self.messages_received = 0
        self.not_ready_count = 0
        self.ready_count = 0
        self.invalid_count = 0
        self.feature_validation_failures = 0
        self.forecast_requests = 0
        self.http_successes = 0
        self.http_failures = 0
        self.forecast_successes = 0
        self.forecast_failures = 0
        self.storage = PostgreSQLStorage()
        self.measurement_writes = 0
        self.prediction_writes = 0
        self.measurement_write_failures = 0
        self.prediction_write_failures = 0
        self.persistence_failures = 0
        self.adapter_failure_kinds = Counter()
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="ai_scada_ai_subscriber",
        )
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        self.client.on_disconnect = self.on_disconnect

    def on_subscribe(self, client, userdata, mid, reason_codes, properties) -> None:
        client_id = getattr(client, "_client_id", None)
        if isinstance(client_id, bytes):
            client_id = client_id.decode("utf-8", errors="replace")
        if client_id is None:
            client_id = "<unavailable>"

        granted = [
            f"{getattr(code, 'value', code)} ({code})"
            for code in reason_codes
        ]
        accepted = bool(reason_codes) and all(
            not getattr(code, "is_failure", False) for code in reason_codes
        )
        print(
            f"MQTT subscription acknowledgment: client_id={client_id}, mid={mid}, "
            f"granted_qos/reason_codes={granted}, broker_accepted={accepted}",
            flush=True,
        )

    def on_disconnect(
        self, client, userdata, disconnect_flags, reason_code, properties
    ) -> None:
        client_id = getattr(client, "_client_id", None)
        if isinstance(client_id, bytes):
            client_id = client_id.decode("utf-8", errors="replace")
        if client_id is None:
            client_id = "<unavailable>"

        reason_value = getattr(reason_code, "value", reason_code)
        normal = reason_value == 0
        from_server = getattr(
            disconnect_flags, "is_disconnect_packet_from_server", None
        )
        if from_server is True:
            disconnect_origin = "broker/server DISCONNECT packet"
        elif from_server is False:
            disconnect_origin = "no broker DISCONNECT packet (client or transport)"
        else:
            disconnect_origin = "unknown"

        print(
            f"MQTT disconnected: client_id={client_id}, "
            f"classification={'normal' if normal else 'unexpected'}, "
            f"reason_code={reason_value} ({reason_code}), "
            f"disconnect_origin={disconnect_origin}, "
            f"disconnect_flags={disconnect_flags!r}, properties={properties!r}",
            flush=True,
        )

    def on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if getattr(reason_code, "is_failure", False):
            print(f"MQTT connection rejected: {reason_code}", flush=True)
            return

        result, _ = client.subscribe(MQTT_TOPIC, qos=0)
        if result != mqtt.MQTT_ERR_SUCCESS:
            print(f"MQTT subscribe failed with code {result}", flush=True)
            client.disconnect()
            return
        print(
            f"Connected to MQTT broker {MQTT_HOST}:{MQTT_PORT}; "
            f"subscribed to {MQTT_TOPIC}.",
            flush=True,
        )

    def on_message(self, client, userdata, mqtt_message) -> None:
        self.messages_received += 1
        message_number = self.messages_received
        step = "?"
        timestamp = "?"
        source = "?"

        try:
            payload = json.loads(mqtt_message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.invalid_count += 1
            self.adapter.reset()
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                f"invalid MQTT JSON: {exc}",
            )
            self._stop_at_limit(client)
            return

        if isinstance(payload, dict):
            timestamp = payload.get("timestamp", "?")
            source = payload.get("source", "?")
            step = payload.get("step", "?")

        try:
            # The adapter receives the MQTT envelope and selects only its
            # authoritative RAW_SOURCE names before feature construction.
            result = self.adapter.process_message(payload)
        except Exception as exc:
            self.invalid_count += 1
            self.adapter.reset()
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                f"runtime adapter raised {type(exc).__name__}: {exc}",
            )
            self._stop_at_limit(client)
            return

        if result.status == "NOT_READY":
            self.not_ready_count += 1
            print(
                f"message={message_number} step={step} source={source} "
                f"status=NOT_READY; forecast skipped ({result.reason})",
                flush=True,
            )
        elif result.status == "INVALID_INPUT":
            self.invalid_count += 1
            if result.failure_kind:
                self.adapter_failure_kinds[result.failure_kind] += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                f"adapter rejected MQTT message "
                f"[{result.failure_kind or 'invalid_input'}]: {result.reason}",
            )
        elif result.status == "READY":
            self.ready_count += 1
            self._forecast_generated_row(
                message_number=message_number,
                step=step,
                timestamp=timestamp,
                source=source,
                features=result.features,
                payload=payload,
                client=client,
            )
        else:
            self.invalid_count += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                f"unrecognized runtime adapter status {result.status!r}",
            )

        self._stop_at_limit(client)

    def _forecast_generated_row(
        self,
        *,
        message_number: int,
        step: object,
        timestamp: object,
        source: object,
        features: object,
        payload: object,
        client,
    ) -> None:
        if not hasattr(features, "columns") or not hasattr(features, "iloc"):
            self.feature_validation_failures += 1
            self.forecast_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                "adapter READY output is not a tabular feature row",
            )
            return

        columns = list(features.columns)
        if (
            getattr(features, "shape", None) != (1, EXPECTED_FEATURE_COUNT)
            or columns != self.expected_features
        ):
            self.feature_validation_failures += 1
            self.forecast_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                "generated output does not match the exact 119-feature "
                f"manifest order (shape={getattr(features, 'shape', None)}, "
                f"feature_count={len(columns)})",
            )
            return

        try:
            values = features.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError, OverflowError) as exc:
            self.feature_validation_failures += 1
            self.forecast_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                f"generated features are not numeric: {exc}",
            )
            return

        nan_count = int(np.isnan(values).sum())
        infinity_count = int(np.isinf(values).sum())
        if nan_count or infinity_count:
            self.feature_validation_failures += 1
            self.forecast_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                "generated features contain invalid values "
                f"(NaN={nan_count}, infinity={infinity_count})",
            )
            return

        # Construct the FastAPI request exclusively from the adapter-generated
        # columns in frozen manifest order. Do not forward payload['features'].
        generated_feature_dict = {
            feature: float(features.iloc[0][feature])
            for feature in self.expected_features
        }
        generated_feature_count = len(generated_feature_dict)
        if generated_feature_count != EXPECTED_FEATURE_COUNT:
            self.feature_validation_failures += 1
            self.forecast_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                f"outgoing feature count is {generated_feature_count}, "
                f"expected {EXPECTED_FEATURE_COUNT}",
            )
            return

        self.forecast_requests += 1
        http_status = None
        try:
            response = requests.post(
                FASTAPI_URL,
                json={"features": generated_feature_dict},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            http_status = response.status_code
        except requests.RequestException as exc:
            self.http_failures += 1
            self.forecast_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                f"generated_features={generated_feature_count} "
                f"HTTP status=unavailable; request failed: {exc}",
            )
            return

        if http_status != 200:
            self.http_failures += 1
            self.forecast_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                f"generated_features={generated_feature_count} "
                f"expected HTTP 200; received {http_status}; "
                f"response={response.text[:800]}",
            )
            return

        self.http_successes += 1
        try:
            result = response.json()
            if not isinstance(result, dict) or result.get("status") != "success":
                raise ValueError(
                    f"unexpected API response status: "
                    f"{result.get('status') if isinstance(result, dict) else type(result).__name__}"
                )

            forecast = result.get("forecast")
            if not isinstance(forecast, dict):
                raise ValueError("API response is missing forecast object")

            normalized_forecast = {}
            for key in FORECAST_KEYS:
                if key not in forecast:
                    raise ValueError(f"forecast response is missing {key}")
                value = forecast[key]
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise TypeError(f"{key} forecast is not numeric: {value!r}")
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise ValueError(f"{key} forecast is not finite: {numeric_value!r}")
                normalized_forecast[key] = numeric_value
        except (ValueError, TypeError, requests.RequestException) as exc:
            self.forecast_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                f"generated_features={generated_feature_count} "
                f"HTTP status={http_status}; invalid forecast response: {exc}",
            )
            return

        self.forecast_successes += 1
        print(
            f"message={message_number} step={step} source={source} "
            f"status=READY generated_features={generated_feature_count} "
            f"HTTP status={http_status} forecast_success=true "
            f"forecast={normalized_forecast}",
            flush=True,
        )
        self._persist_successful_forecast(
            message_number=message_number,
            step=step,
            timestamp=timestamp,
            source=source,
            payload=payload,
            forecasts=normalized_forecast,
            client=client,
        )

    def _persist_successful_forecast(
        self,
        *,
        message_number: int,
        step: object,
        timestamp: object,
        source: object,
        payload: object,
        forecasts: dict[str, float],
        client,
    ) -> None:
        """Persist the original raw power value and this request's forecasts."""
        try:
            if not isinstance(payload, dict):
                raise TypeError("MQTT payload must be an object")

            storage_timestamp = datetime.fromisoformat(payload["timestamp"])
            source_features = payload["features"]
            if not isinstance(source_features, dict):
                raise TypeError("MQTT payload features must be an object")

            raw_power_value = source_features["Power output\n（MW）"]
            if isinstance(raw_power_value, bool):
                raise TypeError("raw Power output value must be numeric")
            raw_power_value = float(raw_power_value)
            if not math.isfinite(raw_power_value):
                raise ValueError("raw Power output value must be finite")
        except Exception as exc:
            self.measurement_write_failures += 1
            self.persistence_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                "PostgreSQL measurement write FAILED before insert: "
                f"{type(exc).__name__}: {exc}; stopping subscriber",
            )
            client.disconnect()
            return

        try:
            measurement_id = self.storage.save_measurement(
                timestamp=storage_timestamp,
                source_id=SOURCE_ID,
                equipment_id=EQUIPMENT_ID,
                parameter_id=POWER_PARAMETER_ID,
                value=raw_power_value,
                quality_status="GOOD",
            )
        except Exception as exc:
            self.measurement_write_failures += 1
            self.persistence_failures += 1
            self._log_error(
                message_number,
                step,
                timestamp,
                source,
                "PostgreSQL measurement write FAILED: "
                f"{type(exc).__name__}: {exc}; stopping subscriber",
            )
            client.disconnect()
            return

        self.measurement_writes += 1
        print(
            f"message={message_number} step={step} "
            f"postgres_measurement=stored measurement_id={measurement_id} "
            f"source_id={SOURCE_ID} equipment_id={EQUIPMENT_ID} "
            f"parameter_id={POWER_PARAMETER_ID} value={raw_power_value} "
            f"timestamp={storage_timestamp.isoformat()} quality=GOOD",
            flush=True,
        )

        stored_predictions_for_message = 0
        for prediction_name, horizon in FORECAST_HORIZONS.items():
            forecast_value = forecasts[prediction_name]
            try:
                prediction_id = self.storage.save_prediction(
                    timestamp=storage_timestamp,
                    source_id=SOURCE_ID,
                    equipment_id=EQUIPMENT_ID,
                    model_name=MODEL_NAME,
                    prediction_type=PREDICTION_TYPE,
                    predicted_value=forecast_value,
                    confidence=None,
                    risk_level=None,
                    prediction_horizon_minutes=horizon,
                    description=PREDICTION_DESCRIPTION,
                )
            except Exception as exc:
                self.prediction_write_failures += 1
                self.persistence_failures += 1
                self._log_error(
                    message_number,
                    step,
                    timestamp,
                    source,
                    "PostgreSQL prediction write FAILED for "
                    f"{prediction_name} horizon={horizon}: "
                    f"{type(exc).__name__}: {exc}; "
                    f"predictions_stored_for_message="
                    f"{stored_predictions_for_message}; "
                    "stopping subscriber",
                )
                client.disconnect()
                return

            self.prediction_writes += 1
            stored_predictions_for_message += 1
            print(
                f"message={message_number} step={step} "
                f"postgres_prediction=stored prediction_id={prediction_id} "
                f"name={prediction_name} horizon={horizon} "
                f"value={forecast_value} timestamp={storage_timestamp.isoformat()}",
                flush=True,
            )

    @staticmethod
    def _log_error(
        message_number: int,
        step: object,
        timestamp: object,
        source: object,
        error: str,
    ) -> None:
        print(
            f"ERROR message={message_number} step={step} timestamp={timestamp} "
            f"source={source}: {error}",
            flush=True,
        )

    def _stop_at_limit(self, client) -> None:
        if self.max_messages is not None and self.messages_received >= self.max_messages:
            print(
                f"Reached validation message limit {self.max_messages}; stopping.",
                flush=True,
            )
            client.disconnect()

    def print_summary(self) -> None:
        print("\nPRODUCTION AI MQTT SUBSCRIBER SUMMARY", flush=True)
        print(f"messages_received={self.messages_received}", flush=True)
        print(f"not_ready={self.not_ready_count}", flush=True)
        print(f"ready={self.ready_count}", flush=True)
        print(f"invalid={self.invalid_count}", flush=True)
        print(f"feature_validation_failures={self.feature_validation_failures}", flush=True)
        print(f"forecast_requests={self.forecast_requests}", flush=True)
        print(f"http_successes={self.http_successes}", flush=True)
        print(f"http_failures={self.http_failures}", flush=True)
        print(f"successful_forecasts={self.forecast_successes}", flush=True)
        print(f"forecast_failures={self.forecast_failures}", flush=True)
        print(f"measurement_writes={self.measurement_writes}", flush=True)
        print(f"prediction_writes={self.prediction_writes}", flush=True)
        print(
            f"measurement_write_failures={self.measurement_write_failures}",
            flush=True,
        )
        print(
            f"prediction_write_failures={self.prediction_write_failures}",
            flush=True,
        )
        print(f"persistence_failures={self.persistence_failures}", flush=True)
        print(f"adapter_failure_kinds={dict(self.adapter_failure_kinds)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="optional message limit for validation; default is continuous operation",
    )
    args = parser.parse_args()

    subscriber = ScadaAIMqttSubscriber(max_messages=args.max_messages)

    print("=" * 70)
    print("AI_SCADA MQTT -> RUNTIME FEATURES -> FASTAPI FORECAST SUBSCRIBER")
    print("=" * 70)
    print(f"MQTT broker : {MQTT_HOST}:{MQTT_PORT}")
    print(f"MQTT topic  : {MQTT_TOPIC}")
    print(f"FastAPI URL : {FASTAPI_URL}")
    print("Replay mode : HISTORICAL 600 MW REPLAY; not live SCADA")
    if args.max_messages is not None:
        print(f"Message limit: {args.max_messages}")

    try:
        subscriber.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        subscriber.client.loop_forever()
    except KeyboardInterrupt:
        print("\nSubscriber stopped by user.", flush=True)
    except Exception as exc:
        print(f"Subscriber stopped with error: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        try:
            subscriber.client.disconnect()
        except Exception:
            pass
        subscriber.print_summary()
        print("MQTT connection closed.", flush=True)


if __name__ == "__main__":
    main()
