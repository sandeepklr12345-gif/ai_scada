"""Validate MQTT replay through runtime features and the existing FastAPI.

This test subscribes only. Start FastAPI and the replay publisher separately.

From the repository root:
  Terminal 1: PostgreSQL is not required by /forecast (it does not connect to
              PostgreSQL); start it only if another API startup dependency
              in your environment requires it.
  Terminal 2:
    .\\start_api.ps1
  Terminal 3:
    python scripts/mqtt/test_600mw_runtime_fastapi_mqtt.py --max-ready 10
  Terminal 4 (after Terminal 3 reports subscribed):
    python scripts/mqtt/scada_replay_publisher.py

The existing start_api.ps1 activates the WSL rapids-gpu environment and runs:
  python -m uvicorn scripts.api.main:app --host 0.0.0.0 --port 8000

This subscriber never starts FastAPI or the publisher, publishes MQTT, writes
PostgreSQL, or invokes forecasting directly. Forecasting happens only through
the existing POST /forecast endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_mqtt_runtime_fastapi_validation.txt"
)
MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "ai_scada/scada/600mw"
FASTAPI_URL = "http://127.0.0.1:8000/forecast"
DEFAULT_MAX_READY = 10
EXPECTED_FEATURE_COUNT = 119
FORECAST_KEYS = ("power_2min", "power_10min", "power_30min")
HTTP_TIMEOUT_SECONDS = 15


class MqttRuntimeFastApiValidation:
    def __init__(self, max_ready: int) -> None:
        if max_ready < 1:
            raise ValueError("max_ready must be at least 1")

        self.max_ready = max_ready
        self.adapter = ReplayRuntimeFeatureAdapter600MW()
        self.expected_features = list(self.adapter.expected_features)

        self.messages_received = 0
        self.not_ready_count = 0
        self.ready_count = 0
        self.invalid_count = 0
        self.invalid_errors: list[str] = []
        self.adapter_failure_kinds = Counter()
        self.ordering_failures = 0
        self.cadence_failures = 0

        self.generated_feature_counts = Counter()
        self.feature_validation_failures = 0
        self.feature_nan_count = 0
        self.feature_infinity_count = 0
        self.api_requests_sent = 0
        self.http_successes = 0
        self.http_failures = 0
        self.forecast_successes = 0
        self.forecast_failures = 0
        self.api_errors: list[str] = []
        self.ready_row_results: list[dict] = []
        self.first_ready_message: Optional[int] = None
        self.first_forecast_success_message: Optional[int] = None

        self.connection_error: Optional[str] = None
        self.interrupted = False
        self.finished = threading.Event()
        self.subscribed = threading.Event()
        self.loop_started = False
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="ai_scada_600mw_runtime_fastapi_validation",
        )
        self.client.on_connect = self.on_connect
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if getattr(reason_code, "is_failure", False):
            self.connection_error = f"MQTT connection rejected: {reason_code}"
            self.finished.set()
            return

        result, message_id = client.subscribe(MQTT_TOPIC, qos=0)
        if result != mqtt.MQTT_ERR_SUCCESS:
            self.connection_error = f"MQTT subscribe failed with code {result}"
            self.finished.set()
            return
        self.pending_subscription_id = message_id

    def on_subscribe(
        self, client, userdata, message_id, reason_codes, properties
    ) -> None:
        failures = [
            str(reason)
            for reason in reason_codes
            if getattr(reason, "is_failure", False)
        ]
        if failures:
            self.connection_error = (
                "MQTT subscription rejected: " + ", ".join(failures)
            )
            self.finished.set()
            return

        self.subscribed.set()
        print(
            f"Subscribed to {MQTT_TOPIC} at {MQTT_HOST}:{MQTT_PORT}; "
            f"waiting for {self.max_ready} READY row(s).",
            flush=True,
        )

    def on_message(self, client, userdata, mqtt_message) -> None:
        self.messages_received += 1
        message_number = self.messages_received

        try:
            message = json.loads(mqtt_message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.invalid_count += 1
            self.invalid_errors.append(
                f"message {message_number}: invalid JSON: {exc}"
            )
            self._finish_if_ready_limit(client)
            return

        try:
            result = self.adapter.process_message(message)
        except Exception as exc:
            self.invalid_count += 1
            self.invalid_errors.append(
                f"message {message_number}: adapter raised "
                f"{type(exc).__name__}: {exc}"
            )
            self._finish_if_ready_limit(client)
            return

        step = message.get("step") if isinstance(message, dict) else None
        if result.status == "NOT_READY":
            self.not_ready_count += 1
            print(f"#{message_number} step={step} status=NOT_READY", flush=True)
        elif result.status == "INVALID_INPUT":
            self.invalid_count += 1
            self.invalid_errors.append(
                f"message {message_number} step={step}: "
                f"{result.reason or 'invalid adapter input'}"
            )
            if result.failure_kind:
                self.adapter_failure_kinds[result.failure_kind] += 1
            self.ordering_failures += result.failure_kind == "timestamp_order"
            self.cadence_failures += result.failure_kind == "cadence"
            print(f"#{message_number} step={step} status=INVALID_INPUT", flush=True)
        elif result.status == "READY":
            self.ready_count += 1
            if self.first_ready_message is None:
                self.first_ready_message = message_number
            self._send_generated_features(
                message_number=message_number,
                step=step,
                timestamp=result.timestamp,
                generated_features=result.features,
            )
        else:
            self.invalid_count += 1
            self.invalid_errors.append(
                f"message {message_number} step={step}: "
                f"unrecognized adapter status {result.status!r}"
            )
            print(f"#{message_number} step={step} status=INVALID_INPUT", flush=True)

        self._finish_if_ready_limit(client)

    def _send_generated_features(
        self,
        *,
        message_number: int,
        step: object,
        timestamp: object,
        generated_features: object,
    ) -> None:
        record = {
            "message": message_number,
            "step": step,
            "timestamp": str(timestamp),
            "generated_feature_count": 0,
            "http_status": None,
            "forecast_response": None,
            "api_error": None,
        }
        self.ready_row_results.append(record)

        if not isinstance(generated_features, pd.DataFrame):
            self.feature_validation_failures += 1
            self.forecast_failures += 1
            self._set_api_error(
                record,
                "adapter did not return generated features as a DataFrame",
            )
            return

        columns = list(generated_features.columns)
        record["generated_feature_count"] = len(columns)
        self.generated_feature_counts[len(columns)] += 1
        missing = set(self.expected_features) - set(columns)
        unexpected = set(columns) - set(self.expected_features)
        if (
            generated_features.shape != (1, EXPECTED_FEATURE_COUNT)
            or columns != self.expected_features
            or missing
            or unexpected
        ):
            self.feature_validation_failures += 1
            self.forecast_failures += 1
            self._set_api_error(
                record,
                "generated feature schema is not exactly the expected "
                f"{EXPECTED_FEATURE_COUNT}-feature manifest order "
                f"(shape={generated_features.shape}, missing={len(missing)}, "
                f"unexpected={len(unexpected)})",
            )
            return

        try:
            values = generated_features.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError, OverflowError) as exc:
            self.feature_validation_failures += 1
            self.forecast_failures += 1
            self._set_api_error(record, f"generated features are not numeric: {exc}")
            return

        self.feature_nan_count += int(np.isnan(values).sum())
        self.feature_infinity_count += int(np.isinf(values).sum())
        if np.isnan(values).any() or np.isinf(values).any():
            self.feature_validation_failures += 1
            self.forecast_failures += 1
            self._set_api_error(record, "generated features contain NaN or infinity")
            return

        # Build the HTTP body only from the adapter-produced manifest columns.
        # No MQTT replay features other than those generated output values are sent.
        feature_dict = {
            feature: float(generated_features.iloc[0][feature])
            for feature in self.expected_features
        }
        record["generated_feature_count"] = len(feature_dict)
        record["outgoing_feature_count"] = len(feature_dict)

        self.api_requests_sent += 1
        try:
            response = requests.post(
                FASTAPI_URL,
                json={"features": feature_dict},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            record["http_status"] = response.status_code
        except requests.RequestException as exc:
            self.http_failures += 1
            self.forecast_failures += 1
            self._set_api_error(record, f"HTTP request failed: {exc}")
            self._print_ready_result(record)
            return

        if 200 <= response.status_code < 300:
            self.http_successes += 1
        else:
            self.http_failures += 1
            self.forecast_failures += 1
            body = response.text[:800]
            self._set_api_error(
                record,
                f"HTTP {response.status_code}: {body or 'empty response body'}",
            )
            self._print_ready_result(record)
            return

        try:
            response_data = response.json()
            if not isinstance(response_data, dict):
                raise ValueError("response JSON is not an object")
            if response_data.get("status") != "success":
                raise ValueError(
                    f"response status was {response_data.get('status')!r}"
                )
            forecast = response_data.get("forecast")
            if not isinstance(forecast, dict):
                raise ValueError("response field 'forecast' is not an object")

            missing_horizons = [
                key for key in FORECAST_KEYS if key not in forecast
            ]
            if missing_horizons:
                raise ValueError(
                    "forecast response is missing horizon(s): "
                    + ", ".join(missing_horizons)
                )

            normalized_forecast = {}
            for key in FORECAST_KEYS:
                value = forecast[key]
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise TypeError(f"{key} is not numeric: {value!r}")
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise ValueError(f"{key} is not finite: {numeric_value!r}")
                normalized_forecast[key] = numeric_value

            record["forecast_response"] = {
                "status": "success",
                "forecast": normalized_forecast,
            }
            self.forecast_successes += 1
            if self.first_forecast_success_message is None:
                self.first_forecast_success_message = message_number
        except (ValueError, TypeError, requests.RequestException) as exc:
            self.forecast_failures += 1
            self._set_api_error(record, f"Invalid forecast response: {exc}")

        self._print_ready_result(record)

    def _set_api_error(self, record: dict, error: str) -> None:
        record["api_error"] = error
        self.api_errors.append(
            f"message {record['message']} step={record['step']}: {error}"
        )

    @staticmethod
    def _print_ready_result(record: dict) -> None:
        print(
            f"#{record['message']} step={record['step']} READY "
            f"features={record['generated_feature_count']} "
            f"http={record['http_status']} "
            f"forecast={record['forecast_response']} "
            f"error={record['api_error']}",
            flush=True,
        )

    def _finish_if_ready_limit(self, client) -> None:
        if self.ready_count >= self.max_ready:
            self.finished.set()
            client.disconnect()

    def run(self) -> dict:
        try:
            self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self.client.loop_start()
            self.loop_started = True

            subscribe_deadline = time.monotonic() + 10.0
            while not self.subscribed.is_set() and not self.finished.is_set():
                if time.monotonic() >= subscribe_deadline:
                    self.connection_error = (
                        "Timed out waiting for MQTT connection/subscription acknowledgement"
                    )
                    self.finished.set()
                    break
                self.finished.wait(0.1)

            if self.subscribed.is_set():
                self.finished.wait()
        except KeyboardInterrupt:
            self.interrupted = True
            self.finished.set()
        except Exception as exc:
            self.connection_error = f"{type(exc).__name__}: {exc}"
            self.finished.set()
        finally:
            if self.loop_started:
                self.client.loop_stop()
            try:
                self.client.disconnect()
            except Exception:
                pass

        return self.summary()

    def summary(self) -> dict:
        ready_target_reached = self.ready_count == self.max_ready
        feature_counts_valid = (
            self.ready_count == len(self.ready_row_results)
            and all(
                count == EXPECTED_FEATURE_COUNT
                for count in self.generated_feature_counts
            )
            and self.feature_validation_failures == 0
            and self.feature_nan_count == 0
            and self.feature_infinity_count == 0
        )
        request_accounting_valid = (
            self.api_requests_sent
            == self.http_successes + self.http_failures
            and self.api_requests_sent
            == self.forecast_successes
            + sum(
                1
                for row in self.ready_row_results
                if row["outgoing_feature_count"]
                and row["api_error"] is not None
            )
        )
        passed = (
            self.connection_error is None
            and not self.interrupted
            and ready_target_reached
            and self.invalid_count == 0
            and self.ordering_failures == 0
            and self.cadence_failures == 0
            and feature_counts_valid
            and request_accounting_valid
            and self.api_requests_sent == self.max_ready
            and self.http_successes == self.max_ready
            and self.http_failures == 0
            and self.forecast_successes == self.max_ready
            and self.forecast_failures == 0
            and len(self.api_errors) == 0
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "messages_received": self.messages_received,
            "not_ready_count": self.not_ready_count,
            "ready_count": self.ready_count,
            "invalid_count": self.invalid_count,
            "invalid_errors": self.invalid_errors,
            "adapter_failure_kinds": dict(self.adapter_failure_kinds),
            "ordering_failures": self.ordering_failures,
            "cadence_failures": self.cadence_failures,
            "generated_feature_counts": dict(self.generated_feature_counts),
            "feature_validation_failures": self.feature_validation_failures,
            "feature_nan_count": self.feature_nan_count,
            "feature_infinity_count": self.feature_infinity_count,
            "api_requests_sent": self.api_requests_sent,
            "http_successes": self.http_successes,
            "http_failures": self.http_failures,
            "forecast_successes": self.forecast_successes,
            "forecast_failures": self.forecast_failures,
            "api_errors": self.api_errors,
            "ready_row_results": self.ready_row_results,
            "first_ready_message": self.first_ready_message,
            "first_forecast_success_message": self.first_forecast_success_message,
            "request_accounting_valid": request_accounting_valid,
            "connection_error": self.connection_error,
            "interrupted": self.interrupted,
        }

    def write_report(self, summary: dict) -> None:
        counts = summary["generated_feature_counts"]
        feature_count_text = (
            ", ".join(
                f"{count} features: {rows} READY row(s)"
                for count, rows in sorted(counts.items())
            )
            if counts
            else "No READY features generated"
        )
        lines = [
            "600 MW MQTT RUNTIME → FASTAPI VALIDATION",
            "=" * 43,
            "",
            "Data classification: HISTORICAL REPLAY; not live SCADA.",
            f"MQTT broker: {MQTT_HOST}:{MQTT_PORT}",
            f"MQTT topic: {MQTT_TOPIC}",
            f"FastAPI endpoint: POST {FASTAPI_URL}",
            f"READY row target: {self.max_ready}",
            "Request schema: {\"features\": {feature_name: numeric_value, ...}}",
            "Response schema: {\"status\": \"success\", \"forecast\": "
            "{\"power_2min\": number, \"power_10min\": number, "
            "\"power_30min\": number}}",
            "Outgoing fields are built only from adapter-generated runtime features.",
            "The /forecast handler runs Forecasting600MW and does not access PostgreSQL.",
            "The API response does not report input feature count; the subscriber "
            "validates the outgoing request locally before sending.",
            f"FINAL RESULT: {summary['status']}",
            "",
            "MQTT counts",
            "------------",
            f"Messages received: {summary['messages_received']}",
            f"NOT_READY: {summary['not_ready_count']}",
            f"READY: {summary['ready_count']}",
            f"Invalid: {summary['invalid_count']}",
            f"Ordering failures: {summary['ordering_failures']}",
            f"Cadence failures: {summary['cadence_failures']}",
            f"First READY message: {summary['first_ready_message']}",
            "",
            "Feature and HTTP counts",
            "------------------------",
            f"Generated feature counts: {feature_count_text}",
            f"Feature validation failures: {summary['feature_validation_failures']}",
            f"Generated feature NaN count: {summary['feature_nan_count']}",
            f"Generated feature infinity count: {summary['feature_infinity_count']}",
            f"API requests sent: {summary['api_requests_sent']}",
            f"HTTP successes: {summary['http_successes']}",
            f"HTTP failures: {summary['http_failures']}",
            f"Forecast successes: {summary['forecast_successes']}",
            f"Forecast failures: {summary['forecast_failures']}",
            "First successful forecast message: "
            f"{summary['first_forecast_success_message']}",
            f"Request accounting valid: {summary['request_accounting_valid']}",
            f"MQTT connection error: {summary['connection_error'] or 'None'}",
            "",
            "Per-READY-row API results",
            "-------------------------",
        ]
        if summary["ready_row_results"]:
            for row in summary["ready_row_results"]:
                response_text = (
                    json.dumps(
                        row["forecast_response"],
                        sort_keys=True,
                        allow_nan=False,
                    )
                    if row["forecast_response"] is not None
                    else "None"
                )
                lines.extend(
                    [
                        f"message={row['message']} step={row['step']} "
                        f"timestamp={row['timestamp']}",
                        f"  generated_feature_count={row['generated_feature_count']} "
                        f"outgoing_feature_count={row.get('outgoing_feature_count', 0)} "
                        f"http_status={row['http_status']}",
                        f"  forecast_response={response_text}",
                        f"  api_error={row['api_error'] or 'None'}",
                    ]
                )
        else:
            lines.append("No READY rows received.")

        lines.extend(["", "API errors", "----------"])
        if summary["api_errors"]:
            lines.extend(f"- {error}" for error in summary["api_errors"])
        else:
            lines.append("None")
        if summary["invalid_errors"]:
            lines.extend(["", "Invalid MQTT messages", "---------------------"])
            lines.extend(f"- {error}" for error in summary["invalid_errors"])

        lines.extend(
            [
                "",
                "Startup and execution commands (repository root)",
                "-------------------------------------------------",
                "Terminal 1 (PostgreSQL): not required by the /forecast route; "
                "start only if your environment has another PostgreSQL dependency.",
                "Terminal 2 (existing repository startup script):",
                "  .\\start_api.ps1",
                "This script activates the WSL rapids-gpu environment and runs:",
                "  python -m uvicorn scripts.api.main:app --host 0.0.0.0 --port 8000",
                "Terminal 3 (start this waiting subscriber):",
                "  python scripts/mqtt/test_600mw_runtime_fastapi_mqtt.py --max-ready 10",
                "Terminal 4 (run separately after Terminal 3 reports subscribed):",
                "  python scripts/mqtt/scada_replay_publisher.py",
                "The subscriber does not start FastAPI or the publisher.",
                "",
            ]
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    def _finish_if_ready_limit(self, client) -> None:
        if self.ready_count >= self.max_ready:
            self.finished.set()
            client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-ready",
        type=int,
        default=DEFAULT_MAX_READY,
        help=f"stop after this many READY rows (default: {DEFAULT_MAX_READY})",
    )
    args = parser.parse_args()

    check = MqttRuntimeFastApiValidation(args.max_ready)
    print(
        f"Waiting on {MQTT_HOST}:{MQTT_PORT} topic={MQTT_TOPIC}; "
        f"FastAPI={FASTAPI_URL}. This script does not start either service.",
        flush=True,
    )
    summary = check.run()
    check.write_report(summary)

    print("\nFINAL RESULT:", flush=True)
    print(summary["status"], flush=True)
    for key in (
        "messages_received",
        "not_ready_count",
        "ready_count",
        "invalid_count",
        "generated_feature_counts",
        "feature_validation_failures",
        "feature_nan_count",
        "feature_infinity_count",
        "api_requests_sent",
        "http_successes",
        "http_failures",
        "forecast_successes",
        "forecast_failures",
        "first_ready_message",
        "first_forecast_success_message",
        "ordering_failures",
        "cadence_failures",
    ):
        print(f"{key}: {summary[key]}", flush=True)
    for row in summary["ready_row_results"]:
        print(
            f"message={row['message']} step={row['step']} "
            f"features={row['generated_feature_count']} "
            f"http={row['http_status']} forecast={row['forecast_response']} "
            f"error={row['api_error']}",
            flush=True,
        )
    if summary["api_errors"]:
        print("API errors:", flush=True)
        for error in summary["api_errors"]:
            print(f"  {error}", flush=True)
    print(f"report: {REPORT_PATH}", flush=True)

    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
