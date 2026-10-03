"""Validate MQTT replay through runtime features and the frozen 600 MW models.

This test subscribes only. Start the historical replay publisher separately
after this script reports that the subscription is active.

From the repository root:
  Terminal 1 (only if no broker is already listening on 127.0.0.1:1883):
    mosquitto -v
  Terminal 2:
    python scripts/mqtt/test_600mw_runtime_forecasting_mqtt.py --max-messages 20
  Terminal 3 (start separately after Terminal 2 reports subscribed):
    python scripts/mqtt/scada_replay_publisher.py

The subscriber does not publish MQTT, call FastAPI, write PostgreSQL, run
anomaly detection, or invoke decision support.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
INTEGRATION_DIR = PROJECT_ROOT / "scripts" / "integration"
if str(INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_DIR))

from scripts.inference.forecasting_600mw import Forecasting600MW
from scripts.integration.replay_runtime_feature_adapter_600mw import (
    ReplayRuntimeFeatureAdapter600MW,
)


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_mqtt_runtime_forecasting_validation.txt"
)
MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "ai_scada/scada/600mw"
DEFAULT_MAX_MESSAGES = 20
EXPECTED_FEATURE_COUNT = 119
FORECAST_COLUMNS = ("power_2min", "power_10min", "power_30min")


class MqttRuntimeForecastValidation:
    def __init__(self, max_messages: int) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")

        self.max_messages = max_messages
        self.messages_received = 0
        self.not_ready_messages = 0
        self.ready_messages = 0
        self.invalid_messages = 0
        self.invalid_reasons: list[str] = []
        self.adapter_failure_kinds = Counter()
        self.ordering_failures = 0
        self.cadence_failures = 0

        self.generated_feature_counts = Counter()
        self.generated_feature_rows = 0
        self.feature_validation_failures = 0
        self.manifest_order_checks = 0
        self.manifest_order_failures = 0
        self.missing_features = 0
        self.unexpected_features = 0
        self.input_nan_count = 0
        self.input_infinite_count = 0
        self.model_input_validation_failures = 0

        self.successful_inferences = 0
        self.inference_failures = 0
        self.model_accepted_inputs = 0
        self.forecast_output_validation_failures = 0
        self.inference_errors: list[str] = []
        self.forecast_outputs: list[dict] = []
        self.first_successful_forecast_message: Optional[int] = None
        self.model_input_feature_counts = Counter()

        self.connection_error: Optional[str] = None
        self.initialization_error: Optional[str] = None
        self.interrupted = False
        self.finished = threading.Event()
        self.subscribed = threading.Event()
        self.loop_started = False
        self.adapter = None
        self.forecaster = None
        self.expected_features: list[str] = []
        self.forecaster_features: list[str] = []

        try:
            self.adapter = ReplayRuntimeFeatureAdapter600MW()
            self.forecaster = Forecasting600MW()
            self.expected_features = list(self.adapter.expected_features)
            self.forecaster_features = list(self.forecaster.features)
        except Exception as exc:
            self.initialization_error = f"{type(exc).__name__}: {exc}"

        try:
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id="ai_scada_600mw_runtime_forecasting_validation",
            )
            self.client.on_connect = self.on_connect
            self.client.on_subscribe = self.on_subscribe
            self.client.on_message = self.on_message
        except Exception as exc:
            self.client = None
            self.initialization_error = (
                self.initialization_error
                or f"MQTT client initialization: {type(exc).__name__}: {exc}"
            )

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
            f"waiting for {self.max_messages} message(s).",
            flush=True,
        )

    def _record_invalid(self, reason: str) -> None:
        self.invalid_messages += 1
        self.invalid_reasons.append(reason)

    def _record_feature_failure(self, reason: str) -> None:
        self.feature_validation_failures += 1
        self.invalid_reasons.append(f"feature validation: {reason}")

    def on_message(self, client, userdata, mqtt_message) -> None:
        self.messages_received += 1
        message_number = self.messages_received

        try:
            message = json.loads(mqtt_message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._record_invalid(f"message {message_number}: invalid JSON: {exc}")
            self._finish_at_limit(client)
            return

        try:
            result = self.adapter.process_message(message)
        except Exception as exc:
            self._record_invalid(
                f"message {message_number}: adapter raised "
                f"{type(exc).__name__}: {exc}"
            )
            self._finish_at_limit(client)
            return

        if result.status == "NOT_READY":
            self.not_ready_messages += 1
            self._print_status(message_number, message, result.status)
        elif result.status == "INVALID_INPUT":
            self._record_invalid(
                f"message {message_number}: {result.reason or 'invalid adapter input'}"
            )
            if result.failure_kind:
                self.adapter_failure_kinds[result.failure_kind] += 1
            self.ordering_failures += result.failure_kind == "timestamp_order"
            self.cadence_failures += result.failure_kind == "cadence"
            self._print_status(message_number, message, result.status)
        elif result.status == "READY":
            self.ready_messages += 1
            self._process_ready(
                message_number, message, result.features, result.timestamp
            )
            self._print_status(message_number, message, result.status)
        else:
            self._record_invalid(
                f"message {message_number}: unrecognized adapter status "
                f"{result.status!r}"
            )
            self._print_status(message_number, message, "INVALID_INPUT")

        self._finish_at_limit(client)

    def _process_ready(
        self, message_number: int, message: object, features, timestamp
    ) -> None:
        if not isinstance(features, pd.DataFrame):
            self._record_feature_failure("adapter did not return a DataFrame")
            return

        self.generated_feature_rows += 1
        columns = list(features.columns)
        self.generated_feature_counts[len(columns)] += 1

        expected = self.expected_features
        missing = set(expected) - set(columns)
        unexpected = set(columns) - set(expected)
        self.missing_features += len(missing)
        self.unexpected_features += len(unexpected)

        order_valid = (
            columns == expected
            and expected == self.forecaster_features
            and len(columns) == EXPECTED_FEATURE_COUNT
        )
        self.manifest_order_checks += 1
        if not order_valid:
            self.manifest_order_failures += 1
            self._record_feature_failure(
                "generated columns do not exactly match the model manifest order"
            )

        if features.shape != (1, EXPECTED_FEATURE_COUNT):
            self.model_input_validation_failures += 1
            self._record_feature_failure(
                f"expected one row and {EXPECTED_FEATURE_COUNT} columns; "
                f"received shape {features.shape}"
            )
            return

        try:
            values = features.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError, OverflowError) as exc:
            self.model_input_validation_failures += 1
            self._record_feature_failure(f"generated input is not numeric: {exc}")
            return

        self.input_nan_count += int(np.isnan(values).sum())
        self.input_infinite_count += int(np.isinf(values).sum())
        if np.isnan(values).any() or np.isinf(values).any():
            self.model_input_validation_failures += 1
            self._record_feature_failure("generated input contains NaN or infinity")
            return

        if not order_valid:
            return

        self.model_input_feature_counts[len(columns)] += 1
        try:
            # Pass only the adapter's generated model-ready row. The forecasting
            # class performs its own manifest validation and model inference.
            prediction = self.forecaster.predict(features)
            self.model_accepted_inputs += 1
            try:
                forecasts = self._validate_forecast_output(prediction)
            except Exception:
                self.forecast_output_validation_failures += 1
                raise
        except Exception as exc:
            self.inference_failures += 1
            message_step = message.get("step") if isinstance(message, dict) else "?"
            self.inference_errors.append(
                f"message {message_number} (step {message_step}): "
                f"{type(exc).__name__}: {exc}"
            )
            return

        self.successful_inferences += 1
        if self.first_successful_forecast_message is None:
            self.first_successful_forecast_message = message_number
        self.forecast_outputs.append(
            {
                "message": message_number,
                "step": message.get("step") if isinstance(message, dict) else None,
                "timestamp": str(timestamp),
                **forecasts,
            }
        )

    @staticmethod
    def _validate_forecast_output(prediction) -> dict[str, float]:
        if not isinstance(prediction, pd.DataFrame) or prediction.shape != (1, 3):
            raise ValueError(
                "forecast engine must return one row and exactly three outputs"
            )
        if list(prediction.columns) != list(FORECAST_COLUMNS):
            raise ValueError(
                "forecast engine returned unexpected horizon columns: "
                f"{list(prediction.columns)!r}"
            )

        forecasts = {}
        for column in FORECAST_COLUMNS:
            value = prediction.iloc[0][column]
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numbers.Number
            ):
                raise TypeError(f"{column} output is not numeric: {value!r}")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"{column} output is not finite: {numeric_value!r}")
            forecasts[column] = numeric_value
        return forecasts

    @staticmethod
    def _print_status(message_number: int, message: object, status: str) -> None:
        step = message.get("step", "?") if isinstance(message, dict) else "?"
        print(f"#{message_number} step={step} status={status}", flush=True)

    def _finish_at_limit(self, client) -> None:
        if self.messages_received >= self.max_messages:
            self.finished.set()
            client.disconnect()

    def run(self) -> dict:
        if self.initialization_error:
            self.connection_error = f"Initialization failed: {self.initialization_error}"
            self.finished.set()
            return self.summary()

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
        expected_not_ready = min(10, self.messages_received)
        expected_ready = max(0, self.messages_received - 10)
        manifest_order_valid = (
            self.manifest_order_checks > 0
            and self.manifest_order_failures == 0
            and self.expected_features == self.forecaster_features
        )
        feature_layer_valid = (
            self.feature_validation_failures == 0
            and self.generated_feature_rows == self.ready_messages
            and all(
                count == EXPECTED_FEATURE_COUNT
                for count in self.generated_feature_counts
            )
            and manifest_order_valid
            and self.input_nan_count == 0
            and self.input_infinite_count == 0
        )
        model_input_valid = (
            self.model_input_validation_failures == 0
            and self.model_accepted_inputs
            == sum(self.model_input_feature_counts.values())
            and all(
                count == EXPECTED_FEATURE_COUNT
                for count in self.model_input_feature_counts
            )
        )
        behavior_valid = (
            self.messages_received == self.max_messages
            and self.not_ready_messages == expected_not_ready
            and self.ready_messages == expected_ready
            and self.invalid_messages == 0
            and self.ordering_failures == 0
            and self.cadence_failures == 0
        )
        forecast_layer_valid = (
            self.successful_inferences == self.ready_messages
            and self.inference_failures == 0
            and len(self.forecast_outputs) == self.successful_inferences
            and self.forecast_output_validation_failures == 0
        )
        forecast_outputs_numeric_finite = (
            self.successful_inferences > 0
            and self.forecast_output_validation_failures == 0
            and len(self.forecast_outputs) == self.successful_inferences
        )
        passed = (
            self.connection_error is None
            and not self.interrupted
            and behavior_valid
            and feature_layer_valid
            and model_input_valid
            and forecast_layer_valid
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "messages_received": self.messages_received,
            "not_ready_messages": self.not_ready_messages,
            "ready_messages": self.ready_messages,
            "invalid_messages": self.invalid_messages,
            "invalid_reasons": self.invalid_reasons,
            "adapter_failure_kinds": dict(self.adapter_failure_kinds),
            "ordering_failures": self.ordering_failures,
            "cadence_failures": self.cadence_failures,
            "generated_feature_rows": self.generated_feature_rows,
            "generated_feature_counts": dict(self.generated_feature_counts),
            "manifest_order_valid": manifest_order_valid,
            "manifest_order_failures": self.manifest_order_failures,
            "missing_features": self.missing_features,
            "unexpected_features": self.unexpected_features,
            "input_nan_count": self.input_nan_count,
            "input_infinite_count": self.input_infinite_count,
            "feature_validation_failures": self.feature_validation_failures,
            "model_input_feature_counts": dict(self.model_input_feature_counts),
            "model_input_validation_failures": self.model_input_validation_failures,
            "model_accepted_inputs": self.model_accepted_inputs,
            "model_input_valid": model_input_valid,
            "successful_inferences": self.successful_inferences,
            "inference_failures": self.inference_failures,
            "forecast_output_validation_failures": self.forecast_output_validation_failures,
            "forecast_outputs_numeric_finite": forecast_outputs_numeric_finite,
            "inference_errors": self.inference_errors,
            "forecast_outputs": self.forecast_outputs,
            "first_successful_forecast_message": self.first_successful_forecast_message,
            "feature_layer_valid": feature_layer_valid,
            "forecast_layer_valid": forecast_layer_valid,
            "connection_error": self.connection_error,
            "interrupted": self.interrupted,
        }

    def write_report(self, summary: dict) -> None:
        generated_counts = summary["generated_feature_counts"]
        generated_text = (
            ", ".join(
                f"{count} features: {rows} READY row(s)"
                for count, rows in sorted(generated_counts.items())
            )
            if generated_counts
            else "No READY feature rows generated"
        )
        model_counts = summary["model_input_feature_counts"]
        model_input_text = (
            ", ".join(
                f"{count} features: {rows} forecast input(s)"
                for count, rows in sorted(model_counts.items())
            )
            if model_counts
            else "No forecast input passed validation"
        )
        lines = [
            "600 MW MQTT RUNTIME FORECASTING VALIDATION",
            "=" * 47,
            "",
            "Data classification: HISTORICAL REPLAY; not live SCADA.",
            f"Broker: {MQTT_HOST}:{MQTT_PORT}",
            f"Topic: {MQTT_TOPIC}",
            f"Message limit: {self.max_messages}",
            "Forecast interface: Forecasting600MW.predict(generated_features)",
            "Input source: adapter-generated 119-feature row from MQTT raw values only.",
            "Replay pre-engineered feature fields are not used as model inputs.",
            "No PostgreSQL, HTTP/FastAPI, anomaly detection, or decision support calls.",
            f"FINAL RESULT: {summary['status']}",
            "",
            "MQTT layer",
            "-----------",
            f"Messages received: {summary['messages_received']}",
            f"NOT_READY: {summary['not_ready_messages']}",
            f"READY: {summary['ready_messages']}",
            f"Invalid: {summary['invalid_messages']}",
            f"Ordering failures: {summary['ordering_failures']}",
            f"Cadence failures: {summary['cadence_failures']}",
            f"Connection error: {summary['connection_error'] or 'None'}",
            "",
            "Feature layer",
            "-------------",
            f"READY feature rows generated: {summary['generated_feature_rows']}",
            f"Generated feature counts: {generated_text}",
            f"Exact model manifest order: {summary['manifest_order_valid']}",
            f"Manifest order failures: {summary['manifest_order_failures']}",
            f"Missing features: {summary['missing_features']}",
            f"Unexpected features: {summary['unexpected_features']}",
            f"Model input NaN count: {summary['input_nan_count']}",
            f"Model input infinity count: {summary['input_infinite_count']}",
            f"Feature validation failures: {summary['feature_validation_failures']}",
            "",
            "Forecast layer",
            "--------------",
            f"Model input feature counts: {model_input_text}",
            f"Model accepted exactly 119 generated features: {summary['model_input_valid']}",
            f"Accepted model input rows: {summary['model_accepted_inputs']}",
            f"Successful inferences: {summary['successful_inferences']}",
            f"Inference failures: {summary['inference_failures']}",
            f"Numeric finite forecast outputs: {summary['forecast_outputs_numeric_finite']}",
            f"Forecast output validation failures: {summary['forecast_output_validation_failures']}",
            "First successful forecast message: "
            f"{summary['first_successful_forecast_message']}",
            "",
            "Forecast outputs by successful MQTT message",
            "---------------------------------------------",
        ]
        if summary["forecast_outputs"]:
            for row in summary["forecast_outputs"]:
                lines.append(
                    f"message={row['message']} step={row['step']} "
                    f"timestamp={row['timestamp']} "
                    f"power_2min={row['power_2min']:.12g} "
                    f"power_10min={row['power_10min']:.12g} "
                    f"power_30min={row['power_30min']:.12g}"
                )
        else:
            lines.append("No successful forecast outputs.")

        lines.extend(
            [
                "",
                "Inference errors",
                "-----------------",
            ]
        )
        if summary["inference_errors"]:
            lines.extend(f"- {error}" for error in summary["inference_errors"])
        else:
            lines.append("None")

        if summary["invalid_reasons"]:
            lines.extend(["", "Validation details", "-------------------"])
            lines.extend(f"- {reason}" for reason in summary["invalid_reasons"])

        lines.extend(
            [
                "",
                "Execution commands from the repository root",
                "--------------------------------------------",
                "Terminal 1 (only if no broker is already listening on 127.0.0.1:1883):",
                "  mosquitto -v",
                "Terminal 2 (start this waiting subscriber first):",
                "  python scripts/mqtt/test_600mw_runtime_forecasting_mqtt.py --max-messages 20",
                "Terminal 3 (run separately after Terminal 2 reports subscribed):",
                "  python scripts/mqtt/scada_replay_publisher.py",
                "The subscriber does not start the publisher and sends no messages.",
                "",
            ]
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-messages",
        type=int,
        default=DEFAULT_MAX_MESSAGES,
        help=f"stop after this many MQTT messages (default: {DEFAULT_MAX_MESSAGES})",
    )
    args = parser.parse_args()

    check = MqttRuntimeForecastValidation(args.max_messages)
    print(
        f"Connecting to {MQTT_HOST}:{MQTT_PORT}; topic={MQTT_TOPIC}. "
        "The replay publisher is not started by this script.",
        flush=True,
    )
    summary = check.run()
    check.write_report(summary)

    print("\nFINAL RESULT:", flush=True)
    print(summary["status"], flush=True)
    for key in (
        "messages_received",
        "not_ready_messages",
        "ready_messages",
        "invalid_messages",
        "generated_feature_rows",
        "generated_feature_counts",
        "manifest_order_valid",
        "feature_validation_failures",
        "input_nan_count",
        "input_infinite_count",
        "model_input_valid",
        "successful_inferences",
        "inference_failures",
        "forecast_outputs_numeric_finite",
        "forecast_output_validation_failures",
        "first_successful_forecast_message",
        "ordering_failures",
        "cadence_failures",
    ):
        print(f"{key}: {summary[key]}", flush=True)

    print("forecast_outputs:", flush=True)
    for row in summary["forecast_outputs"]:
        print(
            f"  message={row['message']} step={row['step']} "
            f"power_2min={row['power_2min']:.12g} "
            f"power_10min={row['power_10min']:.12g} "
            f"power_30min={row['power_30min']:.12g}",
            flush=True,
        )
    if summary["inference_errors"]:
        print("inference_errors:", flush=True)
        for error in summary["inference_errors"]:
            print(f"  {error}", flush=True)
    print(f"report: {REPORT_PATH}", flush=True)

    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
