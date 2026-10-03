"""Validate real MQTT messages through the 600 MW runtime feature adapter.

This subscriber never launches a publisher, writes PostgreSQL, calls an API,
or invokes a model. Run the publisher manually in a separate terminal only
after this subscriber reports that it has subscribed.

From the repository root:
  Terminal 1 (only if 127.0.0.1:1883 is not already listening):
    mosquitto -v
  Terminal 2:
    python scripts/mqtt/test_600mw_runtime_feature_adapter_mqtt.py --max-messages 20
  Terminal 3:
    python scripts/mqtt/scada_replay_publisher.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import paho.mqtt.client as mqtt

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
    / "600mw_mqtt_runtime_feature_adapter_validation.txt"
)
MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "ai_scada/scada/600mw"
DEFAULT_MAX_MESSAGES = 20
EXPECTED_REPLAY_SOURCE = "600MW_HISTORICAL_REPLAY"


class MqttRuntimeFeatureAdapterCheck:
    def __init__(self, max_messages: int) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")

        self.max_messages = max_messages
        self.adapter = ReplayRuntimeFeatureAdapter600MW()
        self.expected_features = list(self.adapter.expected_features)
        self.messages_received = 0
        self.not_ready_messages = 0
        self.ready_messages = 0
        self.invalid_messages = 0
        self.ordering_failures = 0
        self.cadence_failures = 0
        self.schema_validation_failures = 0
        self.missing_features = 0
        self.unexpected_features = 0
        self.nan_count = 0
        self.infinite_count = 0
        self.output_feature_counts = Counter()
        self.status_counts = Counter()
        self.first_ready_message: Optional[int] = None
        self.first_ready_step: Optional[int] = None
        self.first_ready_timestamp: Optional[str] = None
        self.connection_error: Optional[str] = None
        self.interrupted = False
        self.finished = threading.Event()
        self.subscribed = threading.Event()
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="ai_scada_600mw_runtime_feature_adapter_test",
        )
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if getattr(reason_code, "is_failure", False):
            self.connection_error = f"MQTT connection rejected: {reason_code}"
            self.finished.set()
            return

        result, _ = client.subscribe(MQTT_TOPIC, qos=0)
        if result != mqtt.MQTT_ERR_SUCCESS:
            self.connection_error = f"MQTT subscribe failed with code {result}"
            self.finished.set()
            return

        self.subscribed.set()
        print(
            f"Subscribed to {MQTT_TOPIC} at {MQTT_HOST}:{MQTT_PORT}; "
            f"waiting for {self.max_messages} message(s).",
            flush=True,
        )

    def _record_schema_failure(self) -> None:
        self.schema_validation_failures += 1

    def on_message(self, client, userdata, mqtt_message) -> None:
        self.messages_received += 1
        message_number = self.messages_received

        try:
            message = json.loads(mqtt_message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.invalid_messages += 1
            self.status_counts["INVALID_JSON"] += 1
            print(f"#{message_number}: INVALID_JSON ({exc})", flush=True)
            self._finish_at_limit(client)
            return

        try:
            result = self.adapter.process_message(message)
        except Exception as exc:  # Keep a malformed MQTT message from killing the listener.
            self.invalid_messages += 1
            self.status_counts["INVALID_INPUT"] += 1
            print(f"#{message_number}: INVALID_INPUT ({exc})", flush=True)
            self._finish_at_limit(client)
            return

        self.status_counts[result.status] += 1
        if result.status == "NOT_READY":
            self.not_ready_messages += 1
        elif result.status == "INVALID_INPUT":
            self.invalid_messages += 1
            self.ordering_failures += result.failure_kind == "timestamp_order"
            self.cadence_failures += result.failure_kind == "cadence"
        elif result.status == "READY":
            self.ready_messages += 1
            if self.first_ready_message is None:
                self.first_ready_message = message_number
                step = message.get("step") if isinstance(message, dict) else None
                self.first_ready_step = step if isinstance(step, int) else None
                self.first_ready_timestamp = str(result.timestamp)

            output = result.features
            if output is None:
                self._record_schema_failure()
            else:
                columns = list(output.columns)
                self.output_feature_counts[len(columns)] += 1
                missing = set(self.expected_features) - set(columns)
                unexpected = set(columns) - set(self.expected_features)
                self.missing_features += len(missing)
                self.unexpected_features += len(unexpected)
                values = output.to_numpy(dtype=float, na_value=np.nan)
                self.nan_count += int(np.isnan(values).sum())
                self.infinite_count += int(np.isinf(values).sum())

                if (
                    len(columns) != 119
                    or columns != self.expected_features
                    or missing
                    or unexpected
                    or np.isnan(values).any()
                    or np.isinf(values).any()
                ):
                    self._record_schema_failure()

        print(
            f"#{message_number} step={message.get('step') if isinstance(message, dict) else '?'} "
            f"status={result.status}"
            + (
                f" features={len(result.features.columns)}"
                if result.features is not None
                else ""
            ),
            flush=True,
        )
        self._finish_at_limit(client)

    def _finish_at_limit(self, client) -> None:
        if self.messages_received >= self.max_messages:
            self.finished.set()
            client.disconnect()

    def run(self) -> dict:
        try:
            self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self.client.loop_start()
            self.finished.wait()
        except KeyboardInterrupt:
            self.interrupted = True
            self.finished.set()
        except Exception as exc:
            self.connection_error = f"{type(exc).__name__}: {exc}"
            self.finished.set()
        finally:
            try:
                self.client.loop_stop()
            finally:
                try:
                    self.client.disconnect()
                except Exception:
                    pass

        return self.summary()

    def summary(self) -> dict:
        expected_not_ready = min(10, self.messages_received)
        expected_ready = max(0, self.messages_received - 10)
        full_message_count = self.messages_received == self.max_messages
        schema_valid = (
            self.ready_messages == sum(self.output_feature_counts.values())
            and all(count == 119 for count in self.output_feature_counts)
            and self.schema_validation_failures == 0
            and self.missing_features == 0
            and self.unexpected_features == 0
            and self.nan_count == 0
            and self.infinite_count == 0
        )
        behavior_valid = (
            self.not_ready_messages == expected_not_ready
            and self.ready_messages == expected_ready
            and self.invalid_messages == 0
            and self.ordering_failures == 0
            and self.cadence_failures == 0
        )
        passed = (
            full_message_count
            and self.connection_error is None
            and not self.interrupted
            and behavior_valid
            and schema_valid
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "messages_received": self.messages_received,
            "not_ready_messages": self.not_ready_messages,
            "ready_messages": self.ready_messages,
            "invalid_messages": self.invalid_messages,
            "ordering_failures": self.ordering_failures,
            "cadence_failures": self.cadence_failures,
            "schema_validation_failures": self.schema_validation_failures,
            "missing_features": self.missing_features,
            "unexpected_features": self.unexpected_features,
            "output_feature_counts": dict(self.output_feature_counts),
            "nan_count": self.nan_count,
            "infinite_count": self.infinite_count,
            "first_ready_message": self.first_ready_message,
            "first_ready_step": self.first_ready_step,
            "first_ready_timestamp": self.first_ready_timestamp,
            "connection_error": self.connection_error,
            "interrupted": self.interrupted,
            "schema_valid": schema_valid,
            "behavior_valid": behavior_valid,
        }

    def write_report(self, summary: dict) -> None:
        output_counts = summary["output_feature_counts"]
        output_count_text = (
            ", ".join(f"{count} features: {rows} READY row(s)" for count, rows in sorted(output_counts.items()))
            if output_counts
            else "No READY rows received"
        )
        lines = [
            "600 MW MQTT RUNTIME FEATURE ADAPTER VALIDATION",
            "=" * 49,
            "",
            "Data classification: HISTORICAL REPLAY; not live SCADA.",
            f"Broker: {MQTT_HOST}:{MQTT_PORT}",
            f"Topic: {MQTT_TOPIC}",
            f"Message limit: {self.max_messages}",
            f"Final result: {summary['status']}",
            "",
            "Message counts",
            "---------------",
            f"Messages received: {summary['messages_received']}",
            f"NOT_READY messages: {summary['not_ready_messages']}",
            f"READY messages: {summary['ready_messages']}",
            f"Invalid messages: {summary['invalid_messages']}",
            f"First READY message number: {summary['first_ready_message']}",
            f"First READY publisher step: {summary['first_ready_step']}",
            f"First READY timestamp: {summary['first_ready_timestamp']}",
            "",
            "Runtime feature validation",
            "---------------------------",
            f"Observed output feature counts: {output_count_text}",
            f"Expected feature count: 119",
            f"Exact manifest order on all READY rows: {summary['schema_valid']}",
            f"Schema validation failures: {summary['schema_validation_failures']}",
            f"Missing output features: {summary['missing_features']}",
            f"Unexpected output features: {summary['unexpected_features']}",
            f"NaN count: {summary['nan_count']}",
            f"Infinite count: {summary['infinite_count']}",
            "",
            "Input validation",
            "-----------------",
            f"Ordering failures: {summary['ordering_failures']}",
            f"Cadence failures: {summary['cadence_failures']}",
            f"Connection error: {summary['connection_error'] or 'None'}",
            f"Subscriber interrupted before message limit: {summary['interrupted']}",
            "",
            "Execution commands from the repository root",
            "--------------------------------------------",
            "Terminal 1 (only if no broker is already listening on 127.0.0.1:1883):",
            "  mosquitto -v",
            "Terminal 2 (start this waiting subscriber first):",
            "  python scripts/mqtt/test_600mw_runtime_feature_adapter_mqtt.py --max-messages 20",
            "Terminal 3 (run separately after Terminal 2 reports subscribed):",
            "  python scripts/mqtt/scada_replay_publisher.py",
            "The test subscriber does not launch the publisher. It calls only the runtime adapter/feature builder; it does not publish MQTT, call FastAPI, write PostgreSQL, or invoke a model.",
            "",
        ]
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

    check = MqttRuntimeFeatureAdapterCheck(args.max_messages)
    print(
        f"Connecting to {MQTT_HOST}:{MQTT_PORT}; topic={MQTT_TOPIC}. "
        "The publisher is not started by this script.",
        flush=True,
    )
    summary = check.run()
    check.write_report(summary)
    print("\nMQTT RUNTIME FEATURE ADAPTER VALIDATION", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"report: {REPORT_PATH}", flush=True)

    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
