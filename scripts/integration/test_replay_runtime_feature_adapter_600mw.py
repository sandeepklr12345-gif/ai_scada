"""Validate 600 MW historical replay messages through the runtime adapter.

This script is an offline validation harness. It does not publish MQTT
messages and does not call a forecasting model.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np

from replay_runtime_feature_adapter_600mw import (
    ReplayRuntimeFeatureAdapter600MW,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPLAY_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_scada_replay.csv"
)
CLASSIFICATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_runtime_input_classification.csv"
)
VALIDATION_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_replay_runtime_feature_adapter_validation.txt"
)

ABS_TOL = 1e-9
REL_TOL = 1e-9


def _escaped(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


def _load_classification(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {row["feature"]: row["feature_class"] for row in rows}


def run_validation() -> dict:
    if not REPLAY_INPUT_PATH.exists():
        raise FileNotFoundError(f"Replay input not found: {REPLAY_INPUT_PATH}")
    if not CLASSIFICATION_PATH.exists():
        raise FileNotFoundError(f"Feature classification not found: {CLASSIFICATION_PATH}")

    adapter = ReplayRuntimeFeatureAdapter600MW()
    classification = _load_classification(CLASSIFICATION_PATH)
    raw_features = list(adapter.raw_features)
    expected_features = list(adapter.expected_features)

    with REPLAY_INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("Replay input has no CSV header")
        metadata_columns = {"timestamp", "replay_source", "replay_step"}
        missing_metadata = metadata_columns - set(reader.fieldnames)
        if missing_metadata:
            raise ValueError(f"Replay input is missing metadata: {sorted(missing_metadata)}")
        payload_columns = [name for name in reader.fieldnames if name not in metadata_columns]
        if set(payload_columns) != set(expected_features):
            raise ValueError("Replay feature columns do not exactly match the model manifest")

        counters = Counter()
        first_ten_statuses = []
        first_ready_step = None
        ready_rows = 0
        invalid_rows = 0
        raw_values_received = 0
        missing_raw_fields = 0
        non_numeric_values = 0
        non_finite_values = 0
        timestamp_order_failures = 0
        cadence_failures = 0
        output_nan_count = 0
        output_infinite_count = 0
        manifest_order_valid = True
        observed_output_feature_counts = set()
        total_rows = 0
        total_comparisons = 0
        total_mismatches = 0
        maximum_absolute_difference = 0.0
        maximum_relative_difference = 0.0
        per_feature = {
            feature: {
                "compared": 0,
                "max_abs": 0.0,
                "max_rel": 0.0,
                "mismatches": 0,
            }
            for feature in expected_features
        }
        first_timestamp = None
        first_ready_timestamp = None

        for row in reader:
            total_rows += 1
            if first_timestamp is None:
                first_timestamp = row["timestamp"]

            try:
                replay_step = int(row["replay_step"])
                source_features = {
                    name: float(row[name])
                    for name in payload_columns
                }
            except (TypeError, ValueError, KeyError, OverflowError) as exc:
                raise ValueError(
                    f"Could not create publisher-shaped message at CSV row {total_rows}: {exc}"
                ) from exc

            raw_values_received += sum(name in source_features for name in raw_features)

            # Keep source timestamps for this offline comparison so derived
            # time features are compared against their historical CSV values.
            # The production publisher rebases timestamps before publishing.
            message = {
                "timestamp": str(row["timestamp"]),
                "source": str(row["replay_source"]),
                "step": replay_step,
                "features": source_features,
            }
            result = adapter.process_message(message)
            counters[result.status] += 1

            if total_rows <= 10:
                first_ten_statuses.append((replay_step, result.status))

            if result.status == "INVALID_INPUT":
                invalid_rows += 1
                missing_raw_fields += len(result.missing_raw_fields)
                non_numeric_values += len(result.non_numeric_fields)
                non_finite_values += len(result.non_finite_fields)
                timestamp_order_failures += result.failure_kind == "timestamp_order"
                cadence_failures += result.failure_kind == "cadence"
                continue

            if result.status != "READY":
                continue

            ready_rows += 1
            if first_ready_step is None:
                first_ready_step = replay_step
                first_ready_timestamp = str(result.timestamp)

            output = result.features
            if output is None:
                raise AssertionError("READY result has no feature frame")
            output_columns = list(output.columns)
            observed_output_feature_counts.add(len(output_columns))
            manifest_order_valid = manifest_order_valid and output_columns == expected_features

            generated = output.iloc[0]
            values = output.to_numpy(dtype=float)
            output_nan_count += int(np.isnan(values).sum())
            output_infinite_count += int(np.isinf(values).sum())

            for feature in expected_features:
                generated_value = float(generated[feature])
                replay_value = source_features[feature]
                difference = abs(generated_value - replay_value)
                relative = difference / max(abs(replay_value), ABS_TOL)
                stats = per_feature[feature]
                stats["compared"] += 1
                stats["max_abs"] = max(stats["max_abs"], difference)
                stats["max_rel"] = max(stats["max_rel"], relative)
                maximum_absolute_difference = max(maximum_absolute_difference, difference)
                maximum_relative_difference = max(maximum_relative_difference, relative)
                total_comparisons += 1
                if not math.isclose(
                    generated_value,
                    replay_value,
                    rel_tol=REL_TOL,
                    abs_tol=ABS_TOL,
                ):
                    stats["mismatches"] += 1
                    total_mismatches += 1

    expected_raw_count = sum(value == "RAW_SOURCE" for value in classification.values())
    if expected_raw_count != len(raw_features):
        raise AssertionError("Adapter raw feature names disagree with classification metadata")

    expected_ready_rows = max(0, total_rows - adapter.history_size + 1)
    output_feature_count = (
        next(iter(observed_output_feature_counts))
        if len(observed_output_feature_counts) == 1
        else 0
    )
    reproduced = (
        total_mismatches == 0
        and total_comparisons == ready_rows * len(expected_features)
        and ready_rows == expected_ready_rows
        and output_feature_count == len(expected_features)
        and manifest_order_valid
        and output_nan_count == 0
        and output_infinite_count == 0
    )

    lines = [
        "600 MW REPLAY RUNTIME FEATURE ADAPTER VALIDATION",
        "=" * 52,
        "",
        "Status: " + ("PASS" if reproduced and invalid_rows == 0 else "CHECK REQUIRED"),
        "Replay source: HISTORICAL REPLAY (not live SCADA)",
        "Replay input: " + REPLAY_INPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "Feature classification: " + CLASSIFICATION_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "MQTT publish performed: no",
        "Forecasting model called: no",
        "",
        "Replay and adapter counts",
        "-------------------------",
        f"Total replay rows: {total_rows}",
        f"Raw feature names loaded from classification: {expected_raw_count}",
        f"Raw values received: {raw_values_received}",
        f"First 10 row statuses: {first_ten_statuses}",
        f"First model-ready step: {first_ready_step}",
        f"First model-ready timestamp: {first_ready_timestamp}",
        f"Model-ready row count: {ready_rows}",
        f"Expected model-ready row count after 10-row warm-up: {expected_ready_rows}",
        f"Invalid rows: {invalid_rows}",
        f"Timestamp-order failures: {timestamp_order_failures}",
        f"Cadence failures: {cadence_failures}",
        f"Missing raw fields: {missing_raw_fields}",
        f"Non-numeric raw values: {non_numeric_values}",
        f"Non-finite raw values: {non_finite_values}",
        f"Builder output feature count: {output_feature_count}",
        f"Manifest feature order validation: {manifest_order_valid}",
        f"NaN count across model-ready output: {output_nan_count}",
        f"Infinite count across model-ready output: {output_infinite_count}",
        "",
        "Replay feature comparison",
        "--------------------------",
        f"Existing replay model-feature columns compared: {len(expected_features)} per ready row",
        f"Total generated/reference value comparisons: {total_comparisons}",
        f"Absolute tolerance: {ABS_TOL:.1e}",
        f"Relative tolerance: {REL_TOL:.1e}",
        "A value mismatches when math.isclose(generated, replay, rel_tol=REL_TOL, abs_tol=ABS_TOL) is false.",
        "Relative difference is abs(generated - replay) / max(abs(replay), ABS_TOL).",
        f"Maximum absolute difference: {maximum_absolute_difference:.17g}",
        f"Maximum relative difference: {maximum_relative_difference:.17g}",
        f"Total mismatched values: {total_mismatches}",
        f"All 119 features reproduced within documented tolerance: {reproduced}",
        "",
        "Per-feature differences (one row per frozen manifest feature)",
        "feature\tcompared\tmax_abs_diff\tmax_rel_diff\tmismatches",
    ]
    for feature in expected_features:
        stats = per_feature[feature]
        lines.append(
            f"{_escaped(feature)}\t{stats['compared']}\t{stats['max_abs']:.17g}"
            f"\t{stats['max_rel']:.17g}\t{stats['mismatches']}"
        )

    lines.extend(
        [
            "",
            "Replay timestamp note",
            "----------------------",
            "The offline validator uses the original CSV timestamp in the same MQTT-shaped message fields so rebuilt TIME_DERIVED values can be compared to the historical replay columns. The existing publisher rebases timestamps to a current simulation timeline before actual publication; therefore its runtime time-derived values correspond to those rebased timestamps and need not equal the stored historical time-derived columns.",
            "",
            "Required statement",
            '"The replay\'s pre-engineered 119 features were not used as inputs to the runtime feature builder."',
            "",
        ]
    )
    VALIDATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    return {
        "status": "PASS" if reproduced and invalid_rows == 0 else "CHECK REQUIRED",
        "total_rows": total_rows,
        "ready_rows": ready_rows,
        "first_ready_step": first_ready_step,
        "invalid_rows": invalid_rows,
        "mismatches": total_mismatches,
        "maximum_absolute_difference": maximum_absolute_difference,
        "maximum_relative_difference": maximum_relative_difference,
        "output_feature_count": output_feature_count,
        "manifest_order_valid": manifest_order_valid,
        "reproduced": reproduced,
    }


def main() -> None:
    summary = run_validation()
    print(summary)
    print(f"Validation report: {VALIDATION_REPORT_PATH}")
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
