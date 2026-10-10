"""Strict live acquisition contract for the 600 MW forecasting pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT / "data" / "integration" / "600mw_runtime_source_contract.json"
)

EXPECTED_SOURCE = "600MW_LIVE"
EXPECTED_INTERVAL = pd.Timedelta(minutes=2)


def load_raw_features() -> tuple[str, ...]:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)

    features = tuple(
        variable["model_source_feature_name"]
        for variable in contract["variables"]
        if variable.get("source_provenance", {}).get("lineage_feature_class")
        == "RAW_SOURCE"
    )

    if len(features) != 71:
        raise ValueError(f"Expected 71 RAW_SOURCE features, found {len(features)}")

    return features


RAW_FEATURES = load_raw_features()


def validate_live_message(
    timestamp: object,
    features: Mapping[str, object],
    previous_timestamp: object | None = None,
) -> dict:
    """Validate one live 71-signal acquisition sample."""

    parsed_timestamp = pd.Timestamp(timestamp)

    if pd.isna(parsed_timestamp):
        raise ValueError("Invalid timestamp")

    expected = set(RAW_FEATURES)
    received = set(features)

    missing = sorted(expected - received)
    extra = sorted(received - expected)

    if missing:
        raise ValueError(
            f"Missing {len(missing)} required RAW_SOURCE signal(s): {missing}"
        )

    if extra:
        raise ValueError(
            f"Received {len(extra)} unexpected signal(s): {extra}"
        )

    validated = {}

    for name in RAW_FEATURES:
        value = features[name]

        if isinstance(value, bool):
            raise ValueError(f"{name}: boolean value is not allowed")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}: value is not numeric") from exc

        if not math.isfinite(numeric_value):
            raise ValueError(f"{name}: value is not finite")

        validated[name] = numeric_value

    if previous_timestamp is not None:
        previous = pd.Timestamp(previous_timestamp)
        interval = parsed_timestamp - previous

        if interval != EXPECTED_INTERVAL:
            raise ValueError(
                f"Expected 2-minute cadence; received {interval}"
            )

    return {
        "timestamp": parsed_timestamp.isoformat(),
        "source": EXPECTED_SOURCE,
        "features": validated,
    }


if __name__ == "__main__":
    print(f"RAW_SOURCE features: {len(RAW_FEATURES)}")
    print(f"Expected cadence: {EXPECTED_INTERVAL}")
    print("Live acquisition contract: READY")