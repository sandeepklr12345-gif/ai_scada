"""Adapt historical 600 MW MQTT-shaped messages to the frozen feature builder.

The historical replay contains pre-engineered columns as well as raw values.
Only RAW_SOURCE values are retained here; RuntimeFeatureBuilder600MW remains
the single implementation that creates time/lag features and the model schema.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from runtime_feature_builder_600mw import RuntimeFeatureBuilder600MW


EXPECTED_REPLAY_SOURCE = "600MW_HISTORICAL_REPLAY"


@dataclass(frozen=True)
class ReplayFeatureAdapterResult:
    """One adapter outcome for one replay message."""

    status: str
    timestamp: Optional[pd.Timestamp] = None
    features: Optional[pd.DataFrame] = None
    reason: Optional[str] = None
    failure_kind: Optional[str] = None
    missing_raw_fields: Tuple[str, ...] = ()
    non_numeric_fields: Tuple[str, ...] = ()
    non_finite_fields: Tuple[str, ...] = ()


class ReplayRuntimeFeatureAdapter600MW:
    """Extract raw replay values and delegate feature construction.

    Pass one decoded MQTT-shaped message to :meth:`process_message` at a time.
    The adapter loads its authoritative raw feature names, expected feature
    order, lag warm-up length, and sampling interval from the existing runtime
    feature builder, which in turn loads the frozen manifest and lineage CSV.
    """

    def __init__(
        self,
        feature_builder: Optional[RuntimeFeatureBuilder600MW] = None,
        expected_source: str = EXPECTED_REPLAY_SOURCE,
    ) -> None:
        self.feature_builder = feature_builder or RuntimeFeatureBuilder600MW()
        self.raw_features = tuple(self.feature_builder.raw_features)
        self.expected_features = tuple(self.feature_builder.expected_features)
        self.timestamp_column = "timestamp"
        self.expected_source = expected_source
        self.expected_interval = self.feature_builder.expected_sampling_interval
        self.history_size = max(11, self.feature_builder.warmup_rows_required + 1)
        self._history = deque(maxlen=self.history_size)
        self._last_timestamp: Optional[pd.Timestamp] = None

        if len(self.raw_features) != 71:
            raise ValueError(
                f"Expected 71 RAW_SOURCE names from the feature builder; "
                f"found {len(self.raw_features)}"
            )

    @property
    def history_rows(self) -> int:
        """Number of accepted source rows currently retained."""

        return len(self._history)

    def reset(self) -> None:
        """Discard history after a replay sequence is stopped or replaced."""

        self._history.clear()
        self._last_timestamp = None

    def _invalid(
        self,
        reason: str,
        failure_kind: str,
        timestamp: Optional[pd.Timestamp] = None,
        *,
        missing_raw_fields: Tuple[str, ...] = (),
        non_numeric_fields: Tuple[str, ...] = (),
        non_finite_fields: Tuple[str, ...] = (),
    ) -> ReplayFeatureAdapterResult:
        return ReplayFeatureAdapterResult(
            status="INVALID_INPUT",
            timestamp=timestamp,
            reason=reason,
            failure_kind=failure_kind,
            missing_raw_fields=missing_raw_fields,
            non_numeric_fields=non_numeric_fields,
            non_finite_fields=non_finite_fields,
        )

    def process_message(self, message: object) -> ReplayFeatureAdapterResult:
        """Validate one replay message and emit a ready row when possible."""

        if not isinstance(message, Mapping):
            self.reset()
            return self._invalid(
                "Message must be a mapping/object.", "malformed_message"
            )

        try:
            timestamp = pd.Timestamp(message.get(self.timestamp_column))
            if pd.isna(timestamp):
                raise ValueError("timestamp parsed as NaT")
        except (TypeError, ValueError, OverflowError) as exc:
            self.reset()
            return self._invalid(
                f"Invalid timestamp: {exc}", "invalid_timestamp"
            )

        if self._last_timestamp is not None:
            try:
                interval = timestamp - self._last_timestamp
            except (TypeError, ValueError) as exc:
                self._history.clear()
                self._last_timestamp = None
                return self._invalid(
                    f"Timestamp timezone/type is inconsistent: {exc}",
                    "invalid_timestamp",
                    timestamp,
                )

            if interval <= pd.Timedelta(0):
                self._history.clear()
                return self._invalid(
                    "Timestamp is not strictly later than the previous message.",
                    "timestamp_order",
                    timestamp,
                )

            if interval != self.expected_interval:
                self._history.clear()
                self._last_timestamp = timestamp
                return self._invalid(
                    f"Expected cadence {self.expected_interval}; received {interval}.",
                    "cadence",
                    timestamp,
                )

        self._last_timestamp = timestamp

        if message.get("source") != self.expected_source:
            self._history.clear()
            return self._invalid(
                f"Expected historical replay source {self.expected_source!r}; "
                f"received {message.get('source')!r}.",
                "source",
                timestamp,
            )

        feature_values = message.get("features")
        if not isinstance(feature_values, Mapping):
            self._history.clear()
            return self._invalid(
                "Message field 'features' must be an object/mapping.",
                "malformed_features",
                timestamp,
            )

        missing = tuple(
            feature for feature in self.raw_features if feature not in feature_values
        )
        if missing:
            self._history.clear()
            return self._invalid(
                f"Message is missing {len(missing)} RAW_SOURCE field(s).",
                "missing_raw",
                timestamp,
                missing_raw_fields=missing,
            )

        raw_values = {}
        non_numeric = []
        non_finite = []
        for feature in self.raw_features:
            value = feature_values[feature]
            if isinstance(value, bool):
                non_numeric.append(feature)
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError):
                non_numeric.append(feature)
                continue
            if not math.isfinite(numeric_value):
                non_finite.append(feature)
                continue
            raw_values[feature] = numeric_value

        if non_numeric or non_finite:
            self._history.clear()
            return self._invalid(
                "One or more RAW_SOURCE values are non-numeric or non-finite.",
                "non_finite" if non_finite and not non_numeric else "invalid_values",
                timestamp,
                non_numeric_fields=tuple(non_numeric),
                non_finite_fields=tuple(non_finite),
            )

        history_row = {self.timestamp_column: timestamp}
        history_row.update(raw_values)
        self._history.append(history_row)

        if len(self._history) < self.history_size:
            return ReplayFeatureAdapterResult(
                status="NOT_READY",
                timestamp=timestamp,
                reason=(
                    f"Warm-up requires {self.history_size - 1} prior rows; "
                    f"history currently has {len(self._history)} row(s)."
                ),
            )

        history_frame = pd.DataFrame(
            list(self._history),
            columns=[self.timestamp_column, *self.raw_features],
        )
        try:
            build_result = self.feature_builder.build(
                history_frame,
                timestamp_column=self.timestamp_column,
            )
        except Exception as exc:
            self._history.clear()
            return self._invalid(
                f"Runtime feature builder rejected the source history: {exc}",
                "builder_validation",
                timestamp,
            )

        output = build_result.features
        if output.shape != (1, len(self.expected_features)):
            self._history.clear()
            return self._invalid(
                "Builder did not return exactly one model-ready feature row.",
                "builder_output",
                timestamp,
            )
        if tuple(output.columns) != self.expected_features:
            self._history.clear()
            return self._invalid(
                "Builder output does not match the frozen manifest feature order.",
                "builder_output",
                timestamp,
            )

        values = output.to_numpy(dtype=float, na_value=np.nan)
        if np.isnan(values).any() or np.isinf(values).any():
            self._history.clear()
            return self._invalid(
                "Builder output contains NaN or infinite values.",
                "builder_output",
                timestamp,
            )

        return ReplayFeatureAdapterResult(
            status="READY",
            timestamp=timestamp,
            features=output.copy(),
        )
