"""Build the frozen 600 MW model's runtime feature frame.

Lag features are row-based, matching the convention used to train the model.
For a two-minute input stream, row lag 10 is nominally twenty minutes back.
Rows are withheld from ``features`` until they have ten prior observations,
and any lag window with an irregular sampling interval is withheld as well.

Running this module validates the builder against the original workbook and
writes ``data/integration/600mw_runtime_feature_builder_validation.txt``.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.forecasting_config import SAMPLING_MINUTES  # noqa: E402


MODEL_DIR = PROJECT_ROOT / "models" / "forecasting" / "600mw"
MANIFEST_PATH = MODEL_DIR / "600mw_forecasting_feature_manifest.json"
LINEAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_runtime_input_classification.csv"
)
RAW_WORKBOOK_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "600mw"
    / "600 MW unit one-week operating data.xlsx"
)
VALIDATION_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_runtime_feature_builder_validation.txt"
)

EXPECTED_FEATURE_COUNT = 119
EXPECTED_RAW_FEATURE_COUNT = 71
EXPECTED_TIME_FEATURE_COUNT = 4
EXPECTED_LAG_FEATURE_COUNT = 44
EXPECTED_LAG_STEPS = (1, 2, 5, 10)
TIME_FEATURE_BUILDERS = {
    "hour": lambda timestamps: timestamps.dt.hour,
    "minute": lambda timestamps: timestamps.dt.minute,
    "day_of_week": lambda timestamps: timestamps.dt.dayofweek,
    "day_of_month": lambda timestamps: timestamps.dt.day,
}


@dataclass(frozen=True)
class FeatureBuildResult:
    """Model-ready rows plus validation status for every input row."""

    features: pd.DataFrame
    row_status: pd.DataFrame
    sampling_interval_distribution: Dict[str, int]
    expected_sampling_interval: pd.Timedelta
    timestamp_ordering_valid: bool
    sampling_is_regular: bool
    warmup_rows_required: int
    warmup_rows: int
    input_rows: int
    generated_nan_count: int
    generated_infinite_count: int
    missing_features: Tuple[str, ...]
    unexpected_features: Tuple[str, ...]
    order_matches_manifest: bool

    @property
    def model_ready(self) -> bool:
        """Whether at least one validated row can be passed to a model."""

        return not self.features.empty


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required model manifest not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _format_interval(interval: pd.Timedelta) -> str:
    seconds = interval.total_seconds()
    if seconds % 3600 == 0:
        hours = int(seconds // 3600)
        return f"{hours} hour(s)"
    if seconds % 60 == 0:
        minutes = int(seconds // 60)
        return f"{minutes} minute(s)"
    if seconds.is_integer():
        return f"{int(seconds)} second(s)"
    return f"{seconds:g} second(s)"


class RuntimeFeatureBuilder600MW:
    """Construct and validate the exact frozen 600 MW feature schema.

    ``source`` must contain the 71 exact RAW_SOURCE column names and a
    timestamp column. Additional source columns are ignored. The returned
    ``features`` frame contains only rows safe for inference and exactly the
    manifest's 119 columns in manifest order; timestamps and row statuses are
    returned separately.

    Lags use ``Series.shift(step)`` over the supplied chronological rows.
    They are not time-based resampling or duration lookups. Rows whose most
    recent ten intervals are not all the configured sampling interval are
    marked ``IRREGULAR_LAG_WINDOW`` and omitted from ``features``.
    """

    def __init__(
        self,
        manifest_path: Path = MANIFEST_PATH,
        lineage_path: Path = LINEAGE_PATH,
        sampling_minutes: int = SAMPLING_MINUTES,
    ) -> None:
        if sampling_minutes <= 0:
            raise ValueError("sampling_minutes must be a positive integer")

        self.manifest_path = Path(manifest_path)
        self.lineage_path = Path(lineage_path)
        self.sampling_minutes = int(sampling_minutes)
        self.expected_sampling_interval = pd.Timedelta(
            minutes=self.sampling_minutes
        )

        self.manifest = _load_json(self.manifest_path)
        features = self.manifest.get("features", self.manifest.get("feature_names"))
        if not isinstance(features, list):
            raise ValueError("Model manifest does not contain a feature list")
        self.expected_features = [str(feature) for feature in features]
        self.timestamp_manifest_column = str(
            self.manifest.get("timestamp_column", "Time")
        )

        if len(self.expected_features) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_COUNT} manifest features; "
                f"found {len(self.expected_features)}"
            )
        if len(set(self.expected_features)) != len(self.expected_features):
            raise ValueError("Model manifest contains duplicate feature names")
        declared_count = self.manifest.get("feature_count")
        if declared_count is not None and declared_count != len(self.expected_features):
            raise ValueError(
                "Manifest feature_count does not match its feature list: "
                f"{declared_count} != {len(self.expected_features)}"
            )

        self.lineage = self._load_lineage()
        lineage_features = [row["feature"] for row in self.lineage]
        if lineage_features != self.expected_features:
            raise ValueError(
                "Lineage feature names/order do not exactly match the model manifest"
            )

        class_rows: Dict[str, List[dict]] = {}
        for row in self.lineage:
            class_rows.setdefault(row["feature_class"], []).append(row)

        if set(class_rows) != {"RAW_SOURCE", "TIME_DERIVED", "LAG_DERIVED"}:
            raise ValueError(
                "Unexpected feature classes in lineage: "
                f"{sorted(class_rows)}"
            )

        self.raw_rows = class_rows["RAW_SOURCE"]
        self.time_rows = class_rows["TIME_DERIVED"]
        self.lag_rows = class_rows["LAG_DERIVED"]
        self.raw_features = [row["feature"] for row in self.raw_rows]
        self.time_features = [row["feature"] for row in self.time_rows]
        self.lag_features = [row["feature"] for row in self.lag_rows]

        if len(self.raw_features) != EXPECTED_RAW_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_RAW_FEATURE_COUNT} RAW_SOURCE features; "
                f"found {len(self.raw_features)}"
            )
        if len(self.time_features) != EXPECTED_TIME_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_TIME_FEATURE_COUNT} TIME_DERIVED features; "
                f"found {len(self.time_features)}"
            )
        if set(self.time_features) != set(TIME_FEATURE_BUILDERS):
            raise ValueError(
                "TIME_DERIVED names do not match the supported time features: "
                f"{self.time_features}"
            )
        if len(self.lag_features) != EXPECTED_LAG_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_LAG_FEATURE_COUNT} LAG_DERIVED features; "
                f"found {len(self.lag_features)}"
            )

        lag_sources = {
            row["source_feature"]
            for row in self.lag_rows
            if row.get("source_feature")
        }
        if len(lag_sources) != 11:
            raise ValueError(
                f"Expected 11 lag-source variables; found {len(lag_sources)}"
            )
        if not lag_sources.issubset(set(self.raw_features)):
            raise ValueError("A lag source is not an authoritative RAW_SOURCE")

        lag_steps = set()
        for row in self.lag_rows:
            source_feature = row.get("source_feature", "")
            try:
                lag_step = int(row["lag_step"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid lag_step for feature {row['feature']!r}"
                ) from exc
            if row["feature"] != f"{source_feature}_lag_{lag_step}":
                raise ValueError(
                    "Lag feature name does not match its lineage: "
                    f"{row['feature']!r}"
                )
            lag_steps.add(lag_step)

        if tuple(sorted(lag_steps)) != EXPECTED_LAG_STEPS:
            raise ValueError(
                f"Expected lag steps {EXPECTED_LAG_STEPS}; found "
                f"{tuple(sorted(lag_steps))}"
            )
        self.warmup_rows_required = max(lag_steps)

    def _load_lineage(self) -> List[dict]:
        if not self.lineage_path.exists():
            raise FileNotFoundError(f"Feature lineage not found: {self.lineage_path}")
        with self.lineage_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)
            required_columns = {"feature", "feature_class", "source_feature", "lag_step"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    "Feature lineage is missing required columns: "
                    f"{sorted(required_columns)}"
                )
            rows = list(reader)
        if any(not row.get("feature") or not row.get("feature_class") for row in rows):
            raise ValueError("Feature lineage contains a blank feature or class")
        return rows

    def build(
        self,
        source: pd.DataFrame,
        timestamp_column: str = "timestamp",
    ) -> FeatureBuildResult:
        """Build 119 features and expose row-level readiness status.

        Timestamps must parse and be strictly increasing in the supplied row
        order. The builder does not sort input. Initial rows without ten prior
        observations are marked ``WARMUP_INCOMPLETE``. Non-finite generated
        values and irregular lag windows are also withheld from model input.
        """

        if not isinstance(source, pd.DataFrame):
            raise TypeError("Runtime source must be a pandas DataFrame")
        if source.empty:
            raise ValueError("Runtime source contains no rows")
        if timestamp_column not in source.columns:
            raise ValueError(
                f"Runtime source must contain timestamp column {timestamp_column!r}"
            )

        missing_source_features = [
            feature for feature in self.raw_features if feature not in source.columns
        ]
        if missing_source_features:
            raise ValueError(
                "Runtime source is missing required RAW_SOURCE features: "
                + ", ".join(repr(feature) for feature in missing_source_features)
            )

        timestamps = pd.Series(
            pd.to_datetime(source[timestamp_column], errors="coerce")
        ).reset_index(drop=True)
        invalid_timestamp_positions = np.flatnonzero(timestamps.isna().to_numpy())
        if len(invalid_timestamp_positions):
            positions = invalid_timestamp_positions[:10].tolist()
            raise ValueError(
                "Runtime source contains invalid timestamps at row positions "
                f"{positions}"
            )

        intervals = timestamps.diff()
        non_increasing = intervals.iloc[1:].le(pd.Timedelta(0)).to_numpy()
        if non_increasing.any():
            positions = (np.flatnonzero(non_increasing) + 1).tolist()
            raise ValueError(
                "Timestamps must be strictly increasing in input row order; "
                f"non-increasing intervals end at row positions {positions[:10]}. "
                "Input was not reordered."
            )

        try:
            time_values = {
                "hour": timestamps.dt.hour,
                "minute": timestamps.dt.minute,
                "day_of_week": timestamps.dt.dayofweek,
                "day_of_month": timestamps.dt.day,
            }
        except AttributeError as exc:
            raise ValueError(
                "Timestamps must use one consistent datetime/timezone type"
            ) from exc

        raw = source.loc[:, self.raw_features].reset_index(drop=True)
        numeric_raw = pd.DataFrame(index=pd.RangeIndex(len(source)))
        for feature in self.raw_features:
            numeric_raw[feature] = pd.to_numeric(
                raw[feature], errors="coerce"
            ).to_numpy()

        generated = numeric_raw.copy()
        for feature in self.time_features:
            generated[feature] = time_values[feature].to_numpy()

        # These are row-based lags, matching the training pipeline's shift()
        # convention. They are not computed by looking up elapsed durations.
        for row in self.lag_rows:
            source_feature = row["source_feature"]
            lag_step = int(row["lag_step"])
            generated[row["feature"]] = numeric_raw[source_feature].shift(
                lag_step
            ).to_numpy()

        actual_generated_features = list(generated.columns)
        missing_features = tuple(
            feature
            for feature in self.expected_features
            if feature not in actual_generated_features
        )
        unexpected_features = tuple(
            feature
            for feature in actual_generated_features
            if feature not in self.expected_features
        )
        if missing_features or unexpected_features:
            raise RuntimeError(
                "Generated feature schema differs from the manifest; "
                f"missing={missing_features}, unexpected={unexpected_features}"
            )

        ordered = generated.loc[:, self.expected_features]
        order_matches_manifest = list(ordered.columns) == self.expected_features
        if not order_matches_manifest:
            raise RuntimeError("Could not arrange features in manifest order")

        matrix = ordered.to_numpy(dtype=float, na_value=np.nan)
        nan_count = int(np.isnan(matrix).sum())
        infinite_count = int(np.isinf(matrix).sum())
        finite_rows = np.isfinite(matrix).all(axis=1)

        expected = self.expected_sampling_interval
        interval_matches_expected = intervals.eq(expected).fillna(False).to_numpy()
        sampling_interval_distribution: Dict[str, int] = {}
        for interval, count in intervals.iloc[1:].value_counts().sort_index().items():
            sampling_interval_distribution[_format_interval(interval)] = int(count)
        sampling_is_regular = bool(
            len(intervals) <= 1
            or interval_matches_expected[1:].all()
        )

        statuses: List[str] = []
        reasons: List[str] = []
        ready_positions: List[int] = []
        for position in range(len(source)):
            if position < self.warmup_rows_required:
                statuses.append("WARMUP_INCOMPLETE")
                reasons.append(
                    f"requires {self.warmup_rows_required} prior rows; "
                    f"has {position}"
                )
                continue

            window_start = position - self.warmup_rows_required + 1
            regular_lag_window = bool(
                interval_matches_expected[window_start : position + 1].all()
            )
            row_reasons = []
            if not regular_lag_window:
                row_reasons.append("most recent lag window is not regularly sampled")
            if not finite_rows[position]:
                row_reasons.append("generated features contain NaN or infinity")

            if row_reasons:
                statuses.append("BLOCKED")
                reasons.append("; ".join(row_reasons))
            else:
                statuses.append("READY")
                reasons.append("")
                ready_positions.append(position)

        ready_features = ordered.iloc[ready_positions].copy()
        ready_features.index = source.index.take(ready_positions)
        if not np.isfinite(
            ready_features.to_numpy(dtype=float, na_value=np.nan)
        ).all():
            raise RuntimeError("Non-finite values remain in model-ready features")

        row_status = pd.DataFrame(
            {
                "row_position": range(len(source)),
                "source_index": list(source.index),
                "timestamp": timestamps.to_numpy(),
                "status": statuses,
                "reason": reasons,
            }
        )

        return FeatureBuildResult(
            features=ready_features,
            row_status=row_status,
            sampling_interval_distribution=sampling_interval_distribution,
            expected_sampling_interval=expected,
            timestamp_ordering_valid=True,
            sampling_is_regular=sampling_is_regular,
            warmup_rows_required=self.warmup_rows_required,
            warmup_rows=min(len(source), self.warmup_rows_required),
            input_rows=len(source),
            generated_nan_count=nan_count,
            generated_infinite_count=infinite_count,
            missing_features=missing_features,
            unexpected_features=unexpected_features,
            order_matches_manifest=order_matches_manifest,
        )


def load_original_workbook(builder: RuntimeFeatureBuilder600MW) -> pd.DataFrame:
    """Load the source workbook using its established training timestamp parse."""

    if not RAW_WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Original 600 MW workbook not found: {RAW_WORKBOOK_PATH}")
    source = pd.read_excel(RAW_WORKBOOK_PATH)
    timestamp_column = builder.timestamp_manifest_column
    if timestamp_column not in source.columns:
        raise ValueError(
            f"Workbook is missing manifest timestamp column {timestamp_column!r}"
        )

    # Match scripts/cleaning/clean_600mw.py: the original workbook stores
    # yearless month/day timestamps and training parses them with this format.
    # pandas consequently uses year 1900, as in the existing cleaned data.
    timestamp_text = source[timestamp_column].astype(str).str.strip()
    source[timestamp_column] = pd.to_datetime(
        timestamp_text,
        format="%m/%d %H:%M:%S",
        errors="coerce",
    )
    return source


def _report_lines(
    builder: RuntimeFeatureBuilder600MW,
    source: pd.DataFrame,
    result: FeatureBuildResult,
) -> List[str]:
    expected = builder.expected_features
    actual = list(result.features.columns)
    missing = [feature for feature in expected if feature not in actual]
    unexpected = [feature for feature in actual if feature not in expected]
    order_matches = actual == expected

    ready_nan_count = int(result.features.isna().to_numpy().sum())
    ready_values = result.features.to_numpy(dtype=float, na_value=np.nan)
    ready_infinite_count = int(np.isinf(ready_values).sum())
    status_counts = result.row_status["status"].value_counts().to_dict()
    timestamp_column = builder.timestamp_manifest_column
    timestamp_values = pd.to_datetime(source[timestamp_column], errors="coerce")
    non_positive_count = int(timestamp_values.diff().iloc[1:].le(pd.Timedelta(0)).sum())
    irregular_count = sum(
        count
        for interval, count in result.sampling_interval_distribution.items()
        if interval != _format_interval(result.expected_sampling_interval)
    )
    source_extra_columns = [
        column
        for column in source.columns
        if column not in builder.raw_features and column != timestamp_column
    ]
    source_feature_presence = [
        feature for feature in builder.raw_features if feature not in source.columns
    ]
    lag_source_features = sorted(
        {row["source_feature"] for row in builder.lag_rows}
    )
    missing_lag_sources = [
        feature for feature in lag_source_features if feature not in source.columns
    ]

    passed = (
        len(actual) == EXPECTED_FEATURE_COUNT
        and not missing
        and not unexpected
        and order_matches
        and ready_nan_count == 0
        and ready_infinite_count == 0
        and result.timestamp_ordering_valid
        and result.sampling_is_regular
        and result.warmup_rows == result.warmup_rows_required
        and len(result.features) == len(source) - result.warmup_rows
        and not source_feature_presence
        and not missing_lag_sources
    )

    lines = [
        "600 MW RUNTIME FEATURE BUILDER VALIDATION",
        "=" * 45,
        "",
        f"Status: {'PASS' if passed else 'CHECK REQUIRED'}",
        f"Source workbook: {RAW_WORKBOOK_PATH.relative_to(PROJECT_ROOT).as_posix()}",
        f"Feature manifest: {builder.manifest_path.relative_to(PROJECT_ROOT).as_posix()}",
        f"Feature lineage: {builder.lineage_path.relative_to(PROJECT_ROOT).as_posix()}",
        f"Original workbook columns: {len(source.columns)}",
        f"Required raw source columns present: {EXPECTED_RAW_FEATURE_COUNT - len(source_feature_presence)} / {EXPECTED_RAW_FEATURE_COUNT}",
        f"Lag-source variables present: {len(lag_source_features) - len(missing_lag_sources)} / {len(lag_source_features)}",
        f"Other source columns ignored: {len(source_extra_columns)}",
        "",
        "Feature construction",
        "---------------------",
        f"Source rows: {len(source)}",
        f"Output rows passed as model-ready: {len(result.features)}",
        f"Warm-up rows: {result.warmup_rows} (requires {result.warmup_rows_required} prior rows for lag 10)",
        f"Row status counts: {status_counts}",
        f"Feature count: {len(actual)}",
        f"Missing features: {len(missing)}",
        f"Unexpected features: {len(unexpected)}",
        f"Generated feature order matches manifest: {order_matches}",
        f"NaN count in model-ready output: {ready_nan_count}",
        f"Infinite count in model-ready output: {ready_infinite_count}",
        f"NaN count in generated matrix including warm-up: {result.generated_nan_count}",
        f"Infinite count in generated matrix including warm-up: {result.generated_infinite_count}",
        "",
        "Timestamp and sampling validation",
        "----------------------------------",
        f"Timestamp ordering: {'PASS (strictly increasing)' if result.timestamp_ordering_valid else 'FAIL'}",
        f"Duplicate or non-increasing intervals: {non_positive_count}",
        f"Expected sampling interval: {_format_interval(result.expected_sampling_interval)}",
        f"All observed intervals regular: {result.sampling_is_regular}",
        f"Irregular interval count: {irregular_count}",
        "Sampling interval distribution:",
    ]
    if result.sampling_interval_distribution:
        lines.extend(
            f"  {interval}: {count} interval(s)"
            for interval, count in result.sampling_interval_distribution.items()
        )
    else:
        lines.append("  No intervals (fewer than two source rows).")

    lines.extend(
        [
            "",
            "Lag and timestamp conventions",
            "------------------------------",
            f"Lag source variables: {len({row['source_feature'] for row in builder.lag_rows})}",
            f"Lag steps: {', '.join(str(step) for step in EXPECTED_LAG_STEPS)} rows",
            "Lags are row-based shifts, not time-based lookups or resampling.",
            "Rows whose ten-interval lag window is not entirely at the configured cadence are marked BLOCKED and withheld.",
            "The workbook's Time values omit a year. Validation parses them with the same %m/%d %H:%M:%S convention as scripts/cleaning/clean_600mw.py, yielding the existing 1900-based dates. Runtime records should provide their actual observation timestamps.",
            "",
            "Schema comparison",
            "------------------",
            f"Manifest feature count: {len(expected)}",
            f"Generated feature count: {len(actual)}",
            f"Feature names and order identical to frozen model manifest: {order_matches and not missing and not unexpected}",
            f"Missing feature names: {missing if missing else 'None'}",
            f"Unexpected feature names: {unexpected if unexpected else 'None'}",
            "",
            "Inference was not run.",
        ]
    )
    return lines


def validate_original_workbook(
    report_path: Path = VALIDATION_REPORT_PATH,
) -> FeatureBuildResult:
    """Run the requested read-only workbook validation and save its report."""

    builder = RuntimeFeatureBuilder600MW()
    source = load_original_workbook(builder)
    result = builder.build(
        source,
        timestamp_column=builder.timestamp_manifest_column,
    )
    lines = _report_lines(builder, source, result)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = validate_original_workbook()
    print(f"Validation report: {VALIDATION_REPORT_PATH}")
    print(f"Source rows: {result.input_rows}")
    print(f"Model-ready output rows: {len(result.features)}")
    print(f"Warm-up rows: {result.warmup_rows}")
    print(f"Feature count: {len(result.features.columns)}")
    print(f"Sampling interval distribution: {result.sampling_interval_distribution}")
    print(f"Model-ready: {result.model_ready}")


if __name__ == "__main__":
    main()
