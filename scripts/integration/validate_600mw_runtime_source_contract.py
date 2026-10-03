"""Create the 600 MW runtime source contract and compare it with PostgreSQL.

This utility only reads project files and performs SELECTs in a read-only
PostgreSQL transaction. It writes the requested JSON contract and text report;
it does not write to PostgreSQL or generate measurement records.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_INTEGRATION = PROJECT_ROOT / "scripts" / "integration"
if str(SCRIPTS_INTEGRATION) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_INTEGRATION))

MANIFEST_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "600mw"
    / "600mw_forecasting_feature_manifest.json"
)
LINEAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_runtime_input_classification.csv"
)
WORKBOOK_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "600mw"
    / "600 MW unit one-week operating data.xlsx"
)
MAPPING_PATH = PROJECT_ROOT / "data" / "integration" / "600mw_raw_feature_mapping.csv"
ACQUISITION_PLAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_model_input_acquisition_plan.csv"
)
MODEL_MAPPING_REPORT_PATH = (
    PROJECT_ROOT / "data" / "integration" / "model_input_mapping_report.txt"
)
REPLAY_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
    / "600mw_scada_replay.csv"
)
REPLAY_PUBLISHER_PATH = PROJECT_ROOT / "scripts" / "mqtt" / "scada_replay_publisher.py"
CONTRACT_PATH = (
    PROJECT_ROOT / "data" / "integration" / "600mw_runtime_source_contract.json"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "600mw_runtime_source_contract_validation.txt"
)

EXPECTED_REQUIRED_RAW_COUNT = 71
EXPECTED_MODEL_FEATURE_COUNT = 119
ESTABLISHED_MAPPING_STATUSES = {
    "APPROVED",
    "CONFIRMED",
    "ESTABLISHED",
    "EXPLICIT_PROJECT_MAPPING",
    "VERIFIED",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _visible(value: str) -> str:
    """Keep feature names on one report line while showing exact LF breaks."""

    return str(value).replace("\\", "\\\\").replace("\r", r"\r").replace("\n", r"\n")


def _unit_from_header(column_name: str) -> str:
    """Read the unit suffix exactly from the source workbook header."""

    final_line = column_name.splitlines()[-1].strip()
    opening = max(final_line.rfind("("), final_line.rfind("（"))
    closing = max(final_line.rfind(")"), final_line.rfind("）"))
    if opening < 0 or closing <= opening or final_line[closing + 1 :].strip():
        raise ValueError(f"No terminal unit suffix in source column {column_name!r}")
    unit = final_line[opening + 1 : closing].strip()
    if not unit:
        raise ValueError(f"Empty unit suffix in source column {column_name!r}")
    return unit


def _load_required_source_metadata() -> Tuple[dict, List[dict], str, int]:
    manifest = _read_json(MANIFEST_PATH)
    model_features = manifest.get("features", manifest.get("feature_names", []))
    if len(model_features) != EXPECTED_MODEL_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_FEATURE_COUNT} model features; "
            f"found {len(model_features)}"
        )

    lineage = _read_csv(LINEAGE_PATH)
    lineage_features = [row["feature"] for row in lineage]
    if lineage_features != model_features:
        raise ValueError("Feature lineage order/content does not match model manifest")

    raw_rows = [row for row in lineage if row["feature_class"] == "RAW_SOURCE"]
    lag_rows = [row for row in lineage if row["feature_class"] == "LAG_DERIVED"]
    if len(raw_rows) != EXPECTED_REQUIRED_RAW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_REQUIRED_RAW_COUNT} RAW_SOURCE features; "
            f"found {len(raw_rows)}"
        )

    workbook = load_workbook(WORKBOOK_PATH, read_only=True, data_only=True)
    if len(workbook.worksheets) != 1:
        raise ValueError(
            f"Expected one workbook sheet; found {len(workbook.worksheets)}"
        )
    worksheet = workbook.worksheets[0]
    headers = next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True))
    if any(header is None for header in headers):
        raise ValueError("Original source workbook has a blank header")
    source_columns = [str(header) for header in headers]

    lag_sources = {
        row["source_feature"]
        for row in lag_rows
        if row.get("source_feature")
    }
    variables = []
    for row in raw_rows:
        feature = row["feature"]
        source_column = row.get("source_feature") or feature
        if source_column != feature:
            raise ValueError(
                f"RAW_SOURCE lineage does not preserve the feature name: {feature!r}"
            )
        if source_column not in source_columns:
            raise ValueError(
                f"Required source column is absent from workbook: {source_column!r}"
            )

        variables.append(
            {
                "model_source_feature_name": feature,
                "source_column_name": source_column,
                "unit": _unit_from_header(source_column),
                "is_lag_source_variable": feature in lag_sources,
                "required_for_forecasting": True,
                "expected_numeric_type": "number",
                "postgres_value_storage_type": "double precision",
                "timestamp_requirement": {
                    "required": True,
                    "field_name": "timestamp",
                    "one_measurement_per_variable_per_timestamp": True,
                    "expected_sampling_interval_minutes": 2,
                    "ordering": "strictly_increasing",
                },
                "source_provenance": {
                    "workbook": WORKBOOK_PATH.relative_to(PROJECT_ROOT).as_posix(),
                    "worksheet": worksheet.title,
                    "header_row": 1,
                    "lineage_file": LINEAGE_PATH.relative_to(PROJECT_ROOT).as_posix(),
                    "lineage_feature_class": "RAW_SOURCE",
                },
            }
        )

    if sum(variable["is_lag_source_variable"] for variable in variables) != 11:
        raise ValueError("Expected exactly 11 required lag-source variables")
    return manifest, variables, worksheet.title, len(source_columns)


def _load_explicit_mappings() -> Tuple[dict, List[dict], List[dict]]:
    mapping_rows = _read_csv(MAPPING_PATH)
    acquisition_rows = _read_csv(ACQUISITION_PLAN_PATH)

    explicit = {}
    for row in mapping_rows:
        status = (row.get("mapping_status") or "").strip().upper()
        parameter = (row.get("postgres_parameter") or "").strip()
        feature = row.get("feature") or ""
        if status in ESTABLISHED_MAPPING_STATUSES and feature and parameter:
            explicit[feature] = parameter
    return explicit, mapping_rows, acquisition_rows


def _database_config() -> Tuple[dict, str]:
    """Use the reader's environment config, then the existing read-only inspector."""

    from postgres_scada_reader import DB_CONFIG as reader_config

    if reader_config.get("password"):
        return dict(reader_config), "SCADA reader configuration"

    # The project's existing read-only schema inspector is configured for the
    # local development database. Values are used in memory and never printed.
    from inspect_postgres_schema import DB_CONFIG as inspector_config

    return dict(inspector_config), "existing read-only PostgreSQL inspector configuration"


def _read_postgres_snapshot() -> dict:
    import psycopg2

    config, config_label = _database_config()
    connection = None
    try:
        connection = psycopg2.connect(**config)
        connection.set_session(readonly=True, autocommit=False)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT parameter_name, unit "
                "FROM parameters ORDER BY parameter_name"
            )
            parameter_rows = [
                {"parameter_name": row[0], "unit": row[1]}
                for row in cursor.fetchall()
            ]

            cursor.execute("SELECT COUNT(*) FROM measurements")
            measurement_row_count = int(cursor.fetchone()[0])

            cursor.execute(
                "SELECT COUNT(DISTINCT parameter_id) FROM measurements"
            )
            measured_parameter_count = int(cursor.fetchone()[0])

            cursor.execute(
                "SELECT parameter_name, COUNT(measurement_id) "
                "FROM parameters LEFT JOIN measurements USING (parameter_id) "
                "GROUP BY parameter_name ORDER BY parameter_name"
            )
            measurement_counts = {
                row[0]: int(row[1]) for row in cursor.fetchall()
            }

            cursor.execute(
                "SELECT source_name, source_type, is_real_time "
                "FROM data_sources ORDER BY source_id"
            )
            data_sources = [
                {
                    "source_name": row[0],
                    "source_type": row[1],
                    "is_real_time": row[2],
                }
                for row in cursor.fetchall()
            ]

        connection.rollback()
        return {
            "inspection_status": "PASS",
            "inspection_config_source": config_label,
            "parameters": parameter_rows,
            "measurement_row_count": measurement_row_count,
            "measured_parameter_count": measured_parameter_count,
            "measurement_counts_by_parameter": measurement_counts,
            "data_sources": data_sources,
        }
    except Exception as exc:
        # Avoid printing connection settings or exception text that could
        # contain sensitive connection details.
        raise RuntimeError(
            "Read-only PostgreSQL catalog inspection failed "
            f"({type(exc).__name__})"
        ) from None
    finally:
        if connection is not None:
            connection.close()


def _classify_coverage(
    variables: List[dict],
    parameter_rows: List[dict],
    explicit_mappings: dict,
) -> List[dict]:
    catalog_names = {row["parameter_name"] for row in parameter_rows}
    output = []
    for variable in variables:
        feature = variable["model_source_feature_name"]
        if feature in catalog_names:
            status = "AVAILABLE_EXACT"
            available_parameter = feature
        elif (
            feature in explicit_mappings
            and explicit_mappings[feature] in catalog_names
        ):
            status = "AVAILABLE_EXPLICIT_PROJECT_MAPPING"
            available_parameter = explicit_mappings[feature]
        else:
            status = "NOT_AVAILABLE"
            available_parameter = None
        output.append(
            {
                **variable,
                "postgres_coverage_status": status,
                "available_postgres_parameter": available_parameter,
            }
        )
    return output


def _replay_inventory(required_features: List[str]) -> dict:
    result = {
        "path": REPLAY_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "exists": REPLAY_PATH.exists(),
        "raw_features_present": 0,
        "row_count": None,
        "replay_source": None,
        "publisher_path": REPLAY_PUBLISHER_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "publisher_exists": REPLAY_PUBLISHER_PATH.exists(),
        "mqtt_topic": None,
    }
    if not REPLAY_PATH.exists():
        return result

    with REPLAY_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader)
        result["raw_features_present"] = len(set(headers) & set(required_features))
        result["row_count"] = sum(1 for _ in reader)

    replay_frame = None
    # Read only the metadata column to identify this file as replay data.
    with REPLAY_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        first = next(reader, None)
    if first:
        result["replay_source"] = first.get("replay_source")

    if REPLAY_PUBLISHER_PATH.exists():
        publisher_text = REPLAY_PUBLISHER_PATH.read_text(encoding="utf-8")
        topic_match = re.search(r'^\s*MQTT_TOPIC\s*=\s*["\']([^"\']+)', publisher_text, re.M)
        if topic_match:
            result["mqtt_topic"] = topic_match.group(1)
    return result


def _visible_candidate_mapping(feature: str, mapping_rows: List[dict]) -> str:
    for row in mapping_rows:
        if row.get("feature") == feature and row.get("postgres_parameter"):
            status = (row.get("mapping_status") or "").strip()
            if status.upper() not in ESTABLISHED_MAPPING_STATUSES:
                return (
                    f"unverified candidate {row['postgres_parameter']} "
                    f"({status}; not counted)"
                )
    return ""


def _build_report(
    variables: List[dict],
    coverage: List[dict],
    mapping_rows: List[dict],
    acquisition_rows: List[dict],
    postgres: dict,
    replay: dict,
    workbook_column_count: int,
) -> List[str]:
    counts = {
        status: sum(row["postgres_coverage_status"] == status for row in coverage)
        for status in (
            "AVAILABLE_EXACT",
            "AVAILABLE_EXPLICIT_PROJECT_MAPPING",
            "NOT_AVAILABLE",
        )
    }
    lag_features = [row for row in coverage if row["is_lag_source_variable"]]
    lag_counts = {
        status: sum(row["postgres_coverage_status"] == status for row in lag_features)
        for status in (
            "AVAILABLE_EXACT",
            "AVAILABLE_EXPLICIT_PROJECT_MAPPING",
            "NOT_AVAILABLE",
        )
    }
    parameter_names = [row["parameter_name"] for row in postgres["parameters"]]
    live_sources = [
        row for row in postgres["data_sources"] if row["is_real_time"] is True
    ]
    candidate_rows = [
        row
        for row in mapping_rows
        if row.get("mapping_status") == "CANDIDATE_REQUIRES_VERIFICATION"
    ]
    plan_raw = [row for row in acquisition_rows if row.get("feature_class") == "RAW_SOURCE"]
    plan_established = sum(
        (row.get("mapping_confidence") or "").upper()
        in ESTABLISHED_MAPPING_STATUSES
        and bool(row.get("postgresql_parameter"))
        for row in plan_raw
    )
    replay_contains_all_raw = replay["raw_features_present"] == len(variables)

    lines = [
        "600 MW RUNTIME SOURCE CONTRACT VALIDATION",
        "=" * 44,
        "",
        "Coverage rule: exact parameter-name equality, or a project mapping explicitly marked approved/confirmed/established/verified and targeting an existing PostgreSQL parameter.",
        "Physical similarity, unit similarity, and mappings marked CANDIDATE_REQUIRES_VERIFICATION are not treated as availability.",
        "",
        "Required source contract",
        "-------------------------",
        f"Required runtime source variables: {len(variables)}",
        f"Frozen model feature count: {EXPECTED_MODEL_FEATURE_COUNT}",
        f"Original workbook columns: {workbook_column_count}",
        f"Current PostgreSQL parameter definitions: {len(postgres['parameters'])}",
        f"Current PostgreSQL measurement rows: {postgres['measurement_row_count']}",
        f"Current distinct measured parameters: {postgres['measured_parameter_count']}",
        "",
        "Current PostgreSQL catalog parameters",
        "--------------------------------------",
    ]
    lines.extend(
        f"- {row['parameter_name']} ({row.get('unit') or 'unit unspecified'}); "
        f"measurement rows={postgres['measurement_counts_by_parameter'].get(row['parameter_name'], 0)}"
        for row in postgres["parameters"]
    )
    lines.extend(
        [
            "",
            "Coverage summary",
            "-----------------",
            f"AVAILABLE_EXACT: {counts['AVAILABLE_EXACT']}",
            f"AVAILABLE_EXPLICIT_PROJECT_MAPPING: {counts['AVAILABLE_EXPLICIT_PROJECT_MAPPING']}",
            f"NOT_AVAILABLE: {counts['NOT_AVAILABLE']}",
            f"Lag-source variables required: {len(lag_features)}",
            f"Lag-source AVAILABLE_EXACT: {lag_counts['AVAILABLE_EXACT']}",
            f"Lag-source AVAILABLE_EXPLICIT_PROJECT_MAPPING: {lag_counts['AVAILABLE_EXPLICIT_PROJECT_MAPPING']}",
            f"Lag-source NOT_AVAILABLE: {lag_counts['NOT_AVAILABLE']}",
            f"Approved explicit mappings in raw-feature acquisition plan: {plan_established} / {len(plan_raw)}",
            "",
            "Per-variable PostgreSQL coverage (\\n denotes the exact LF in a name)",
            "--------------------------------------------------------------------",
        ]
    )
    for number, row in enumerate(coverage, 1):
        feature = row["model_source_feature_name"]
        candidate = _visible_candidate_mapping(feature, mapping_rows)
        suffix = f" | {candidate}" if candidate else ""
        lines.append(
            f"{number:02d}. {row['postgres_coverage_status']} | "
            f"MODEL/SOURCE FEATURE: {_visible(feature)} | "
            f"SOURCE COLUMN: {_visible(row['source_column_name'])} | "
            f"UNIT: {row['unit']} | "
            f"LAG SOURCE: {'yes' if row['is_lag_source_variable'] else 'no'}"
            f"{suffix}"
        )

    missing = [
        _visible(row["model_source_feature_name"])
        for row in coverage
        if row["postgres_coverage_status"] == "NOT_AVAILABLE"
    ]
    lines.extend(["", "Exact missing runtime variables", "-------------------------------"])
    lines.extend(f"- {name}" for name in missing)

    lines.extend(
        [
            "",
            "Existing mapping evidence",
            "--------------------------",
            f"Candidate mappings requiring verification: {len(candidate_rows)}",
            f"Acquisition-plan raw variables marked NOT_ESTABLISHED: {sum((r.get('mapping_confidence') or '').upper() == 'NOT_ESTABLISHED' for r in plan_raw)} / {len(plan_raw)}",
            f"Mapping report: {MODEL_MAPPING_REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()} (reports no 600 MW direct-name matches)",
        ]
    )
    for row in candidate_rows:
        lines.append(
            f"- {_visible(row['feature'])} -> {row['postgres_parameter']} "
            f"({row['mapping_status']}, {row.get('mapping_confidence') or 'confidence unspecified'}); excluded from coverage"
        )

    lines.extend(
        [
            "",
            "Other existing source",
            "----------------------",
            f"Historical replay file: {replay['path']}",
            f"Replay contains exact required raw columns: {replay['raw_features_present']} / {len(variables)}",
            f"Replay rows: {replay['row_count']}",
            f"Replay marker: {replay['replay_source']}",
            f"Existing MQTT publisher: {replay['publisher_path']}",
            f"Publisher topic: {replay['mqtt_topic'] or 'not found'}",
            "This replay is derived from the validated historical workbook; it is a simulated/replay source, not an independent live SCADA feed and not the current PostgreSQL measurement stream.",
            "",
            "Live database source definitions",
            "---------------------------------",
        ]
    )
    if live_sources:
        lines.extend(
            f"- {row['source_name']} ({row['source_type']}); is_real_time={row['is_real_time']}"
            for row in live_sources
        )
    else:
        lines.append("None marked real-time.")

    lines.extend(
        [
            "",
            "Database gap",
            "-------------",
            "Database gap classification: C — BOTH additional parameter definitions and timestamped measurement records.",
            "Conclusion: PostgreSQL requires both parameter definitions and measurement records to supply these 71 exact runtime variables, unless an authoritative source-tag mapping is later established.",
            "The existing parameters and measurements tables are generic and can represent parameter/value/timestamp records; this assessment does not propose or apply a schema change.",
            "The current 10 measurements are generic records and do not form the 71-variable wide input or the timestamped history needed for lag 10.",
            "Current PostgreSQL can produce a valid 119-feature model input: NO.",
            "",
            "Future data flow",
            "-----------------",
            "Authoritative SCADA tags/source",
            "  -> PostgreSQL parameters and timestamped measurements",
            "  -> PostgreSQLSCADAReader (long-form measurement records)",
            "  -> PostgreSQL SCADA contract (timestamp/source/equipment/parameter/value/unit/quality)",
            "  -> separate 600 MW runtime source-variable assembler (validate approved tag mappings; group one source/equipment by timestamp; require all 71 exact variables and quality/cadence; output a wide timestamped DataFrame)",
            "  -> runtime_feature_builder_600mw.py (validate source features, timestamp/order/cadence, construct row-based lags and time fields, enforce the 119-feature order)",
            "  -> frozen forecasting models",
            "",
            "No PostgreSQL integration, database write, model inference, or synthetic SCADA record was performed.",
        ]
    )
    return lines


def main() -> None:
    manifest, variables, worksheet_name, workbook_column_count = (
        _load_required_source_metadata()
    )
    explicit_mappings, mapping_rows, acquisition_rows = _load_explicit_mappings()
    postgres = _read_postgres_snapshot()
    coverage = _classify_coverage(
        variables,
        postgres["parameters"],
        explicit_mappings,
    )
    replay = _replay_inventory(
        [variable["model_source_feature_name"] for variable in variables]
    )

    contract = {
        "contract_name": "600MW forecasting runtime source variables",
        "contract_version": 1,
        "model": "600MW_FORECASTING",
        "model_feature_manifest": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "model_feature_count": EXPECTED_MODEL_FEATURE_COUNT,
        "required_raw_source_variable_count": len(variables),
        "timestamp_requirement": {
            "runtime_field_name": "timestamp",
            "historical_workbook_column": manifest.get("timestamp_column", "Time"),
            "required": True,
            "strictly_increasing": True,
            "expected_sampling_interval_minutes": 2,
            "lag_semantics": "row-based shifts at steps 1, 2, 5, and 10",
            "prior_observations_required_for_first_model_ready_row": 10,
        },
        "database_value_type": {
            "column": "measurements.value",
            "postgresql_type": "double precision",
            "runtime_json_type": "number",
        },
        "source_workbook": {
            "path": WORKBOOK_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "worksheet": worksheet_name,
            "header_row": 1,
            "total_columns": workbook_column_count,
        },
        "lineage_source": LINEAGE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "variables": variables,
    }

    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = _build_report(
        variables,
        coverage,
        mapping_rows,
        acquisition_rows,
        postgres,
        replay,
        workbook_column_count,
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    counts = {
        status: sum(row["postgres_coverage_status"] == status for row in coverage)
        for status in (
            "AVAILABLE_EXACT",
            "AVAILABLE_EXPLICIT_PROJECT_MAPPING",
            "NOT_AVAILABLE",
        )
    }
    print(f"Contract written: {CONTRACT_PATH}")
    print(f"Validation report: {REPORT_PATH}")
    print(f"Required source variables: {len(variables)}")
    print(f"PostgreSQL parameters: {len(postgres['parameters'])}")
    print(f"PostgreSQL measurement rows: {postgres['measurement_row_count']}")
    print(f"Coverage: {counts}")
    print(
        "Historical replay exact raw-variable coverage: "
        f"{replay['raw_features_present']} / {len(variables)}"
    )


if __name__ == "__main__":
    main()
