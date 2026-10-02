from datetime import datetime
from typing import Any, Dict


# ============================================================
# SCADA DATA CONTRACT
# ============================================================

SCADA_REQUIRED_FIELDS = [
    "timestamp",
    "source_id",
    "plant_id",
    "equipment_id",
    "parameter",
    "value",
    "unit",
    "quality",
]


# ============================================================
# VALIDATION
# ============================================================

def validate_scada_record(record: Dict[str, Any]) -> Dict[str, Any]:

    if not isinstance(record, dict):
        raise TypeError("SCADA record must be a dictionary")

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    missing = [
        field
        for field in SCADA_REQUIRED_FIELDS
        if field not in record
    ]

    if missing:
        raise ValueError(
            f"Missing required SCADA fields: {missing}"
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    try:
        timestamp = datetime.fromisoformat(
            str(record["timestamp"])
        )
    except ValueError as exc:
        raise ValueError(
            "Invalid SCADA timestamp"
        ) from exc

    # --------------------------------------------------------
    # Numeric value
    # --------------------------------------------------------

    try:
        value = float(record["value"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SCADA value must be numeric"
        ) from exc

    # --------------------------------------------------------
    # String fields
    # --------------------------------------------------------

    string_fields = [
        "source_id",
        "plant_id",
        "equipment_id",
        "parameter",
        "unit",
        "quality",
    ]

    for field in string_fields:

        if not isinstance(record[field], str):
            raise TypeError(
                f"SCADA field '{field}' must be a string"
            )

        if not record[field].strip():
            raise ValueError(
                f"SCADA field '{field}' cannot be empty"
            )

    # --------------------------------------------------------
    # Normalized record
    # --------------------------------------------------------

    return {
        "timestamp": timestamp.isoformat(),
        "source_id": record["source_id"],
        "plant_id": record["plant_id"],
        "equipment_id": record["equipment_id"],
        "parameter": record["parameter"],
        "value": value,
        "unit": record["unit"],
        "quality": record["quality"],
    }


# ============================================================
# BATCH VALIDATION
# ============================================================

def validate_scada_records(records):

    if not isinstance(records, list):
        raise TypeError(
            "SCADA records must be provided as a list"
        )

    validated_records = []

    for index, record in enumerate(records):

        try:
            validated_records.append(
                validate_scada_record(record)
            )

        except (TypeError, ValueError) as exc:

            raise ValueError(
                f"Invalid SCADA record at index {index}: {exc}"
            ) from exc

    return validated_records


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SCADA DATA CONTRACT")
    print("=" * 70)

    sample_record = {
        "timestamp": "2026-09-29T19:00:00",
        "source_id": "ESP32_001",
        "plant_id": "PLANT_001",
        "equipment_id": "GENERATOR_01",
        "parameter": "power_output",
        "value": 346.75,
        "unit": "MW",
        "quality": "GOOD",
    }

    validated = validate_scada_record(sample_record)

    print("\nVALID RECORD")
    print(validated)

    print("\nRequired fields:")
    for field in SCADA_REQUIRED_FIELDS:
        print(f"  {field}")

    print("\nSCADA DATA CONTRACT: PASS")