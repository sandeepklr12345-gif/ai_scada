import pandas as pd

from postgres_scada_reader import (
    PostgreSQLSCADAReader,
    DB_CONFIG,
)


REQUIRED_DB_COLUMNS = [
    "timestamp",
    "source_id",
    "plant_id",
    "equipment_id",
    "parameter_name",
    "value",
    "unit",
    "quality_status",
]


class PostgreSQLSCADAContractAdapter:

    def __init__(self, db_config):
        self.reader = PostgreSQLSCADAReader(db_config)

    def read_contract(
        self,
        limit=None,
        source_id=None,
        equipment_id=None,
        parameter_id=None,
    ):
        df = self.reader.read_measurements(
            limit=limit,
            source_id=source_id,
            equipment_id=equipment_id,
            parameter_id=parameter_id,
        )

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "source_id",
                    "plant_id",
                    "equipment_id",
                    "parameter",
                    "value",
                    "unit",
                    "quality",
                ]
            )

        missing = [
            column
            for column in REQUIRED_DB_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required database columns: {missing}"
            )

        contract_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["timestamp"]),
                "source_id": df["source_id"],
                "plant_id": df["plant_id"],
                "equipment_id": df["equipment_id"],
                "parameter": df["parameter_name"],
                "value": pd.to_numeric(df["value"]),
                "unit": df["unit"],
                "quality": df["quality_status"],
            }
        )

        return contract_df


def validate_contract_dataframe(df):
    required_columns = [
        "timestamp",
        "source_id",
        "plant_id",
        "equipment_id",
        "parameter",
        "value",
        "unit",
        "quality",
    ]

    if list(df.columns) != required_columns:
        raise ValueError(
            "SCADA contract schema mismatch.\n"
            f"Expected: {required_columns}\n"
            f"Received: {list(df.columns)}"
        )

    if df.empty:
        return True

    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamp detected.")

    if df["value"].isna().any():
        raise ValueError("Invalid numeric value detected.")

    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError("SCADA value column is not numeric.")

    for column in [
        "source_id",
        "plant_id",
        "equipment_id",
        "parameter",
        "unit",
        "quality",
    ]:
        if df[column].isna().any():
            raise ValueError(
                f"Missing value detected in column: {column}"
            )

    return True


if __name__ == "__main__":

    print("=" * 70)
    print("POSTGRESQL → SCADA CONTRACT ADAPTER")
    print("=" * 70)

    adapter = PostgreSQLSCADAContractAdapter(DB_CONFIG)

    # The database currently has no permanent measurements.
    df = adapter.read_contract()

    print("\nContract rows returned :", len(df))
    print("Contract columns      :", len(df.columns))

    print("\nSCADA CONTRACT SCHEMA")

    for column in df.columns:
        print(f"  {column}")

    validate_contract_dataframe(df)

    print("\nContract validation: PASS")

    if df.empty:
        print("\nNo permanent SCADA measurements currently stored.")
        print("This is expected at this stage.")

    print("\n" + "=" * 70)
    print("POSTGRESQL → SCADA CONTRACT ADAPTER: PASS")
    print("=" * 70)