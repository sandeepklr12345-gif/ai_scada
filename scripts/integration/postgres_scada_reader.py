import os

import psycopg2
import pandas as pd


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.getenv("AI_SCADA_DB_HOST", "localhost"),
    "port": int(os.getenv("AI_SCADA_DB_PORT", "5432")),
    "database": os.getenv("AI_SCADA_DB_NAME", "ai_scada"),
    "user": os.getenv("AI_SCADA_DB_USER", "postgres"),
    "password": os.getenv("AI_SCADA_DB_PASSWORD"),
}


# ============================================================
# SCADA READER
# ============================================================

class PostgreSQLSCADAReader:

    def __init__(self, db_config):

        self.db_config = db_config

    # --------------------------------------------------------
    # CONNECTION
    # --------------------------------------------------------

    def _connect(self):

        return psycopg2.connect(**self.db_config)

    # --------------------------------------------------------
    # READ MEASUREMENTS
    # --------------------------------------------------------

    def read_measurements(
        self,
        limit=None,
        source_id=None,
        equipment_id=None,
        parameter_id=None,
    ):

        query = """
            SELECT
                m.measurement_id,
                m.timestamp,
                m.source_id,
                ds.source_name,
                m.equipment_id,
                e.equipment_name,
                e.equipment_type,
                e.plant_id,
                p.plant_name,
                m.parameter_id,
                prm.parameter_name,
                prm.unit,
                prm.category,
                m.value,
                m.quality_status

            FROM measurements m

            LEFT JOIN data_sources ds
                ON m.source_id = ds.source_id

            LEFT JOIN equipment e
                ON m.equipment_id = e.equipment_id

            LEFT JOIN plants p
                ON e.plant_id = p.plant_id

            LEFT JOIN parameters prm
                ON m.parameter_id = prm.parameter_id

            WHERE 1 = 1
        """

        params = []

        if source_id is not None:
            query += " AND m.source_id = %s"
            params.append(source_id)

        if equipment_id is not None:
            query += " AND m.equipment_id = %s"
            params.append(equipment_id)

        if parameter_id is not None:
            query += " AND m.parameter_id = %s"
            params.append(parameter_id)

        query += " ORDER BY m.timestamp ASC"

        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)

        connection = None

        try:

            connection = self._connect()

            df = pd.read_sql_query(
                query,
                connection,
                params=params,
            )

            return df

        finally:

            if connection is not None:
                connection.close()


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("POSTGRESQL SCADA READER")
    print("=" * 70)

    reader = PostgreSQLSCADAReader(DB_CONFIG)

    df = reader.read_measurements()

    print("\nDatabase connection: PASS")

    print(f"Rows returned    : {len(df)}")
    print(f"Columns returned : {len(df.columns)}")

    print("\nSCHEMA")

    for column in df.columns:
        print(f"  {column}")

    if df.empty:

        print("\nNo SCADA measurements currently stored.")

        print(
            "This is expected because the measurements "
            "table currently contains 0 rows."
        )

    else:

        print("\nSAMPLE DATA")
        print(df.head())

    print("\n" + "=" * 70)
    print("POSTGRESQL SCADA READER: PASS")
    print("=" * 70)
