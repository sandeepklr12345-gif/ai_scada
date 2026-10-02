import psycopg2
import pandas as pd

from postgres_scada_reader import (
    PostgreSQLSCADAReader,
    DB_CONFIG,
)


TEST_TIMESTAMP = "2026-09-29 19:00:00"
TEST_SOURCE_ID = 6
TEST_EQUIPMENT_ID = 3
TEST_PARAMETER_ID = 1
TEST_VALUE = 346.75
TEST_QUALITY = "GOOD"


def run_test():

    print("=" * 70)
    print("POSTGRESQL SCADA READER DATABASE TEST")
    print("=" * 70)

    connection = None

    try:
        # ============================================================
        # 1. CONNECT
        # ============================================================

        connection = psycopg2.connect(**DB_CONFIG)
        connection.autocommit = False

        cursor = connection.cursor()

        print("\nDatabase connection: PASS")

        # ============================================================
        # 2. INSERT TEMPORARY TEST RECORD
        # ============================================================

        insert_query = """
            INSERT INTO measurements
            (
                timestamp,
                source_id,
                equipment_id,
                parameter_id,
                value,
                quality_status
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        cursor.execute(
            insert_query,
            (
                TEST_TIMESTAMP,
                TEST_SOURCE_ID,
                TEST_EQUIPMENT_ID,
                TEST_PARAMETER_ID,
                TEST_VALUE,
                TEST_QUALITY,
            ),
        )

        print("Temporary measurement inserted: PASS")

        # ============================================================
        # 3. COMMIT
        #
        # Reader uses a separate connection, so the row must be
        # committed before the reader can see it.
        # ============================================================

        connection.commit()

        print("Temporary measurement committed: PASS")

        # Close the first connection before using the reader.
        connection.close()
        connection = None

        # ============================================================
        # 4. READ USING ACTUAL SCADA READER
        # ============================================================

        reader = PostgreSQLSCADAReader(DB_CONFIG)

        df = reader.read_measurements(
            source_id=TEST_SOURCE_ID,
            equipment_id=TEST_EQUIPMENT_ID,
            parameter_id=TEST_PARAMETER_ID,
        )

        print(f"\nRows returned: {len(df)}")

        if df.empty:
            raise RuntimeError(
                "FAIL: Temporary measurement was not returned "
                "by PostgreSQLSCADAReader."
            )

        # ============================================================
        # 5. FIND OUR TEST RECORD
        # ============================================================

        matching = df[
            (df["value"] == TEST_VALUE)
            & (df["quality_status"] == TEST_QUALITY)
        ]

        if matching.empty:
            raise RuntimeError(
                "FAIL: Reader returned rows, but the expected "
                "test measurement was not found."
            )

        row = matching.iloc[0]

        print("\nSAMPLE JOINED RECORD")
        print(row.to_dict())

        # ============================================================
        # 6. VALIDATE METADATA JOINS
        # ============================================================

        checks = {
            "source_name":
                row["source_name"] == "ESP32",

            "equipment_name":
                row["equipment_name"] == "Generator",

            "equipment_type":
                row["equipment_type"] == "generator",

            "plant_name":
                row["plant_name"] == "AI-SCADA Demo Thermal Plant",

            "parameter_name":
                row["parameter_name"] == "power_output",

            "unit":
                row["unit"] == "MW",

            "category":
                row["category"] == "electrical",

            "value":
                float(row["value"]) == TEST_VALUE,

            "quality_status":
                row["quality_status"] == TEST_QUALITY,
        }

        print("\nJOIN VALIDATION")

        all_passed = True

        for name, passed in checks.items():

            status = "PASS" if passed else "FAIL"

            print(f"  {name:<20}: {status}")

            if not passed:
                all_passed = False

        if not all_passed:
            raise RuntimeError(
                "One or more metadata validation checks failed."
            )

        print("\nReader validation: PASS")

    except Exception as error:

        if connection is not None:
            connection.rollback()

        print("\nTEST FAILED")
        print(error)

        raise

    finally:

        if connection is not None:
            connection.close()

    # ================================================================
    # 7. CLEANUP TEST RECORD
    # ================================================================

    cleanup_connection = None

    try:

        cleanup_connection = psycopg2.connect(**DB_CONFIG)

        cleanup_cursor = cleanup_connection.cursor()

        delete_query = """
            DELETE FROM measurements
            WHERE timestamp = %s
              AND source_id = %s
              AND equipment_id = %s
              AND parameter_id = %s
              AND value = %s
              AND quality_status = %s
        """

        cleanup_cursor.execute(
            delete_query,
            (
                TEST_TIMESTAMP,
                TEST_SOURCE_ID,
                TEST_EQUIPMENT_ID,
                TEST_PARAMETER_ID,
                TEST_VALUE,
                TEST_QUALITY,
            ),
        )

        deleted_rows = cleanup_cursor.rowcount

        cleanup_connection.commit()

        print("\nCLEANUP")
        print(f"Test rows deleted: {deleted_rows}")

        if deleted_rows == 0:
            raise RuntimeError(
                "WARNING: Test record could not be found during cleanup."
            )

        print("Cleanup: PASS")

    finally:

        if cleanup_connection is not None:
            cleanup_connection.close()

    # ================================================================
    # 8. VERIFY DATABASE IS CLEAN
    # ================================================================

    verification_connection = None

    try:

        verification_connection = psycopg2.connect(**DB_CONFIG)

        verification_query = """
            SELECT COUNT(*)
            FROM measurements
            WHERE timestamp = %s
              AND source_id = %s
              AND equipment_id = %s
              AND parameter_id = %s
              AND value = %s
              AND quality_status = %s
        """

        verification_df = pd.read_sql_query(
            verification_query,
            verification_connection,
            params=[
                TEST_TIMESTAMP,
                TEST_SOURCE_ID,
                TEST_EQUIPMENT_ID,
                TEST_PARAMETER_ID,
                TEST_VALUE,
                TEST_QUALITY,
            ],
        )

        remaining_rows = int(verification_df.iloc[0, 0])

        print("\nDATABASE CLEANUP VERIFICATION")
        print(f"Matching test rows remaining: {remaining_rows}")

        if remaining_rows != 0:
            raise RuntimeError(
                "FAIL: Test measurement still exists."
            )

        print("Database clean: PASS")

    finally:

        if verification_connection is not None:
            verification_connection.close()

    print("\n" + "=" * 70)
    print("POSTGRESQL SCADA READER DATABASE TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    run_test()