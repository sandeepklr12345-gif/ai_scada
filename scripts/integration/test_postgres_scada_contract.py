import psycopg2
import pandas as pd

from postgres_to_scada_contract import (
    PostgreSQLSCADAContractAdapter,
)
from postgres_scada_reader import DB_CONFIG


TEST_TIMESTAMP = "2026-09-29 19:05:00"
TEST_SOURCE_ID = 6
TEST_EQUIPMENT_ID = 3
TEST_PARAMETER_ID = 1
TEST_VALUE = 346.75
TEST_QUALITY = "GOOD"


def run_test():

    print("=" * 70)
    print("POSTGRESQL → SCADA CONTRACT TRANSFORMATION TEST")
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
        # 2. INSERT TEMPORARY MEASUREMENT
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

        connection.commit()

        print("Temporary measurement inserted: PASS")
        print("Temporary measurement committed: PASS")

        connection.close()
        connection = None

        # ============================================================
        # 3. READ THROUGH CONTRACT ADAPTER
        # ============================================================

        adapter = PostgreSQLSCADAContractAdapter(DB_CONFIG)

        df = adapter.read_contract(
            source_id=TEST_SOURCE_ID,
            equipment_id=TEST_EQUIPMENT_ID,
            parameter_id=TEST_PARAMETER_ID,
        )

        print(f"\nContract rows returned: {len(df)}")

        if df.empty:
            raise RuntimeError(
                "FAIL: Contract adapter returned no rows."
            )

        # ============================================================
        # 4. FIND TEST RECORD
        # ============================================================

        matching = df[
            (df["value"] == TEST_VALUE)
            & (df["quality"] == TEST_QUALITY)
        ]

        if matching.empty:
            raise RuntimeError(
                "FAIL: Expected test record was not found."
            )

        row = matching.iloc[0]

        # ============================================================
        # 5. DISPLAY CONTRACT RECORD
        # ============================================================

        print("\nSCADA CONTRACT RECORD")

        print({
            "timestamp": row["timestamp"],
            "source_id": row["source_id"],
            "plant_id": row["plant_id"],
            "equipment_id": row["equipment_id"],
            "parameter": row["parameter"],
            "value": row["value"],
            "unit": row["unit"],
            "quality": row["quality"],
        })

        # ============================================================
        # 6. VALIDATE ALL 8 CONTRACT FIELDS
        # ============================================================

        expected = {
            "source_id": TEST_SOURCE_ID,
            "plant_id": 1,
            "equipment_id": TEST_EQUIPMENT_ID,
            "parameter": "power_output",
            "value": TEST_VALUE,
            "unit": "MW",
            "quality": TEST_QUALITY,
        }

        print("\nCONTRACT FIELD VALIDATION")

        all_passed = True

        for field, expected_value in expected.items():

            actual_value = row[field]

            passed = actual_value == expected_value

            print(
                f"  {field:<15}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

            if not passed:
                print(
                    f"      Expected: {expected_value}"
                )
                print(
                    f"      Actual  : {actual_value}"
                )
                all_passed = False

        # Timestamp validation
        timestamp_valid = (
            pd.Timestamp(row["timestamp"])
            == pd.Timestamp(TEST_TIMESTAMP)
        )

        print(
            f"  {'timestamp':<15}: "
            f"{'PASS' if timestamp_valid else 'FAIL'}"
        )

        if not timestamp_valid:
            all_passed = False

        if not all_passed:
            raise RuntimeError(
                "One or more SCADA contract fields failed validation."
            )

        print("\nContract transformation: PASS")

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
    # 7. DELETE TEMPORARY TEST RECORD
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

        if deleted_rows != 1:
            raise RuntimeError(
                "FAIL: Expected exactly one test row to be deleted."
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

        remaining_rows = int(
            verification_df.iloc[0, 0]
        )

        print("\nDATABASE CLEANUP VERIFICATION")
        print(
            f"Matching test rows remaining: "
            f"{remaining_rows}"
        )

        if remaining_rows != 0:
            raise RuntimeError(
                "FAIL: Test record still exists."
            )

        print("Database clean: PASS")

    finally:

        if verification_connection is not None:
            verification_connection.close()

    print("\n" + "=" * 70)
    print("POSTGRESQL → SCADA CONTRACT TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    run_test()