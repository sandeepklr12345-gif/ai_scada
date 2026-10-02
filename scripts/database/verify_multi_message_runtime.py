import os
import psycopg2


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "ai_scada",
    "user": "postgres",
    "password": os.getenv("AI_SCADA_DB_PASSWORD"),
}


def main():

    connection = psycopg2.connect(**DB_CONFIG)

    try:
        with connection.cursor() as cursor:

            cursor.execute("""
                SELECT COUNT(*)
                FROM measurements
                WHERE measurement_id BETWEEN 5 AND 15;
            """)

            measurement_count = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*)
                FROM ai_predictions
                WHERE prediction_id BETWEEN 2 AND 34;
            """)

            prediction_count = cursor.fetchone()[0]

            cursor.execute("""
                SELECT
                    MIN(timestamp),
                    MAX(timestamp),
                    COUNT(DISTINCT timestamp)
                FROM measurements
                WHERE measurement_id BETWEEN 5 AND 15;
            """)

            timestamp_info = cursor.fetchone()

            print("=" * 70)
            print("MULTI-MESSAGE DATABASE VERIFICATION")
            print("=" * 70)

            print(f"\nMeasurements verified : {measurement_count}")
            print(f"Predictions verified  : {prediction_count}")

            print("\nMeasurement timestamp range:")
            print(f"First timestamp       : {timestamp_info[0]}")
            print(f"Last timestamp        : {timestamp_info[1]}")
            print(f"Distinct timestamps   : {timestamp_info[2]}")

            print("\nExpected:")
            print("Measurements          : 11")
            print("Predictions           : 33")

            if measurement_count != 11:
                raise RuntimeError(
                    f"Expected 11 measurements, found {measurement_count}"
                )

            if prediction_count != 33:
                raise RuntimeError(
                    f"Expected 33 predictions, found {prediction_count}"
                )

            print("\n" + "=" * 70)
            print("MULTI-MESSAGE DATABASE VERIFICATION: PASS")
            print("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()