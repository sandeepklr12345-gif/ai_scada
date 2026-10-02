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

            print("=" * 70)
            print("RUNTIME INTEGRATION CLEANUP VERIFICATION")
            print("=" * 70)

            print(f"\nRemaining test measurements : {measurement_count}")
            print(f"Remaining test predictions  : {prediction_count}")

            if measurement_count != 0:
                raise RuntimeError(
                    f"Expected 0 test measurements, found {measurement_count}"
                )

            if prediction_count != 0:
                raise RuntimeError(
                    f"Expected 0 test predictions, found {prediction_count}"
                )

            print("\n" + "=" * 70)
            print("RUNTIME INTEGRATION CLEANUP VERIFICATION: PASS")
            print("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()