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
                DELETE FROM measurements
                WHERE measurement_id = 4;
            """)

            measurement_deleted = cursor.rowcount

            cursor.execute("""
                DELETE FROM ai_predictions
                WHERE prediction_id = 1;
            """)

            prediction_deleted = cursor.rowcount

        connection.commit()

        print("=" * 70)
        print("POSTGRESQL TEST DATA CLEANUP")
        print("=" * 70)

        print(f"\nMeasurements deleted : {measurement_deleted}")
        print(f"Predictions deleted  : {prediction_deleted}")

        print("\n" + "=" * 70)
        print("POSTGRESQL TEST DATA CLEANUP: PASS")
        print("=" * 70)

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()