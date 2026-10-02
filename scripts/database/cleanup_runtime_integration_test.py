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
                WHERE measurement_id BETWEEN 5 AND 15;
            """)

            measurement_deleted = cursor.rowcount

            cursor.execute("""
                DELETE FROM ai_predictions
                WHERE prediction_id BETWEEN 2 AND 34;
            """)

            prediction_deleted = cursor.rowcount

        connection.commit()

        print("=" * 70)
        print("RUNTIME INTEGRATION TEST CLEANUP")
        print("=" * 70)

        print(f"\nMeasurements deleted : {measurement_deleted}")
        print(f"Predictions deleted  : {prediction_deleted}")

        if measurement_deleted != 11:
            raise RuntimeError(
                f"Expected 11 measurements to be deleted, "
                f"found {measurement_deleted}"
            )

        if prediction_deleted != 33:
            raise RuntimeError(
                f"Expected 33 predictions to be deleted, "
                f"found {prediction_deleted}"
            )

        print("\n" + "=" * 70)
        print("RUNTIME INTEGRATION TEST CLEANUP: PASS")
        print("=" * 70)

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()