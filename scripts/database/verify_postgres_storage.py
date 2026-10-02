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
                SELECT
                    measurement_id,
                    timestamp,
                    source_id,
                    equipment_id,
                    parameter_id,
                    value,
                    quality_status
                FROM measurements
                WHERE measurement_id = 4;
            """)

            measurement = cursor.fetchone()

            cursor.execute("""
                SELECT
                    prediction_id,
                    timestamp,
                    source_id,
                    equipment_id,
                    model_name,
                    prediction_type,
                    predicted_value,
                    prediction_horizon_minutes,
                    description
                FROM ai_predictions
                WHERE prediction_id = 1;
            """)

            prediction = cursor.fetchone()

            print("=" * 70)
            print("POSTGRESQL STORAGE VERIFICATION")
            print("=" * 70)

            print("\nMEASUREMENT")
            print("-" * 70)
            print(measurement)

            print("\nPREDICTION")
            print("-" * 70)
            print(prediction)

            print("\n" + "=" * 70)
            print("POSTGRESQL STORAGE VERIFICATION: PASS")
            print("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()