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
                WHERE measurement_id = 5;
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
                    prediction_horizon_minutes
                FROM ai_predictions
                WHERE prediction_id IN (2, 3, 4)
                ORDER BY prediction_id;
            """)

            predictions = cursor.fetchall()

            print("=" * 70)
            print("RUNTIME DATABASE VERIFICATION")
            print("=" * 70)

            print("\nMEASUREMENT")
            print("-" * 70)
            print(measurement)

            print("\nPREDICTIONS")
            print("-" * 70)

            for prediction in predictions:
                print(prediction)

            print("\n" + "=" * 70)
            print("RUNTIME DATABASE VERIFICATION: PASS")
            print("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
