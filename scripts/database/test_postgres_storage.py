from datetime import datetime

from scripts.database.postgres_storage import PostgreSQLStorage


def main():

    print("=" * 70)
    print("POSTGRESQL STORAGE WRITE TEST")
    print("=" * 70)

    storage = PostgreSQLStorage()

    test_timestamp = datetime(2026, 10, 1, 12, 0, 0)

    print("\n1. Testing measurement insertion...")

    measurement_id = storage.save_measurement(
        timestamp=test_timestamp,
        source_id=1,
        equipment_id=3,
        parameter_id=1,
        value=346.75,
        quality_status="TEST",
    )

    print(f"Measurement inserted: ID={measurement_id}")

    print("\n2. Testing prediction insertion...")

    prediction_id = storage.save_prediction(
        timestamp=test_timestamp,
        source_id=1,
        equipment_id=3,
        model_name="600MW_TEST",
        prediction_type="power_forecast",
        predicted_value=346.75,
        confidence=None,
        risk_level=None,
        prediction_horizon_minutes=2,
        description="Temporary PostgreSQL storage integration test",
    )

    print(f"Prediction inserted: ID={prediction_id}")

    print("\n3. Write test completed.")

    print("\n" + "=" * 70)
    print("POSTGRESQL STORAGE WRITE TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()