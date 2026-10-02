import os
from datetime import datetime

import psycopg2


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
# POSTGRESQL STORAGE
# ============================================================

class PostgreSQLStorage:

    def __init__(self, db_config=None):
        self.db_config = db_config or DB_CONFIG

    # --------------------------------------------------------
    # CONNECTION
    # --------------------------------------------------------

    def _connect(self):
        return psycopg2.connect(**self.db_config)

    # --------------------------------------------------------
    # HEALTH CHECK
    # --------------------------------------------------------

    def health_check(self):
        connection = None

        try:
            connection = self._connect()

            with connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()

            return result[0] == 1

        finally:
            if connection is not None:
                connection.close()

    # --------------------------------------------------------
    # SAVE MEASUREMENT
    # --------------------------------------------------------

    def save_measurement(
        self,
        timestamp,
        source_id,
        equipment_id,
        parameter_id,
        value,
        quality_status="GOOD",
    ):
        connection = None

        try:
            connection = self._connect()

            query = """
                INSERT INTO measurements (
                    timestamp,
                    source_id,
                    equipment_id,
                    parameter_id,
                    value,
                    quality_status
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING measurement_id;
            """

            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        timestamp,
                        source_id,
                        equipment_id,
                        parameter_id,
                        float(value),
                        quality_status,
                    ),
                )

                measurement_id = cursor.fetchone()[0]

            connection.commit()

            return measurement_id

        except Exception:
            if connection is not None:
                connection.rollback()
            raise

        finally:
            if connection is not None:
                connection.close()

    # --------------------------------------------------------
    # SAVE AI PREDICTION
    # --------------------------------------------------------

    def save_prediction(
        self,
        timestamp,
        source_id,
        equipment_id,
        model_name,
        prediction_type,
        predicted_value,
        confidence=None,
        risk_level=None,
        prediction_horizon_minutes=None,
        description=None,
    ):
        connection = None

        try:
            connection = self._connect()

            query = """
                INSERT INTO ai_predictions (
                    timestamp,
                    source_id,
                    equipment_id,
                    model_name,
                    prediction_type,
                    predicted_value,
                    confidence,
                    risk_level,
                    prediction_horizon_minutes,
                    description
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                RETURNING prediction_id;
            """

            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        timestamp,
                        source_id,
                        equipment_id,
                        model_name,
                        prediction_type,
                        predicted_value,
                        confidence,
                        risk_level,
                        prediction_horizon_minutes,
                        description,
                    ),
                )

                prediction_id = cursor.fetchone()[0]

            connection.commit()

            return prediction_id

        except Exception:
            if connection is not None:
                connection.rollback()
            raise

        finally:
            if connection is not None:
                connection.close()

    # --------------------------------------------------------
    # SAVE ALARM / EVENT
    # --------------------------------------------------------

    def save_alarm(
        self,
        timestamp,
        source_id,
        equipment_id,
        event_type,
        alarm_code=None,
        severity=None,
        status=None,
        description=None,
    ):
        connection = None

        try:
            connection = self._connect()

            query = """
                INSERT INTO alarms_events (
                    timestamp,
                    source_id,
                    equipment_id,
                    event_type,
                    alarm_code,
                    severity,
                    status,
                    description
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                RETURNING event_id;
            """

            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        timestamp,
                        source_id,
                        equipment_id,
                        event_type,
                        alarm_code,
                        severity,
                        status,
                        description,
                    ),
                )

                event_id = cursor.fetchone()[0]

            connection.commit()

            return event_id

        except Exception:
            if connection is not None:
                connection.rollback()
            raise

        finally:
            if connection is not None:
                connection.close()


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("POSTGRESQL STORAGE SELF TEST")
    print("=" * 70)

    storage = PostgreSQLStorage()

    print("\nTesting database connection...")

    if storage.health_check():
        print("Database connection: PASS")
    else:
        print("Database connection: FAIL")

    print("\nDatabase:")
    print(f"  Host     : {DB_CONFIG['host']}")
    print(f"  Port     : {DB_CONFIG['port']}")
    print(f"  Database : {DB_CONFIG['database']}")
    print(f"  User     : {DB_CONFIG['user']}")

    print("\n" + "=" * 70)
    print("POSTGRESQL STORAGE SELF TEST: PASS")
    print("=" * 70)