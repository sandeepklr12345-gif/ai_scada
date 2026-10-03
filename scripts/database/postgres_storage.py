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
    # GET LATEST MEASUREMENT
    # --------------------------------------------------------

    def get_latest_measurement(
        self,
        source_id=None,
        equipment_id=None,
        parameter_id=None,
    ):
        connection = None

        try:
            connection = self._connect()

            query = """
                SELECT
                    measurement_id,
                    timestamp,
                    source_id,
                    equipment_id,
                    parameter_id,
                    value,
                    quality_status
                FROM measurements
                WHERE 1 = 1
            """

            params = []

            if source_id is not None:
                query += " AND source_id = %s"
                params.append(source_id)

            if equipment_id is not None:
                query += " AND equipment_id = %s"
                params.append(equipment_id)

            if parameter_id is not None:
                query += " AND parameter_id = %s"
                params.append(parameter_id)

            query += """
                ORDER BY measurement_id DESC
                LIMIT 1;
            """

            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()

            if row is None:
                return None

            return {
                "measurement_id": row[0],
                "timestamp": row[1],
                "source_id": row[2],
                "equipment_id": row[3],
                "parameter_id": row[4],
                "value": float(row[5]),
                "quality_status": row[6],
            }

        finally:
            if connection is not None:
                connection.close()

    # --------------------------------------------------------
    # GET LATEST PREDICTIONS
    # --------------------------------------------------------

    def get_latest_predictions(
        self,
        source_id=None,
        equipment_id=None,
        prediction_type="power_forecast",
    ):
        connection = None

        try:
            connection = self._connect()

            query = """
                SELECT
                    prediction_id,
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
                FROM ai_predictions
                WHERE prediction_type = %s
            """

            params = [prediction_type]

            if source_id is not None:
                query += " AND source_id = %s"
                params.append(source_id)

            if equipment_id is not None:
                query += " AND equipment_id = %s"
                params.append(equipment_id)

            query += """
                ORDER BY timestamp DESC, prediction_horizon_minutes ASC
                LIMIT 3;
            """

            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()

            return [
                {
                    "prediction_id": row[0],
                    "timestamp": row[1],
                    "source_id": row[2],
                    "equipment_id": row[3],
                    "model_name": row[4],
                    "prediction_type": row[5],
                    "predicted_value": float(row[6]),
                    "confidence": (
                        float(row[7])
                        if row[7] is not None
                        else None
                    ),
                    "risk_level": row[8],
                    "prediction_horizon_minutes": row[9],
                    "description": row[10],
                }
                for row in rows
            ]

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