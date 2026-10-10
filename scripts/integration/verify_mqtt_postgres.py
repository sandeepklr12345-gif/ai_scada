import os
import psycopg2

conn = psycopg2.connect(
    host=os.getenv("AI_SCADA_DB_HOST", "localhost"),
    port=int(os.getenv("AI_SCADA_DB_PORT", "5432")),
    dbname=os.getenv("AI_SCADA_DB_NAME", "ai_scada"),
    user=os.getenv("AI_SCADA_DB_USER", "postgres"),
    password=os.getenv("AI_SCADA_DB_PASSWORD"),
)

cur = conn.cursor()

cur.execute("""
    SELECT COUNT(*)
    FROM measurements
    WHERE measurement_id >= 40
""")
print("New measurements:", cur.fetchone()[0])

cur.execute("""
    SELECT COUNT(*)
    FROM ai_predictions
    WHERE prediction_id >= 107
""")
print("New predictions:", cur.fetchone()[0])

print("\nLatest measurements:")
cur.execute("""
    SELECT measurement_id, timestamp, value
    FROM measurements
    ORDER BY measurement_id DESC
    LIMIT 10
""")

for row in cur.fetchall():
    print(row)

print("\nLatest predictions:")
cur.execute("""
    SELECT
        prediction_id,
        timestamp,
        prediction_type,
        predicted_value,
        prediction_horizon_minutes
    FROM ai_predictions
    ORDER BY prediction_id DESC
    LIMIT 30
""")

for row in cur.fetchall():
    print(row)

cur.close()
conn.close()
