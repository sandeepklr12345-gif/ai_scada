import os

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
# DISPLAY TABLE
# ============================================================

def display_table(cursor, table_name):

    print("\n" + "=" * 70)
    print(f"{table_name.upper()}")
    print("=" * 70)

    cursor.execute(
        f"SELECT * FROM {table_name} ORDER BY 1;"
    )

    rows = cursor.fetchall()

    column_names = [
        description[0]
        for description in cursor.description
    ]

    print("Columns:")
    print("  " + " | ".join(column_names))

    print("\nRows:")

    if not rows:
        print("  No rows found.")
        return

    for row in rows:
        print("  " + " | ".join(str(value) for value in row))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("POSTGRESQL REFERENCE DATA INSPECTION")
    print("=" * 70)

    connection = None

    try:

        connection = psycopg2.connect(**DB_CONFIG)

        cursor = connection.cursor()

        print("\nDatabase connection: PASS")

        tables = [
            "plants",
            "data_sources",
            "equipment",
            "parameters",
        ]

        for table in tables:
            display_table(cursor, table)

        cursor.close()

        print("\n" + "=" * 70)
        print("REFERENCE DATA INSPECTION: PASS")
        print("=" * 70)

    except Exception as exc:

        print("\nREFERENCE DATA INSPECTION: FAILED")
        print(f"Error: {exc}")

    finally:

        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
