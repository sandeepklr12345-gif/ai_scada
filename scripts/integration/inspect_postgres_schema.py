import os

import psycopg2
from psycopg2 import sql


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
# CONNECT
# ============================================================

def connect_database():

    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# LIST TABLES
# ============================================================

def get_tables(cursor):

    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """

    cursor.execute(query)

    return [row[0] for row in cursor.fetchall()]


# ============================================================
# GET TABLE COLUMNS
# ============================================================

def get_columns(cursor, table_name):

    query = """
        SELECT
            column_name,
            data_type,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position;
    """

    cursor.execute(query, (table_name,))

    return cursor.fetchall()


# ============================================================
# GET ROW COUNT
# ============================================================

def get_row_count(cursor, table_name):

    query = sql.SQL(
        "SELECT COUNT(*) FROM {}"
    ).format(sql.Identifier(table_name))

    cursor.execute(query)

    return cursor.fetchone()[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("POSTGRESQL SCHEMA INSPECTION")
    print("=" * 70)

    connection = None

    try:

        connection = connect_database()

        cursor = connection.cursor()

        print("\nDatabase connection: PASS")
        print(f"Database: {DB_CONFIG['database']}")
        print(f"Host: {DB_CONFIG['host']}")
        print(f"Port: {DB_CONFIG['port']}")

        # ----------------------------------------------------
        # Tables
        # ----------------------------------------------------

        tables = get_tables(cursor)

        print("\nPUBLIC TABLES")
        print("-" * 70)

        if not tables:
            print("No public tables found.")
            return

        for table in tables:

            print(f"\nTABLE: {table}")

            columns = get_columns(cursor, table)
            row_count = get_row_count(cursor, table)

            print(f"Rows: {row_count}")

            print("Columns:")

            for column_name, data_type, nullable in columns:

                print(
                    f"  {column_name:<35}"
                    f"{data_type:<20}"
                    f"nullable={nullable}"
                )

        cursor.close()

        print("\n" + "=" * 70)
        print("POSTGRESQL SCHEMA INSPECTION: PASS")
        print("=" * 70)

    except Exception as exc:

        print("\nPOSTGRESQL SCHEMA INSPECTION: FAILED")
        print(f"Error: {exc}")

    finally:

        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
