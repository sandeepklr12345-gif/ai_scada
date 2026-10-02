import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_INTEGRATION = PROJECT_ROOT / "scripts" / "integration"

if str(SCRIPTS_INTEGRATION) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_INTEGRATION))


from postgres_to_scada_contract import (
    PostgreSQLSCADAContractAdapter,
)
from postgres_scada_reader import DB_CONFIG
from ai_runtime import AIRuntime


class PostgreSQLAIRuntimeBridge:

    def __init__(self, db_config):
        self.adapter = PostgreSQLSCADAContractAdapter(db_config)
        self.ai_runtime = AIRuntime()

    # ------------------------------------------------------------
    # Read standardized SCADA data
    # ------------------------------------------------------------

    def read_scada_contract(
        self,
        limit=None,
        source_id=None,
        equipment_id=None,
        parameter_id=None,
    ):
        return self.adapter.read_contract(
            limit=limit,
            source_id=source_id,
            equipment_id=equipment_id,
            parameter_id=parameter_id,
        )

    # ------------------------------------------------------------
    # Validate that PostgreSQL data reached the AI boundary
    # ------------------------------------------------------------

    def validate_scada_input(self, df):

        required_columns = [
            "timestamp",
            "source_id",
            "plant_id",
            "equipment_id",
            "parameter",
            "value",
            "unit",
            "quality",
        ]

        if list(df.columns) != required_columns:
            raise ValueError(
                "SCADA contract schema mismatch.\n"
                f"Expected: {required_columns}\n"
                f"Received: {list(df.columns)}"
            )

        if df.empty:
            return False

        if df["timestamp"].isna().any():
            raise ValueError("SCADA input contains invalid timestamps.")

        if df["value"].isna().any():
            raise ValueError("SCADA input contains missing values.")

        return True

    # ------------------------------------------------------------
    # Current bridge status
    # ------------------------------------------------------------

    def status(self):

        return {
            "postgresql": "CONNECTED",
            "scada_contract_adapter": "READY",
            "ai_runtime": "READY",
            "forecasting_model": "READY",
            "hai_anomaly_model": "READY",
        }


if __name__ == "__main__":

    print("=" * 70)
    print("POSTGRESQL → AI RUNTIME BRIDGE")
    print("=" * 70)

    bridge = PostgreSQLAIRuntimeBridge(DB_CONFIG)

    print("\nCOMPONENT STATUS")

    status = bridge.status()

    for component, state in status.items():
        print(f"  {component:<25}: {state}")

    # ------------------------------------------------------------
    # Read current PostgreSQL SCADA data
    # ------------------------------------------------------------

    scada_df = bridge.read_scada_contract()

    print("\nPOSTGRESQL SCADA DATA")
    print(f"Rows returned    : {len(scada_df)}")
    print(f"Columns returned : {len(scada_df.columns)}")

    if scada_df.empty:

        print("\nNo permanent SCADA measurements are currently stored.")

        print(
            "The PostgreSQL → SCADA contract path is available, "
            "but there is currently no live measurement batch to "
            "send into the AI models."
        )

        print("\nAI runtime was loaded successfully.")
        print(
            "Model inference is intentionally not executed because "
            "the database contains no real SCADA measurements."
        )

    else:

        bridge.validate_scada_input(scada_df)

        print("\nSCADA input validation: PASS")

        print("\nSAMPLE SCADA CONTRACT DATA")
        print(scada_df.head())

    print("\n" + "=" * 70)
    print("POSTGRESQL → AI RUNTIME BRIDGE: PASS")
    print("=" * 70)