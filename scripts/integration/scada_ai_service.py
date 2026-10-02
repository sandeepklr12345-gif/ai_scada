from pathlib import Path
import sys
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
INTEGRATION_DIR = SCRIPTS_DIR / "integration"

if str(INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_DIR))

from ai_runtime import AIRuntime


# ============================================================
# SCADA AI SERVICE
# ============================================================

class SCADAAIService:

    def __init__(self):
        self.runtime = AIRuntime()

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    def process_forecast(self, scada_data):
        """
        Process one SCADA-compatible row for power forecasting.
        """

        result = self.runtime.forecast(scada_data)

        return {
            "forecast": result
        }

    # --------------------------------------------------------
    # Anomaly Detection
    # --------------------------------------------------------

    def process_anomaly(self, scada_data):
        """
        Process HAI/SCADA data for anomaly detection.

        This uses the existing batch interface.
        Stateful streaming will be handled separately.
        """

        result = self.runtime.detect_anomaly(scada_data)

        return {
            "anomaly": result
        }

    # --------------------------------------------------------
    # Unified Processing
    # --------------------------------------------------------

    def process(self, forecast_data, anomaly_data):

        forecast = self.runtime.forecast(forecast_data)

        anomaly = self.runtime.detect_anomaly(anomaly_data)

        return {
            "forecast": forecast,
            "anomaly": anomaly
        }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SCADA → AI SERVICE")
    print("=" * 70)

    service = SCADAAIService()

    print("\nAI runtime loaded successfully.")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    forecast_path = (
        PROJECT_ROOT
        / "data"
        / "features"
        / "600mw"
        / "candidates"
        / "feature_set_D_full_candidate_pool.csv"
    )

    hai_path = (
        PROJECT_ROOT
        / "data"
        / "features"
        / "hai"
        / "hai-23.05"
        / "model_ready"
        / "hai-test1_model_ready.csv"
    )

    forecast_df = pd.read_csv(forecast_path)
    hai_df = pd.read_csv(hai_path)

    print(f"600 MW rows loaded : {len(forecast_df)}")
    print(f"HAI rows loaded    : {len(hai_df)}")

    # --------------------------------------------------------
    # Forecast test
    # --------------------------------------------------------

    forecast_row = forecast_df.iloc[[100]].copy()

    forecast_result = service.process_forecast(
        forecast_row
    )

    print("\nFORECAST RESULT")
    print(forecast_result)

    # --------------------------------------------------------
    # Anomaly test
    # --------------------------------------------------------

    hai_features = service.runtime.integration.hai_detector.original_features

    anomaly_rows = hai_df[hai_features].iloc[:4].copy()

    anomaly_result = service.process_anomaly(
        anomaly_rows
    )

    print("\nANOMALY RESULT")
    print(anomaly_result)

    # --------------------------------------------------------
    # Unified interface test
    # --------------------------------------------------------

    unified_result = service.process(
        forecast_row,
        anomaly_rows
    )

    print("\nUNIFIED SCADA → AI RESULT")
    print(unified_result)

    print("\n" + "=" * 70)
    print("SCADA → AI SERVICE TEST COMPLETE")
    print("=" * 70)