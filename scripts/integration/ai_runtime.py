from pathlib import Path
import sys
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts" / "integration")
)

from ai_model_integration import AIModelIntegration


# ============================================================
# UNIFIED AI RUNTIME
# ============================================================

class AIRuntime:

    def __init__(self):
        """
        Load the validated AI model integration layer.
        """

        self.integration = AIModelIntegration()

    # ========================================================
    # FORECAST
    # ========================================================

    def forecast(self, data):
        """
        Generate 600 MW power-output forecasts.

        Returns:
            dict with:
                power_2min
                power_10min
                power_30min
        """

        result = self.integration.predict_forecast(
            data
        )

        return {
            "power_2min": float(
                result["power_2min"].iloc[0]
            ),
            "power_10min": float(
                result["power_10min"].iloc[0]
            ),
            "power_30min": float(
                result["power_30min"].iloc[0]
            ),
        }

    # ========================================================
    # ANOMALY
    # ========================================================

    def detect_anomaly(self, data):
        """
        Generate HAI Candidate C anomaly result.
        """

        result = self.integration.predict_anomaly(
            data
        )

        row = result.iloc[0]

        return {
            "anomaly_score": (
                None
                if pd.isna(row["anomaly_score"])
                else float(row["anomaly_score"])
            ),
            "prediction": (
                None
                if pd.isna(row["prediction"])
                else int(row["prediction"])
            ),
            "status": str(row["status"]),
        }

    # ========================================================
    # UNIFIED PREDICTION
    # ========================================================

    def predict(self, forecast_data, hai_data):
        """
        Generate a unified AI result.

        forecast_data:
            600 MW model input.

        hai_data:
            58 original HAI SCADA features.
        """

        forecast = self.forecast(
            forecast_data
        )

        anomaly = self.detect_anomaly(
            hai_data
        )

        return {
            "forecast": forecast,
            "anomaly": anomaly,
        }

    # ========================================================
    # SINGLE ROW INTERFACE
    # ========================================================

    def predict_single(
        self,
        forecast_data,
        hai_data
    ):
        """
        Unified single-row prediction interface.
        """

        return self.predict(
            forecast_data,
            hai_data
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("UNIFIED AI RUNTIME")
    print("=" * 80)

    runtime = AIRuntime()

    print()
    print("AI runtime loaded successfully.")

    print()
    print("Available interfaces:")
    print("  runtime.forecast()")
    print("  runtime.detect_anomaly()")
    print("  runtime.predict()")
    print("  runtime.predict_single()")

    print()
    print("=" * 80)
    print("UNIFIED AI RUNTIME: READY")
    print("=" * 80)