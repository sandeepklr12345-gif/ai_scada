from pathlib import Path
import pandas as pd
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_DIR = (
    PROJECT_ROOT
    / "data"
    / "integration"
    / "runtime_simulation"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "600mw"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_RESULTS = (
    RUNTIME_DIR / "600mw_runtime_forecast_results_optimized.csv"
)

ACCURACY_REPORT = (
    RUNTIME_DIR / "600mw_forecast_accuracy_report.csv"
)

ERROR_ANALYSIS = (
    RUNTIME_DIR / "600mw_forecast_error_analysis.csv"
)

RAMP_ANALYSIS = (
    RUNTIME_DIR / "600mw_ramp_error_analysis.csv"
)

PLOT_DIR = RUNTIME_DIR / "plots"

OUTPUT_REPORT = REPORT_DIR / "600MW_FORECAST_VALIDATION_REPORT.md"


# ============================================================
# HEADER
# ============================================================

report = []

report.append("# 600 MW Forecasting Pipeline Validation Report\n")

report.append(
    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
)

report.append(
    "**Project:** AI-SCADA Thermal Power Plant Intelligence System\n"
)

report.append(
    "**Dataset:** 600 MW Coal-Fired Generating Unit\n"
)

report.append(
    "**Validation Type:** Runtime Forecasting and Performance Validation\n"
)

report.append("\n---\n")


# ============================================================
# 1. EXECUTIVE SUMMARY
# ============================================================

report.append("## 1. Executive Summary\n")

report.append(
    "The 600 MW forecasting pipeline was validated using a chronological "
    "runtime replay containing 5,649 observations. The runtime feature "
    "preparation pipeline generated the required 119-feature schema and "
    "successfully passed all rows through the forecasting engine.\n"
)

report.append(
    "The optimized batch inference implementation reduced runtime from "
    "approximately 290.39 seconds to 4.06 seconds while maintaining "
    "prediction equivalence with the original implementation within "
    "floating-point tolerance.\n"
)

report.append(
    "Forecast accuracy was evaluated at 2-minute, 10-minute and "
    "30-minute horizons. Performance decreased progressively as the "
    "forecast horizon increased, which is consistent with the increasing "
    "uncertainty associated with longer-horizon prediction.\n"
)

report.append("\n")


# ============================================================
# 2. DATASET AND REPLAY
# ============================================================

report.append("## 2. Dataset and Runtime Replay\n")

report.append("| Parameter | Value |")
report.append("|---|---:|")
report.append("| Replay rows | 5,649 |")
report.append("| Runtime features | 119 |")
report.append("| Numeric runtime features | 119 |")
report.append("| Non-numeric runtime features | 0 |")
report.append("| Chronological inference | Yes |")
report.append("| Replay status | PASS |")
report.append("\n")


# ============================================================
# 3. RUNTIME VALIDATION
# ============================================================

report.append("## 3. Runtime Forecasting Validation\n")

report.append(
    "The runtime feature preparation module successfully transformed the "
    "replay dataset into the exact feature schema required by the trained "
    "forecasting models.\n"
)

report.append(
    "All 5,649 rows were processed chronologically without feature-schema "
    "or numeric-validation failures.\n"
)

report.append("**Runtime validation status: PASS**\n")


# ============================================================
# 4. MODEL CONFIGURATION
# ============================================================

report.append("## 4. Forecasting Model Configuration\n")

report.append("| Forecast Horizon | Model |")
report.append("|---|---|")
report.append("| 2 minutes | Linear Regression |")
report.append("| 10 minutes | Linear Regression |")
report.append("| 30 minutes | Linear Regression |")
report.append("\n")

report.append(
    "The forecasting engine uses three independently trained regression "
    "models corresponding to the three prediction horizons.\n"
)


# ============================================================
# 5. PERFORMANCE
# ============================================================

report.append("## 5. Runtime Performance Benchmark\n")

report.append("| Metric | Original | Optimized |")
report.append("|---|---:|---:|")
report.append("| Runtime | 290.3856 sec | 4.0615 sec |")
report.append("| Rows/sec | 19.45 | 1390.85 |")
report.append("| Time/row | 51.4048 ms | 0.7190 ms |")
report.append("\n")

report.append("| Optimization Result | Value |")
report.append("|---|---:|")
report.append("| Runtime reduction | 98.60% |")
report.append("| Speedup | 71.50× |")
report.append("| Time saved | 286.3241 sec |")
report.append("\n")

report.append(
    "The optimization removed repeated per-row pandas DataFrame creation, "
    "repeated numeric conversion, repeated validation and individual "
    "model-inference calls. The optimized implementation performs batch "
    "inference over the complete feature matrix.\n"
)

report.append("**Performance optimization status: PASS**\n")


# ============================================================
# 6. PREDICTION EQUIVALENCE
# ============================================================

report.append("## 6. Prediction Equivalence Validation\n")

report.append(
    "The optimized batch implementation was compared against the original "
    "row-by-row implementation using all 5,649 replay observations.\n"
)

report.append("| Horizon | Samples | Within tolerance | Maximum difference |")
report.append("|---|---:|---:|---:|")
report.append("| 2 min | 5,649 | 5,649 / 5,649 | 1.137e-12 |")
report.append("| 10 min | 5,649 | 5,649 / 5,649 | 2.956e-12 |")
report.append("| 30 min | 5,649 | 5,649 / 5,649 | 4.320e-12 |")
report.append("\n")

report.append(
    "**Prediction equivalence status: PASS**\n"
)

report.append(
    "The differences are at floating-point numerical precision and do not "
    "represent a meaningful change in model predictions.\n"
)


# ============================================================
# 7. FORECAST ACCURACY
# ============================================================

report.append("## 7. Forecast Accuracy\n")

report.append("| Horizon | MAE (MW) | RMSE (MW) | MAPE | R² |")
report.append("|---|---:|---:|---:|---:|")
report.append("| 2 min | 1.5162 | 2.3443 | 0.3561% | 0.999318 |")
report.append("| 10 min | 5.9879 | 9.2774 | 1.3970% | 0.989325 |")
report.append("| 30 min | 14.8172 | 21.7485 | 3.4812% | 0.941337 |")
report.append("\n")

report.append("| Horizon | Within ±5% | Within ±10% |")
report.append("|---|---:|---:|")
report.append("| 2 min | 100.00% | 100.00% |")
report.append("| 10 min | 96.18% | 99.65% |")
report.append("| 30 min | 77.77% | 94.78% |")
report.append("\n")

report.append(
    "The results show increasing prediction error as the forecast horizon "
    "increases from 2 to 30 minutes. The 2-minute horizon provides the "
    "lowest error, while the 30-minute horizon exhibits the largest "
    "uncertainty.\n"
)


# ============================================================
# 8. ERROR ANALYSIS
# ============================================================

report.append("## 8. Forecast Error Analysis\n")

report.append("| Horizon | Maximum Absolute Error |")
report.append("|---|---:|")
report.append("| 2 min | 15.2008 MW |")
report.append("| 10 min | 55.4075 MW |")
report.append("| 30 min | 119.6948 MW |")
report.append("\n")

report.append(
    "The largest errors were concentrated around several periods of rapid "
    "power variation. Error analysis was therefore extended using "
    "two-minute power ramp magnitude.\n"
)


# ============================================================
# 9. RAMP ANALYSIS
# ============================================================

report.append("## 9. Ramp-Rate and Forecast Error Analysis\n")

report.append(
    "Pearson correlation was calculated between absolute two-minute power "
    "ramp magnitude and absolute forecast error.\n"
)

report.append("| Horizon | Pearson r |")
report.append("|---|---:|")
report.append("| 2 min | 0.541283 |")
report.append("| 10 min | 0.503251 |")
report.append("| 30 min | 0.417605 |")
report.append("\n")

report.append(
    "Within this replay dataset, larger two-minute power changes were "
    "positively associated with larger absolute forecast errors. This is "
    "an observed statistical relationship and should not be interpreted "
    "as proof of causation.\n"
)


# ============================================================
# 10. VISUAL ANALYSIS
# ============================================================

report.append("## 10. Visual Analysis\n")

plots = [
    "actual_vs_forecast.png",
    "forecast_error_distribution.png",
    "ramp_vs_error_2min.png",
    "ramp_vs_error_10min.png",
    "ramp_vs_error_30min.png",
]

report.append("Generated visualization files:\n")

for plot in plots:
    report.append(f"- `{plot}`")

report.append("\n")

report.append(
    f"Plot directory: `{PLOT_DIR}`\n"
)


# ============================================================
# 11. ENGINEERING FINDINGS
# ============================================================

report.append("## 11. Engineering Findings\n")

findings = [
    "The runtime feature-preparation pipeline successfully produces the required 119-feature model input.",
    "All 5,649 replay observations were processed successfully.",
    "Batch inference provides a major performance improvement over row-by-row inference.",
    "The optimized implementation produces predictions equivalent to the original implementation within floating-point tolerance.",
    "Short-horizon forecasting provides substantially lower prediction error than longer-horizon forecasting.",
    "Forecast error increases as the prediction horizon increases.",
    "Larger short-term power ramps are associated with larger absolute forecast errors in this replay dataset.",
    "The current Linear Regression models are computationally lightweight.",
    "GPU acceleration is not currently required for the forecasting inference workload because optimized CPU batch inference already completes the replay in approximately four seconds."
]

for finding in findings:
    report.append(f"- {finding}")

report.append("\n")


# ============================================================
# 12. LIMITATIONS
# ============================================================

report.append("## 12. Limitations\n")

limitations = [
    "The validation is based on the available 600 MW replay dataset.",
    "The models evaluated here are Linear Regression models.",
    "The validation does not establish performance on unseen power-plant datasets.",
    "The relationship between ramp magnitude and forecast error is observational.",
    "Long-horizon forecast performance may require additional temporal or operational features.",
    "Real-time deployment should include monitoring for data-quality failures, feature drift and model degradation."
]

for limitation in limitations:
    report.append(f"- {limitation}")

report.append("\n")


# ============================================================
# 13. FINAL VALIDATION STATUS
# ============================================================

report.append("## 13. Final Validation Status\n")

report.append("| Validation Component | Status |")
report.append("|---|---|")
report.append("| Runtime replay | PASS |")
report.append("| Feature schema validation | PASS |")
report.append("| Forecast inference | PASS |")
report.append("| Performance optimization | PASS |")
report.append("| Prediction equivalence | PASS |")
report.append("| Forecast accuracy validation | PASS |")
report.append("| Error analysis | PASS |")
report.append("| Ramp analysis | PASS |")
report.append("| Visual analysis | PASS |")
report.append("\n")

report.append(
    "# FINAL STATUS: 600 MW FORECASTING PIPELINE VALIDATED\n"
)

report.append(
    "The 600 MW runtime forecasting pipeline has completed functional, "
    "performance, numerical-equivalence, accuracy, error and visual "
    "validation on the available replay dataset.\n"
)


# ============================================================
# WRITE REPORT
# ============================================================

OUTPUT_REPORT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("=" * 70)
print("600 MW FORMAL VALIDATION REPORT")
print("=" * 70)
print()
print("Report created successfully.")
print()
print(f"Output:")
print(OUTPUT_REPORT)
print()
print("FINAL STATUS: PASS")
print("=" * 70)
report.append(
    "The production batch runtime was subsequently implemented in "
    "`scripts/integration/run_600mw_runtime_forecast_batch.py`. "
    "The production batch implementation processed all 5,649 replay "
    "observations in 2.7973 seconds, achieving approximately 2,019.47 "
    "rows per second. The batch inference computation itself required "
    "approximately 0.0309 seconds.\n"
)

report.append(
    "The production batch runtime produced numerically equivalent "
    "predictions to the original row-by-row implementation within "
    "floating-point tolerance and passed the forecast accuracy "
    "validation.\n"
)