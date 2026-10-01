from pathlib import Path

# Project root:
# config/project_paths.py
# -> parents[1] = Ai_Scada
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "config"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
INTEGRATION_DATA_DIR = DATA_DIR / "integration"

FORECASTING_MODELS_DIR = MODELS_DIR / "forecasting"