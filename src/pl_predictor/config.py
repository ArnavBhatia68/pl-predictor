import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
STATE_DIR = PROJECT_ROOT / "state"

MATCHES_PATH = PROCESSED_DIR / "matches.csv"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"
V2_MODEL_PATH = MODEL_DIR / "v2_model.joblib"
V2_METRICS_PATH = REPORT_DIR / "v2_metrics.json"
V2_PREDICTIONS_PATH = REPORT_DIR / "v2_test_predictions.csv"
V2_WALK_FORWARD_PATH = REPORT_DIR / "v2_walk_forward.csv"
V2_IMPORTANCE_PATH = REPORT_DIR / "v2_feature_importance.csv"
V3_MODEL_PATH = MODEL_DIR / "v3_poisson_model.joblib"
V3_METRICS_PATH = REPORT_DIR / "v3_metrics.json"
V3_PREDICTIONS_PATH = REPORT_DIR / "v3_test_predictions.csv"
V3_WALK_FORWARD_PATH = REPORT_DIR / "v3_walk_forward.csv"
V3_IMPORTANCE_PATH = REPORT_DIR / "v3_feature_importance.csv"
V4_MODEL_PATH = MODEL_DIR / "v4_ensemble_model.joblib"
V4_METRICS_PATH = REPORT_DIR / "v4_metrics.json"
V4_PREDICTIONS_PATH = REPORT_DIR / "v4_test_predictions.csv"
V4_WALK_FORWARD_PATH = REPORT_DIR / "v4_walk_forward.csv"
LIVE_DB_PATH = Path(
    os.getenv("PL_PREDICTOR_DB_PATH", str(STATE_DIR / "pl_predictor.sqlite3"))
)


def ensure_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)
