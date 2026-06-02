# config
from pathlib import Path

TRAFFIC_PROJECT_DIR = Path(__file__).resolve().parents[1]
SYSTEM_DIR = TRAFFIC_PROJECT_DIR / "Traffic_detection_system"
DATASET_DIR = TRAFFIC_PROJECT_DIR / "Traffic_detection_dataset"
DATABASE_CONFIG = {
    "Driver": "{ODBC Driver 17 for SQL Server}",
    "Server": "JUANLEGIONR7000,1433",  # 显式指定TCP
    "Database": "Traffic_system_database",
    "Trusted_Connection": "yes"
}

DEFAULT_WEIGHTS_DIR = str(SYSTEM_DIR / "best_weights")
DEFAULT_MEDIA_DIR = str(DATASET_DIR / "images_videos_test")
DEFAULT_RESULTS_DIR = str(SYSTEM_DIR / "Target_detection_results")
DEFAULT_VIOLATION_SCREENSHOT_DIR = str(SYSTEM_DIR / "Traffic_violation_screenshot")
