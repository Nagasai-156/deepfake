import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STATS_FILE = DATA_DIR / "user_stats.json"

# Ensure directories exist
for directory in [MODEL_DIR, DATA_DIR, UPLOAD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# App configuration
APP_CONFIG = {
    "model": {
        "json_path": MODEL_DIR / "dffnetv2B0.json",
        "weights_path": MODEL_DIR / "dffnetv2B0.h5",
        "input_size": (256, 256),
        "confidence_threshold": 0.5
    },
    "upload": {
        "allowed_types": ["jpg", "jpeg", "png", "webp"],
        "max_size_mb": 5
    },
    "game": {
        "achievements": {
            "streak_5": "5 correct in a row!",
            "streak_10": "10 correct in a row!",
            "accuracy_80": "80% accuracy reached!",
            "games_50": "50 games played!"
        }
    }
} 