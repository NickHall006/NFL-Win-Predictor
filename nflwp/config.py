from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
MODEL_DIR = ROOT / "models"

FIRST_SEASON = 2011   # first season we download (used to warm up rolling stats)
TRAIN_START = 2012    # first season used for training rows
DEFAULT_MODEL = "ensemble"

# nflverse already uses current abbreviations for most seasons, but be safe.
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
