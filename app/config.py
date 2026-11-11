from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LOST_FOUND_DATA", PROJECT_ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "lost_found.sqlite3"
WEB_DIR = PROJECT_ROOT / "web"
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MAX_UPLOAD = 10 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
CATEGORIES = ["Bag", "Wallet", "Bottle", "Electronics", "Book", "Keys", "Clothing", "ID card", "Other"]
MATCH_MIN_SCORE = 0.60
