from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
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
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", SMTP_USERNAME)
SMTP_USE_SSL = os.environ.get("SMTP_USE_SSL", "0").lower() in ("1", "true", "yes")
SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", f"http://127.0.0.1:{PORT}").rstrip("/")
