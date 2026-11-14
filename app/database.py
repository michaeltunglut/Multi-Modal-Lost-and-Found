import sqlite3
from datetime import datetime, timezone
from .config import DATA_DIR, DB_PATH, UPLOAD_DIR

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          email TEXT NOT NULL COLLATE NOCASE UNIQUE,
          password_hash TEXT NOT NULL,
          password_salt TEXT NOT NULL,
          role TEXT NOT NULL DEFAULT 'USER' CHECK(role IN ('USER','ADMIN')),
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
          token_hash TEXT PRIMARY KEY,
          user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          expires_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE TABLE IF NOT EXISTS reports (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          report_type TEXT NOT NULL CHECK(report_type IN ('LOST','FOUND')),
          title TEXT NOT NULL, description TEXT NOT NULL,
          category TEXT NOT NULL, colour TEXT NOT NULL DEFAULT '',
          brand TEXT NOT NULL DEFAULT '', contact_method TEXT NOT NULL DEFAULT '', location TEXT NOT NULL DEFAULT '',
          event_date TEXT NOT NULL DEFAULT '', image_path TEXT NOT NULL,
          image_features TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN'
            CHECK(status IN ('OPEN','UNDER_REVIEW','RESOLVED','CLOSED')),
          created_at TEXT NOT NULL,
          reporter_id INTEGER REFERENCES users(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reports_type_status ON reports(report_type,status);
        CREATE INDEX IF NOT EXISTS idx_reports_category ON reports(category);
        CREATE TABLE IF NOT EXISTS sightings (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lost_report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
          description TEXT NOT NULL,
          location TEXT NOT NULL,
          event_date TEXT NOT NULL DEFAULT '',
          contact_method TEXT NOT NULL,
          image_path TEXT NOT NULL UNIQUE,
          image_features TEXT NOT NULL,
          similarity_score REAL NOT NULL,
          image_score REAL,
          text_score REAL,
          status TEXT NOT NULL DEFAULT 'PENDING'
            CHECK(status IN ('PENDING','REVIEWED','REJECTED','RESOLVED')),
          created_at TEXT NOT NULL,
          reporter_id INTEGER REFERENCES users(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sightings_lost_report ON sightings(lost_report_id,status);
        CREATE TABLE IF NOT EXISTS email_notifications (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          event_key TEXT NOT NULL UNIQUE,
          event_type TEXT NOT NULL CHECK(event_type IN ('MATCH','SIGHTING')),
          recipient TEXT NOT NULL,
          subject TEXT NOT NULL,
          body TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING','SENT','FAILED')),
          last_error TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          sent_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS matches (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          lost_report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
          found_report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
          image_score REAL NOT NULL, text_score REAL NOT NULL,
          category_score REAL NOT NULL, final_score REAL NOT NULL,
          status TEXT NOT NULL DEFAULT 'SUGGESTED'
            CHECK(status IN ('SUGGESTED','REVIEWED','REJECTED','RESOLVED')),
          created_at TEXT NOT NULL,
          UNIQUE(lost_report_id,found_report_id),
          CHECK(lost_report_id != found_report_id)
        );
        """)
        sighting_columns = {row["name"] for row in c.execute("PRAGMA table_info(sightings)")}
        if "image_score" not in sighting_columns:
            c.execute("ALTER TABLE sightings ADD COLUMN image_score REAL")
        if "text_score" not in sighting_columns:
            c.execute("ALTER TABLE sightings ADD COLUMN text_score REAL")
        columns = {row["name"] for row in c.execute("PRAGMA table_info(reports)")}
        if "reporter_id" not in columns:
            c.execute("ALTER TABLE reports ADD COLUMN reporter_id INTEGER REFERENCES users(id) ON DELETE SET NULL")
        if "contact_method" not in columns:
            c.execute("ALTER TABLE reports ADD COLUMN contact_method TEXT NOT NULL DEFAULT ''")
