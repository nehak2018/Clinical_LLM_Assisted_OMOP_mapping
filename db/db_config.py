import sqlite3
import json
from pathlib import Path
import os

#DB_PATH = Path("LLM_assisted_OMOP.db")
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "LLM_assisted_OMOP.db"

print("DB absolute path:", os.path.abspath(DB_PATH))

def get_connection():
    """Return SQLite connection."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create required tables if they do not exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS note_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        note_id INTEGER,
        model TEXT,
        note_text TEXT,
        raw_preds TEXT,
        icd_validation TEXT,
        grounded_results TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# Run once at startup
#init_db()