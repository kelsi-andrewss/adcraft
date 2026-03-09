"""Database initialization for AdCraft."""

import sqlite3
from pathlib import Path


def ensure_data_dir(db_path: str | Path) -> Path:
    """Create the parent directory for the database file if it doesn't exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(db_path: str | Path = "data/ads.db") -> sqlite3.Connection:
    """Initialize the SQLite database with the AdCraft schema.

    Reads schema.sql relative to this file, executes it against a connection
    with WAL mode enabled. Idempotent — safe to call multiple times.

    Returns the open connection for immediate use.
    """
    if str(db_path) != ":memory:":
        ensure_data_dir(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    schema_path = Path(__file__).parent / "schema.sql"
    schema_sql = schema_path.read_text()
    conn.executescript(schema_sql)

    return conn
