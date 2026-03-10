"""Database initialization for AdCraft."""

import sqlite3
from pathlib import Path


def ensure_data_dir(db_path: str | Path) -> Path:
    """Create the parent directory for the database file if it doesn't exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_migrations(conn: sqlite3.Connection) -> None:
    """Run idempotent schema migrations for columns added after initial schema."""
    _migrate_image_columns(conn)


def _migrate_image_columns(conn: sqlite3.Connection) -> None:
    """Add image-related columns to ads table (story-596)."""
    columns = [
        ("image_path", "TEXT"),
        ("visual_prompt", "TEXT"),
        ("image_model", "TEXT"),
        ("image_cost_usd", "REAL"),
        ("variant_group_id", "TEXT"),
        ("variant_type", "TEXT"),
    ]
    for col_name, col_type in columns:
        try:
            conn.execute(f"ALTER TABLE ads ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()


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

    run_migrations(conn)

    return conn
