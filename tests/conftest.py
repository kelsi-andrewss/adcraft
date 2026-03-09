"""Shared test fixtures for AdCraft tests."""

import os

import pytest

from src.db.init_db import init_db


@pytest.fixture(autouse=True)
def _test_database_path(tmp_path):
    """Ensure log_decision's fallback DB path points to a valid schema'd database.

    This covers code paths that call log_decision() without an explicit conn
    (e.g., extract_patterns, calculate_cost for unknown models).
    """
    db_path = str(tmp_path / "test_decisions.db")
    conn = init_db(db_path)
    conn.close()
    old = os.environ.get("DATABASE_PATH")
    os.environ["DATABASE_PATH"] = db_path
    yield
    if old is None:
        os.environ.pop("DATABASE_PATH", None)
    else:
        os.environ["DATABASE_PATH"] = old


@pytest.fixture()
def db_conn():
    """Fresh in-memory SQLite database with AdCraft schema applied.

    Yields an open connection, then closes it after the test.
    """
    conn = init_db(":memory:")
    yield conn
    conn.close()
