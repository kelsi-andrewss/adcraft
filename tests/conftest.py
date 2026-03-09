"""Shared test fixtures for AdCraft tests."""


import pytest

from src.db.init_db import init_db


@pytest.fixture()
def db_conn():
    """Fresh in-memory SQLite database with AdCraft schema applied.

    Yields an open connection, then closes it after the test.
    """
    conn = init_db(":memory:")
    yield conn
    conn.close()
