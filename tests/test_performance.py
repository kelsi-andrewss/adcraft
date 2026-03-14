"""Tests for performance_feedback table and PerformanceFeedback model."""

import sqlite3
from datetime import date

import pytest
from pydantic import ValidationError

from src.db.queries import (
    get_performance_feedback_for_ad,
    insert_ad,
    insert_performance_feedback,
)
from src.models.performance import PerformanceFeedback


def _seed_ad(conn):
    """Insert a minimal ad and return its ID."""
    return insert_ad(
        conn,
        primary_text="Learn math",
        headline="Tutoring",
        description="Best tutors",
        cta_button="Sign Up",
    )


def test_performance_feedback_table_exists(db_conn):
    rows = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='performance_feedback'"
    ).fetchall()
    assert len(rows) == 1


def test_insert_and_retrieve_performance_feedback(db_conn):
    ad_id = _seed_ad(db_conn)
    fb_id = insert_performance_feedback(
        db_conn,
        ad_id=ad_id,
        platform="facebook",
        impressions=1000,
        clicks=50,
        conversions=5,
        spend_usd=25.0,
        date_start=date(2026, 3, 1),
        date_end=date(2026, 3, 7),
    )
    assert fb_id

    rows = get_performance_feedback_for_ad(db_conn, ad_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == fb_id
    assert row["ad_id"] == ad_id
    assert row["platform"] == "facebook"
    assert row["impressions"] == 1000
    assert row["clicks"] == 50
    assert row["conversions"] == 5
    assert row["spend_usd"] == 25.0
    assert row["date_start"] == "2026-03-01"
    assert row["date_end"] == "2026-03-07"


def test_multiple_feedback_rows_ordered_by_date(db_conn):
    ad_id = _seed_ad(db_conn)

    insert_performance_feedback(
        db_conn,
        ad_id=ad_id,
        platform="facebook",
        impressions=500,
        clicks=20,
        date_start=date(2026, 3, 8),
        date_end=date(2026, 3, 14),
    )
    insert_performance_feedback(
        db_conn,
        ad_id=ad_id,
        platform="facebook",
        impressions=1000,
        clicks=50,
        date_start=date(2026, 3, 1),
        date_end=date(2026, 3, 7),
    )

    rows = get_performance_feedback_for_ad(db_conn, ad_id)
    assert len(rows) == 2
    assert rows[0]["date_start"] == "2026-03-01"
    assert rows[1]["date_start"] == "2026-03-08"


def test_foreign_key_constraint(db_conn):
    with pytest.raises(sqlite3.IntegrityError):
        insert_performance_feedback(
            db_conn,
            ad_id="nonexistent-ad-id",
            platform="facebook",
            date_start=date(2026, 3, 1),
            date_end=date(2026, 3, 7),
        )


def test_ctr_and_conversion_rate_properties():
    fb = PerformanceFeedback(
        ad_id="a1",
        platform="facebook",
        impressions=1000,
        clicks=50,
        conversions=5,
        date_start=date(2026, 3, 1),
        date_end=date(2026, 3, 7),
    )
    assert fb.ctr == pytest.approx(0.05)
    assert fb.conversion_rate == pytest.approx(0.1)

    # Zero impressions -> ctr 0
    fb_zero_imp = PerformanceFeedback(
        ad_id="a2",
        platform="instagram",
        impressions=0,
        clicks=0,
        date_start=date(2026, 3, 1),
        date_end=date(2026, 3, 7),
    )
    assert fb_zero_imp.ctr == 0.0
    assert fb_zero_imp.conversion_rate == 0.0

    # Impressions > 0 but zero clicks -> conversion_rate 0
    fb_zero_clicks = PerformanceFeedback(
        ad_id="a3",
        platform="google",
        impressions=500,
        clicks=0,
        date_start=date(2026, 3, 1),
        date_end=date(2026, 3, 7),
    )
    assert fb_zero_clicks.ctr == 0.0
    assert fb_zero_clicks.conversion_rate == 0.0


def test_performance_feedback_model_validation():
    with pytest.raises(ValidationError):
        PerformanceFeedback(
            ad_id="a1",
            platform="facebook",
            date_start="not-a-date",
            date_end=date(2026, 3, 7),
        )

    fb = PerformanceFeedback.model_validate(
        {
            "ad_id": "a1",
            "platform": "facebook",
            "impressions": 100,
            "clicks": 10,
            "conversions": 1,
            "spend_usd": 5.50,
            "date_start": "2026-03-01",
            "date_end": "2026-03-07",
        }
    )
    data = fb.model_dump()
    assert data["ad_id"] == "a1"
    assert data["date_start"] == date(2026, 3, 1)
    assert data["spend_usd"] == 5.50
