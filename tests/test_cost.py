"""Tests for cost analytics: pricing calculation, quality snapshots."""

from __future__ import annotations

import pytest

from src.analytics.cost import (
    calculate_cost,
    compute_quality_snapshot,
    get_performance_per_token,
    record_api_cost,
)
from src.db.init_db import init_db
from src.db.queries import insert_ad, insert_evaluation


@pytest.fixture()
def db_conn():
    conn = init_db(":memory:")
    yield conn
    conn.close()


class TestCalculateCost:
    def test_gemini_flash_pricing(self) -> None:
        cost = calculate_cost("gemini-2.5-flash", 1000, 500)
        # $0.30/1M input + $2.50/1M output
        expected = (1000 * 0.30 / 1_000_000) + (500 * 2.50 / 1_000_000)
        assert abs(cost - expected) < 1e-10

    def test_gemini_pro_pricing(self) -> None:
        cost = calculate_cost("gemini-2.5-pro", 1000, 500)
        expected = (1000 * 1.25 / 1_000_000) + (500 * 10.00 / 1_000_000)
        assert abs(cost - expected) < 1e-10

    def test_claude_sonnet_pricing(self) -> None:
        cost = calculate_cost("claude-sonnet-4-6", 1000, 500)
        expected = (1000 * 3.00 / 1_000_000) + (500 * 15.00 / 1_000_000)
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_returns_zero(self) -> None:
        cost = calculate_cost("unknown-model-xyz", 1000, 500)
        assert cost == 0.0

    def test_zero_tokens(self) -> None:
        cost = calculate_cost("gemini-2.5-flash", 0, 0)
        assert cost == 0.0


class TestRecordApiCost:
    def test_updates_ad_row(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test ad",
            headline="Test",
            description="Desc",
            cta_button="Learn More",
        )

        cost = record_api_cost(db_conn, "ads", ad_id, "gemini-2.5-flash", 500, 200)
        assert cost > 0

        row = db_conn.execute(
            "SELECT input_tokens, output_tokens, cost_usd FROM ads WHERE id = ?", (ad_id,)
        ).fetchone()
        assert row[0] == 500
        assert row[1] == 200
        assert row[2] == cost


class TestComputeQualitySnapshot:
    def test_empty_db_returns_zero_snapshot(self, db_conn) -> None:
        snapshot = compute_quality_snapshot(db_conn, cycle_number=1)
        assert snapshot["total_ads"] == 0
        assert snapshot["quality_per_dollar"] is None

    def test_snapshot_with_data(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="Learn More",
            cost_usd=0.001,
        )
        for dim, score in [
            ("clarity", 8.0),
            ("value_prop", 7.0),
            ("cta_effectiveness", 7.5),
            ("brand_voice", 8.0),
            ("emotional_resonance", 7.0),
        ]:
            insert_evaluation(
                db_conn,
                ad_id=ad_id,
                dimension=dim,
                score=score,
                eval_mode="final",
            )

        snapshot = compute_quality_snapshot(db_conn, cycle_number=1)
        assert snapshot["total_ads"] == 1
        assert snapshot["avg_weighted_score"] is not None
        assert snapshot["avg_weighted_score"] > 0
        assert snapshot["ads_above_threshold"] >= 0
        assert snapshot["token_spend_usd"] >= 0

    def test_zero_spend_quality_per_dollar_is_none(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="Learn More",
            cost_usd=0.0,
        )
        insert_evaluation(db_conn, ad_id=ad_id, dimension="clarity", score=8.0)

        snapshot = compute_quality_snapshot(db_conn, cycle_number=1)
        # With zero spend, quality_per_dollar should be None (no div by zero)
        assert snapshot["quality_per_dollar"] is None


class TestGetPerformancePerToken:
    def test_returns_snapshots(self, db_conn) -> None:
        from src.db.queries import insert_quality_snapshot

        insert_quality_snapshot(
            db_conn,
            cycle_number=1,
            avg_weighted_score=7.5,
            quality_per_dollar=100.0,
            token_spend_usd=0.075,
        )
        results = get_performance_per_token(db_conn)
        assert len(results) >= 1
