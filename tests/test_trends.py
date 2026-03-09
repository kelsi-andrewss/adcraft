"""Tests for trend chart generation: empty-data handling and figure structure."""

from __future__ import annotations

import pytest

from src.analytics.trends import (
    before_after_comparison,
    convergence_curves,
    cost_efficiency_trend,
    dimension_breakdown,
    score_distribution,
)
from src.db.init_db import init_db
from src.db.queries import insert_ad, insert_evaluation, insert_quality_snapshot


@pytest.fixture()
def db_conn():
    conn = init_db(":memory:")
    yield conn
    conn.close()


class TestEmptyDataHandling:
    """All chart functions should return valid figures with 'No data available' on empty DB."""

    def test_score_distribution_empty(self, db_conn) -> None:
        fig = score_distribution(db_conn)
        assert fig.layout.title.text == "Score Distribution"
        assert any("No data" in a.text for a in fig.layout.annotations)

    def test_convergence_curves_empty(self, db_conn) -> None:
        fig = convergence_curves(db_conn)
        assert fig.layout.title.text == "Convergence Curves"
        assert any("No data" in a.text for a in fig.layout.annotations)

    def test_dimension_breakdown_empty(self, db_conn) -> None:
        fig = dimension_breakdown(db_conn)
        assert fig.layout.title.text == "Dimension Breakdown"
        assert any("No data" in a.text for a in fig.layout.annotations)

    def test_before_after_empty(self, db_conn) -> None:
        fig = before_after_comparison(db_conn, "nonexistent-id")
        assert "Before/After" in fig.layout.title.text
        assert any("No data" in a.text for a in fig.layout.annotations)

    def test_cost_efficiency_empty(self, db_conn) -> None:
        fig = cost_efficiency_trend(db_conn)
        assert fig.layout.title.text == "Cost Efficiency Trend"
        assert any("No data" in a.text for a in fig.layout.annotations)


class TestWithData:
    """Chart functions should produce valid figures when data exists."""

    @pytest.fixture()
    def seeded_db(self, db_conn):
        """Insert sample ads and evaluations."""
        for i in range(3):
            ad_id = insert_ad(
                db_conn,
                primary_text=f"Ad text {i}",
                headline=f"Headline {i}",
                description=f"Desc {i}",
                cta_button="Learn More",
                cost_usd=0.001,
            )
            for dim, score in [
                ("clarity", 7.0 + i * 0.5),
                ("value_prop", 6.5 + i * 0.3),
                ("cta_effectiveness", 7.0),
                ("brand_voice", 6.0 + i),
                ("emotional_resonance", 7.5),
            ]:
                insert_evaluation(
                    db_conn,
                    ad_id=ad_id,
                    dimension=dim,
                    score=score,
                    eval_mode="final",
                )
                # Also insert iteration-mode evals for before/after test
                insert_evaluation(
                    db_conn,
                    ad_id=ad_id,
                    dimension=dim,
                    score=score - 1.0,
                    eval_mode="iteration",
                )

        insert_quality_snapshot(
            db_conn,
            cycle_number=1,
            avg_weighted_score=7.0,
            dimension_averages={"clarity": 7.0, "value_prop": 6.5},
            quality_per_dollar=100.0,
            token_spend_usd=0.07,
        )
        insert_quality_snapshot(
            db_conn,
            cycle_number=2,
            avg_weighted_score=7.5,
            dimension_averages={"clarity": 7.5, "value_prop": 7.0},
            quality_per_dollar=110.0,
            token_spend_usd=0.068,
        )

        return db_conn

    def test_score_distribution_with_data(self, seeded_db) -> None:
        fig = score_distribution(seeded_db)
        assert fig.layout.title.text == "Score Distribution"
        assert len(fig.data) > 0

    def test_convergence_curves_with_data(self, seeded_db) -> None:
        fig = convergence_curves(seeded_db)
        assert fig.layout.title.text == "Convergence Curves"
        assert len(fig.data) > 0  # At least the weighted avg line

    def test_dimension_breakdown_with_data(self, seeded_db) -> None:
        fig = dimension_breakdown(seeded_db)
        assert fig.layout.title.text == "Dimension Breakdown"
        assert len(fig.data) == 1  # One bar trace
        assert len(fig.data[0].x) == 5  # 5 dimensions

    def test_before_after_with_data(self, seeded_db) -> None:
        # Get an ad_id from the DB
        row = seeded_db.execute("SELECT id FROM ads LIMIT 1").fetchone()
        ad_id = row[0]
        fig = before_after_comparison(seeded_db, ad_id)
        assert "Before/After" in fig.layout.title.text
        assert len(fig.data) == 2  # Before and After traces

    def test_cost_efficiency_with_data(self, seeded_db) -> None:
        fig = cost_efficiency_trend(seeded_db)
        assert fig.layout.title.text == "Cost Efficiency Trend"
        assert len(fig.data) == 2  # quality_per_dollar + spend lines
