"""Tests for WeightEvolver shadow analysis and extracted helpers."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from src.analytics.weights import WeightEvolver
from src.db.init_db import init_db
from src.db.queries import insert_ad, insert_evaluation
from src.evaluate.rubrics import DIMENSIONS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

UNIFORM_WEIGHTS = {d: 1.0 / len(DIMENSIONS) for d in DIMENSIONS}


@pytest.fixture()
def db_conn():
    conn = init_db(":memory:")
    yield conn
    conn.close()


def _seed_ads(conn, count=20):
    """Insert `count` ads with all 5 dimensions scored.

    Scores are deterministic: dimension index and ad index drive variation
    so tests can reason about rankings.
    """
    ad_ids = []
    for i in range(count):
        ad_id = insert_ad(
            conn,
            primary_text=f"Ad text {i}",
            headline=f"Headline {i}",
            description=f"Description {i}",
            cta_button="Learn More",
        )
        ad_ids.append(ad_id)
        for j, dim in enumerate(DIMENSIONS):
            score = 5.0 + (i * 0.2) + (j * 0.5)
            score = min(score, 10.0)
            insert_evaluation(conn, ad_id=ad_id, dimension=dim, score=score)
    return ad_ids


@pytest.fixture()
def seeded_db(db_conn):
    _seed_ads(db_conn, count=20)
    return db_conn


# ---------------------------------------------------------------------------
# _compute_weighted_scores
# ---------------------------------------------------------------------------


class TestComputeWeightedScores:
    def test_basic(self, db_conn):
        evolver = WeightEvolver(db_conn, min_sample_size=1)
        complete_ads = {
            "ad1": {d: 8.0 for d in DIMENSIONS},
            "ad2": {d: 6.0 for d in DIMENSIONS},
        }
        weights = {d: 0.2 for d in DIMENSIONS}
        result = evolver._compute_weighted_scores(complete_ads, weights)

        assert len(result) == 2
        assert result["ad1"] == pytest.approx(8.0)
        assert result["ad2"] == pytest.approx(6.0)

    def test_uniform_weights(self, db_conn):
        evolver = WeightEvolver(db_conn, min_sample_size=1)
        scores = {}
        for i, d in enumerate(DIMENSIONS):
            scores[d] = float(i + 1)

        complete_ads = {"ad1": scores}
        result = evolver._compute_weighted_scores(complete_ads, UNIFORM_WEIGHTS)

        expected = sum(scores.values()) / len(DIMENSIONS)
        assert result["ad1"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _fetch_complete_ads
# ---------------------------------------------------------------------------


class TestFetchCompleteAds:
    def test_returns_complete_ads(self, seeded_db):
        evolver = WeightEvolver(seeded_db, min_sample_size=1)
        result = evolver._fetch_complete_ads()
        assert len(result) == 20
        for scores in result.values():
            assert set(scores.keys()) == set(DIMENSIONS)

    def test_filters_incomplete(self, db_conn):
        ad_id = insert_ad(
            db_conn,
            primary_text="Incomplete",
            headline="H",
            description="D",
            cta_button="CTA",
        )
        insert_evaluation(db_conn, ad_id=ad_id, dimension="clarity", score=7.0)
        insert_evaluation(db_conn, ad_id=ad_id, dimension="value_prop", score=6.0)

        evolver = WeightEvolver(db_conn, min_sample_size=1)
        result = evolver._fetch_complete_ads()
        assert len(result) == 0

    def test_empty_db(self, db_conn):
        evolver = WeightEvolver(db_conn, min_sample_size=1)
        result = evolver._fetch_complete_ads()
        assert result == {}


# ---------------------------------------------------------------------------
# run_shadow_analysis
# ---------------------------------------------------------------------------


class TestShadowAnalysis:
    def test_identical_weights(self, seeded_db):
        evolver = WeightEvolver(seeded_db, min_sample_size=1)
        current = dict(evolver._initial_weights)
        result = evolver.run_shadow_analysis(current)

        assert result["status"] == "complete"
        assert result["sample_count"] == 20
        assert result["mean_absolute_deviation"] == pytest.approx(0.0, abs=1e-6)
        assert result["pearson_correlation"] == pytest.approx(1.0, abs=1e-6)
        assert result["top_decile_shift_count"] == 0
        assert result["top_decile_shift_pct"] == pytest.approx(0.0)

    def test_detects_score_shifts(self, seeded_db):
        evolver = WeightEvolver(seeded_db, min_sample_size=1)
        shifted_weights = {d: 0.2 for d in DIMENSIONS}
        shifted_weights["clarity"] = 0.6
        shifted_weights["emotional_resonance"] = 0.05
        # Re-normalize to sum to ~1
        total = sum(shifted_weights.values())
        shifted_weights = {d: w / total for d, w in shifted_weights.items()}

        result = evolver.run_shadow_analysis(shifted_weights)

        assert result["status"] == "complete"
        assert result["mean_absolute_deviation"] > 0
        assert (
            result["score_deltas_summary"]["max_positive_delta"]
            != result["score_deltas_summary"]["max_negative_delta"]
        )

    def test_ranking_shift(self, seeded_db):
        evolver = WeightEvolver(seeded_db, min_sample_size=1)

        # Extreme weight shift: put all weight on one dimension
        extreme_weights = {d: 0.01 for d in DIMENSIONS}
        extreme_weights[DIMENSIONS[0]] = 0.96
        result = evolver.run_shadow_analysis(extreme_weights)

        assert result["status"] == "complete"
        top_k = math.ceil(20 * 0.10)
        assert result["top_decile_shift_count"] <= top_k
        assert 0.0 <= result["top_decile_shift_pct"] <= 1.0

    def test_insufficient_data(self, db_conn):
        _seed_ads(db_conn, count=5)
        evolver = WeightEvolver(db_conn, min_sample_size=50)
        result = evolver.run_shadow_analysis({d: 0.2 for d in DIMENSIONS})

        assert result["status"] == "insufficient_data"
        assert result["sample_count"] == 5

    def test_logs_decision(self, seeded_db):
        evolver = WeightEvolver(seeded_db, min_sample_size=1)
        with patch("src.analytics.weights.log_decision") as mock_log:
            evolver.run_shadow_analysis(dict(evolver._initial_weights))

        actions = [call.args[1] for call in mock_log.call_args_list]
        assert "shadow_analysis_complete" in actions


# ---------------------------------------------------------------------------
# evolve integration
# ---------------------------------------------------------------------------


class TestEvolveIncludesShadowAnalysis:
    def test_evolve_includes_shadow_analysis(self, seeded_db):
        evolver = WeightEvolver(seeded_db, min_sample_size=1)
        result = evolver.evolve()

        assert result["status"] == "complete"
        # recommend_weights normalizes correlations, which won't exactly match
        # initial weights, so shadow_analysis should be present
        if result["recommended_weights"] != evolver._initial_weights:
            assert "shadow_analysis" in result
            assert result["shadow_analysis"]["status"] == "complete"

    def test_evolve_insufficient_data(self, db_conn):
        _seed_ads(db_conn, count=3)
        evolver = WeightEvolver(db_conn, min_sample_size=50)
        result = evolver.evolve()

        assert result["status"] == "insufficient_data"
        assert "shadow_analysis" not in result
