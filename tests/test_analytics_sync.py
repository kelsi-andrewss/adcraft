"""Parity tests: SQL weighted average in score_distribution must match Python computation.

These tests verify that _build_weighted_avg_case() produces SQL that agrees with
Python-level arithmetic using DIMENSION_WEIGHTS, and that weight changes propagate
into the SQL without any manual SQL edits.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.analytics.trends import _build_weighted_avg_case
from src.db.queries import insert_ad, insert_evaluation
from src.evaluate.rubrics import DIMENSION_WEIGHTS

# ---------------------------------------------------------------------------
# Known scores fixture
# ---------------------------------------------------------------------------

KNOWN_SCORES: dict[str, float] = {
    "clarity": 8.0,
    "learner_benefit": 7.0,
    "cta_effectiveness": 6.0,
    "brand_voice": 5.0,
    "student_empathy": 9.0,
    "pedagogical_integrity": 7.5,
}


def _seed_ad_with_scores(
    conn: sqlite3.Connection, scores: dict[str, float], eval_mode: str = "final"
) -> str:
    """Insert one ad with known per-dimension scores. Returns the ad_id."""
    ad_id = insert_ad(
        conn,
        primary_text="Test ad",
        headline="Test headline",
        description="Test description",
        cta_button="Learn More",
        cost_usd=0.001,
    )
    for dim, score in scores.items():
        insert_evaluation(conn, ad_id=ad_id, dimension=dim, score=score, eval_mode=eval_mode)
    return ad_id


def _query_weighted_avg(conn: sqlite3.Connection, ad_id: str) -> float:
    """Run the dynamic CASE expression against the DB and return the weighted avg for one ad."""
    conn.row_factory = sqlite3.Row
    case_expr = _build_weighted_avg_case()
    row = conn.execute(
        f"""
        SELECT SUM(e.score * {case_expr}) as weighted_avg
        FROM evaluations e
        WHERE e.ad_id = ? AND (e.eval_mode = 'final' OR e.eval_mode IS NULL)
        """,
        (ad_id,),
    ).fetchone()
    assert row is not None, "Expected one result row"
    return row["weighted_avg"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sql_weighted_avg_matches_python(db_conn: sqlite3.Connection) -> None:
    """SQL CASE expression must produce the same weighted average as Python arithmetic."""
    ad_id = _seed_ad_with_scores(db_conn, KNOWN_SCORES, eval_mode="final")

    sql_result = _query_weighted_avg(db_conn, ad_id)
    python_result = sum(KNOWN_SCORES[dim] * DIMENSION_WEIGHTS[dim] for dim in DIMENSION_WEIGHTS)

    assert sql_result == pytest.approx(python_result, abs=1e-9)


def test_weight_change_propagates_to_sql(db_conn: sqlite3.Connection, monkeypatch) -> None:
    """Monkeypatching DIMENSION_WEIGHTS on the trends module must change SQL output."""
    import src.analytics.trends as trends_mod

    patched_weights = {dim: 0.0 for dim in DIMENSION_WEIGHTS}
    # Put all weight on clarity so the expected result is clarity_score * 1.0
    patched_weights["clarity"] = 1.0

    monkeypatch.setattr(trends_mod, "DIMENSION_WEIGHTS", patched_weights)

    ad_id = _seed_ad_with_scores(db_conn, KNOWN_SCORES, eval_mode="final")

    sql_result = _query_weighted_avg(db_conn, ad_id)
    expected = KNOWN_SCORES["clarity"] * 1.0  # only clarity contributes

    assert sql_result == pytest.approx(expected, abs=1e-9)
