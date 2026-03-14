"""Tests for calibration metric utilities.

Pure math tests -- no API mocking, no EvaluationEngine. These define the
contract for calculate_metrics() and check_gold_set_overlap() that will be
implemented in the calibrate.py rewrite story.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.db.init_db import init_db
from src.db.queries import insert_calibration_run
from src.evaluate.calibrate import (
    ALPHA_THRESHOLD,
    CONSECUTIVE_FAILURES,
    DRIFT_WINDOW,
    MAE_INCREASE_RUNS,
    calculate_metrics,
    calculate_mid_range_metrics,
    check_gold_set_overlap,
    detect_drift,
    filter_mid_range_scores,
)

# ---------------------------------------------------------------------------
# Tests: Krippendorff's Alpha
# ---------------------------------------------------------------------------


def test_alpha_perfect_agreement():
    """Identical human and LLM ratings produce alpha >= 0.99."""
    human = [8, 3, 9, 2, 7]
    llm = [8, 3, 9, 2, 7]

    result = calculate_metrics(human, llm)

    assert result["alpha"] >= 0.99


def test_alpha_random_disagreement():
    """Shuffled ratings with no systematic agreement produce alpha < 0.3."""
    human = [8, 3, 9, 2, 7]
    llm = [2, 9, 3, 7, 8]

    result = calculate_metrics(human, llm)

    assert result["alpha"] < 0.3


# ---------------------------------------------------------------------------
# Tests: Spearman rank correlation
# ---------------------------------------------------------------------------


def test_spearman_preserves_ranking():
    """Different values with identical rank ordering produce rho >= 0.99."""
    human = [2, 4, 6, 8, 10]
    llm = [1, 3, 5, 7, 9]

    result = calculate_metrics(human, llm)

    assert result["spearman_rho"] >= 0.99


# ---------------------------------------------------------------------------
# Tests: per-dimension MAE
# ---------------------------------------------------------------------------


def test_mae_per_dimension_calculation():
    """Known per-dimension scores produce exact MAE values."""
    human_scores = {"clarity": 8.0, "value_prop": 6.0, "brand_voice": 7.0}
    llm_scores = {"clarity": 6.0, "value_prop": 6.0, "brand_voice": 9.0}

    result = calculate_metrics(
        list(human_scores.values()),
        list(llm_scores.values()),
    )

    # Overall MAE = mean(|8-6|, |6-6|, |7-9|) = mean(2, 0, 2) = 4/3
    assert result["mae"] == pytest.approx(4.0 / 3.0)


# ---------------------------------------------------------------------------
# Tests: gold set overlap guard
# ---------------------------------------------------------------------------


def test_gold_set_overlap_guard(tmp_path):
    """Overlap between few-shot and gold set ad IDs raises ValueError."""
    few_shot_path = tmp_path / "few_shot.json"
    gold_set_path = tmp_path / "gold_set.json"

    few_shot_path.write_text(json.dumps([{"id": "ad-001"}, {"id": "ad-002"}]))
    gold_set_path.write_text(json.dumps([{"id": "ad-002"}, {"id": "ad-003"}]))

    with pytest.raises(ValueError, match="ad-002"):
        check_gold_set_overlap(few_shot_path, gold_set_path)

    # Clean case: no overlap should not raise
    gold_set_clean = tmp_path / "gold_set_clean.json"
    gold_set_clean.write_text(json.dumps([{"id": "ad-003"}, {"id": "ad-004"}]))
    check_gold_set_overlap(few_shot_path, gold_set_clean)


# ---------------------------------------------------------------------------
# Tests: Calibration drift detection
# ---------------------------------------------------------------------------


def _make_conn():
    """Create an in-memory database with the full schema."""
    return init_db(":memory:")


def _seed_calibration_runs(conn, runs: list[dict]) -> None:
    """Insert calibration runs in order. Each dict can override defaults."""
    defaults = {
        "model_version": "gemini-2.5-pro",
        "alpha_overall": 0.9,
        "spearman_rho": 0.8,
        "mae_per_dimension": {
            "clarity": 0.5,
            "learner_benefit": 0.5,
            "cta_effectiveness": 0.5,
            "brand_voice": 0.5,
            "student_empathy": 0.5,
            "pedagogical_integrity": 0.5,
        },
        "ad_count": 4,
        "passed": True,
    }
    for run in runs:
        kwargs = {**defaults}
        # Handle alpha_overall override -> goes into kwargs directly
        if "alpha_overall" in run:
            kwargs["alpha_overall"] = run["alpha_overall"]
        # Handle per-dimension MAE overrides
        mae = dict(kwargs["mae_per_dimension"])
        for key, val in run.items():
            if key.startswith("mae_"):
                dim = key.removeprefix("mae_")
                mae[dim] = val
        kwargs["mae_per_dimension"] = mae
        insert_calibration_run(conn, **kwargs)


def test_detect_drift_no_runs():
    """No calibration runs produces no alerts."""
    conn = _make_conn()
    alerts = detect_drift(conn)
    assert alerts == []
    conn.close()


def test_detect_drift_healthy():
    """All runs above threshold produces no alerts."""
    conn = _make_conn()
    _seed_calibration_runs(conn, [{"alpha_overall": 0.95}] * DRIFT_WINDOW)
    alerts = detect_drift(conn)
    assert alerts == []
    conn.close()


@patch("src.evaluate.calibrate.log_decision")
def test_detect_drift_alpha_consecutive_failures(mock_log):
    """CONSECUTIVE_FAILURES runs below ALPHA_THRESHOLD triggers alpha_drift."""
    conn = _make_conn()
    healthy = [{"alpha_overall": 0.9}] * 2
    failing = [{"alpha_overall": ALPHA_THRESHOLD - 0.1}] * CONSECUTIVE_FAILURES
    _seed_calibration_runs(conn, healthy + failing)

    alerts = detect_drift(conn)

    alpha_alerts = [a for a in alerts if a.alert_type == "alpha_drift"]
    assert len(alpha_alerts) == 1
    assert alpha_alerts[0].detail["consecutive_failures"] == CONSECUTIVE_FAILURES
    conn.close()


@patch("src.evaluate.calibrate.log_decision")
def test_detect_drift_alpha_broken_streak(mock_log):
    """A passing run in the middle breaks the consecutive failure streak."""
    conn = _make_conn()
    runs = [
        {"alpha_overall": ALPHA_THRESHOLD - 0.1},
        {"alpha_overall": ALPHA_THRESHOLD - 0.1},
        {"alpha_overall": 0.95},  # breaks the streak
        {"alpha_overall": ALPHA_THRESHOLD - 0.1},
        {"alpha_overall": ALPHA_THRESHOLD - 0.1},
    ]
    _seed_calibration_runs(conn, runs)

    alerts = detect_drift(conn)

    alpha_alerts = [a for a in alerts if a.alert_type == "alpha_drift"]
    assert len(alpha_alerts) == 0
    conn.close()


@patch("src.evaluate.calibrate.log_decision")
def test_detect_drift_mae_increasing(mock_log):
    """Monotonically increasing MAE over MAE_INCREASE_RUNS triggers mae_drift."""
    conn = _make_conn()
    runs = []
    for i in range(MAE_INCREASE_RUNS):
        runs.append({"mae_clarity": 0.5 + i * 0.1})
    _seed_calibration_runs(conn, runs)

    alerts = detect_drift(conn)

    mae_alerts = [a for a in alerts if a.alert_type == "mae_drift"]
    assert len(mae_alerts) == 1
    assert mae_alerts[0].detail["dimension"] == "clarity"
    conn.close()


@patch("src.evaluate.calibrate.log_decision")
def test_detect_drift_mae_stable(mock_log):
    """Stable MAE values produce no mae_drift alerts."""
    conn = _make_conn()
    runs = [{"mae_clarity": 0.5}] * DRIFT_WINDOW
    _seed_calibration_runs(conn, runs)

    alerts = detect_drift(conn)

    mae_alerts = [a for a in alerts if a.alert_type == "mae_drift"]
    assert len(mae_alerts) == 0
    conn.close()


@patch("src.evaluate.calibrate.log_decision")
def test_detect_drift_logs_decisions(mock_log):
    """Each drift alert triggers a log_decision call."""
    conn = _make_conn()
    runs = []
    for i in range(max(CONSECUTIVE_FAILURES, MAE_INCREASE_RUNS)):
        runs.append(
            {
                "alpha_overall": ALPHA_THRESHOLD - 0.1,
                "mae_clarity": 0.5 + i * 0.1,
            }
        )
    _seed_calibration_runs(conn, runs)

    alerts = detect_drift(conn)

    assert len(alerts) >= 2
    assert mock_log.call_count == len(alerts)
    for call in mock_log.call_args_list:
        assert call.args[0] == "calibration"
        assert call.args[1] in ("alpha_drift", "mae_drift")
    conn.close()


# ---------------------------------------------------------------------------
# Tests: filter_mid_range_scores
# ---------------------------------------------------------------------------


def test_filter_mid_range_inclusive_bounds():
    """Scores exactly at bounds are included."""
    human = [4.0, 6.0, 3.0, 7.0]
    llm = [8.0, 2.0, 5.0, 9.0]

    h_out, l_out = filter_mid_range_scores(human, llm)

    # pair 0: human=4.0 in [4,6] -> yes
    # pair 1: human=6.0 in [4,6] -> yes
    # pair 2: llm=5.0 in [4,6] -> yes
    # pair 3: neither in [4,6] -> no
    assert h_out == [4.0, 6.0, 3.0]
    assert l_out == [8.0, 2.0, 5.0]


def test_filter_mid_range_exclusive_outside():
    """Scores strictly outside bounds on both sides are excluded."""
    human = [3.9, 6.1]
    llm = [3.0, 7.0]

    h_out, l_out = filter_mid_range_scores(human, llm)

    assert h_out == []
    assert l_out == []


def test_filter_mid_range_either_side():
    """Pair qualifies when only the LLM score falls in bounds."""
    human = [8.0]
    llm = [5.0]

    h_out, l_out = filter_mid_range_scores(human, llm)

    assert h_out == [8.0]
    assert l_out == [5.0]


def test_filter_mid_range_empty_inputs():
    """Empty input lists return empty output lists."""
    h_out, l_out = filter_mid_range_scores([], [])

    assert h_out == []
    assert l_out == []


def test_filter_mid_range_mismatched_lengths():
    """Mismatched list lengths raise ValueError."""
    with pytest.raises(ValueError, match="length"):
        filter_mid_range_scores([1.0, 2.0], [3.0])


# ---------------------------------------------------------------------------
# Tests: calculate_mid_range_metrics
# ---------------------------------------------------------------------------


def test_mid_range_metrics_known_values():
    """Known mid-range subset produces correct alpha, mae, count, pct_of_total."""
    human = [4.0, 5.0, 6.0, 4.5, 1.0, 9.0]
    llm = [4.0, 5.0, 6.0, 4.5, 1.0, 9.0]

    result = calculate_mid_range_metrics(human, llm)

    # First 4 pairs have at least one side in [4,6], last 2 have neither
    assert result["count"] == 4
    assert result["pct_of_total"] == pytest.approx(4 / 6 * 100, abs=0.1)
    assert result["alpha"] >= 0.99
    assert result["mae"] == pytest.approx(0.0)


def test_mid_range_metrics_insufficient_data():
    """Fewer than 3 qualifying pairs returns insufficient_data=True."""
    human = [5.0, 5.0, 1.0, 9.0]
    llm = [5.0, 5.0, 1.0, 9.0]

    result = calculate_mid_range_metrics(human, llm)

    # Only 2 pairs in [4,6]
    assert result["insufficient_data"] is True
    assert result["count"] == 2


def test_mid_range_metrics_pct_of_total():
    """pct_of_total reflects proportion of qualifying pairs."""
    human = [4.0, 4.5, 5.0, 5.5, 6.0, 4.0, 4.5, 5.0, 5.5, 6.0]
    llm = [4.0, 4.5, 5.0, 5.5, 6.0, 4.0, 4.5, 5.0, 5.5, 6.0]

    result = calculate_mid_range_metrics(human, llm)

    assert result["count"] == 10
    assert result["pct_of_total"] == pytest.approx(100.0)
