"""Tests for calibration metric utilities.

Pure math tests -- no API mocking, no EvaluationEngine. These define the
contract for calculate_metrics() and check_gold_set_overlap() that will be
implemented in the calibrate.py rewrite story.
"""

from __future__ import annotations

import json

import pytest

from src.evaluate.calibrate import calculate_metrics, check_gold_set_overlap

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
