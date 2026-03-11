"""Tests for the evaluation engine.

All Gemini API calls are mocked — no API keys required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.evaluate.engine import EvaluationEngine
from src.evaluate.rubrics import (
    DIMENSION_WEIGHTS,
    DIMENSIONS,
    PASSING_THRESHOLD,
)
from src.models.ad import AdCopy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ad(**kwargs) -> AdCopy:
    defaults = {
        "id": "test-ad-001",
        "primary_text": "Test primary text about SAT prep.",
        "headline": "Test Headline",
        "description": "Test description for SAT tutoring.",
        "cta_button": "Learn More",
    }
    defaults.update(kwargs)
    return AdCopy(**defaults)


def _make_all_dimensions_response(
    scores: dict[str, float] | None = None,
) -> dict:
    """Build a mock all-dimensions JSON response."""
    default_scores = {
        "clarity": 7.0,
        "value_prop": 8.0,
        "cta_effectiveness": 7.0,
        "brand_voice": 8.0,
        "emotional_resonance": 7.0,
    }
    if scores:
        default_scores.update(scores)
    return {
        dim: {
            "rationale": f"Good performance on {dim}.",
            "score": default_scores[dim],
            "confidence": 0.9,
        }
        for dim in DIMENSIONS
    }


def _make_single_dimension_response(dim: str, score: float) -> dict:
    """Build a mock single-dimension JSON response."""
    return {
        "rationale": f"Detailed analysis of {dim}.",
        "score": score,
        "confidence": 0.85,
    }


def _mock_response(data: dict, total_tokens: int = 100) -> MagicMock:
    """Create a mock Gemini response object."""
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.total_token_count = total_tokens
    return resp


def _make_engine(mock_client: MagicMock | None = None) -> EvaluationEngine:
    """Create engine with a mock client."""
    client = mock_client or MagicMock()
    return EvaluationEngine(client=client)


# ---------------------------------------------------------------------------
# Tests: iteration mode
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_iteration_mode_returns_all_dimensions(mock_log):
    """Iteration mode returns scores for all 5 dimensions."""
    mock_client = MagicMock()
    response_data = _make_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert len(result.scores) == 5
    dims_returned = {s.dimension for s in result.scores}
    assert dims_returned == set(DIMENSIONS)
    # Single API call in iteration mode
    assert mock_client.models.generate_content.call_count == 1


# ---------------------------------------------------------------------------
# Tests: final mode
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_final_mode_makes_separate_calls(mock_log):
    """Final mode makes 5 separate API calls, one per dimension."""
    mock_client = MagicMock()

    responses = []
    for dim in DIMENSIONS:
        responses.append(_mock_response(_make_single_dimension_response(dim, 7.5)))
    mock_client.models.generate_content.side_effect = responses

    engine = _make_engine(mock_client)
    result = engine.evaluate_final(_make_ad())

    assert len(result.scores) == 5
    assert mock_client.models.generate_content.call_count == 5


# ---------------------------------------------------------------------------
# Tests: weighted average
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_weighted_average_calculation(mock_log):
    """Weighted average matches manual calculation."""
    scores = {
        "clarity": 8.0,
        "value_prop": 6.0,
        "cta_effectiveness": 7.0,
        "brand_voice": 9.0,
        "emotional_resonance": 5.0,
    }
    expected = sum(scores[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert abs(result.weighted_average - expected) < 0.01


# ---------------------------------------------------------------------------
# Tests: hard gate
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_hard_gate_brand_voice_below_5_fails(mock_log):
    """brand_voice < 5 triggers hard gate failure regardless of other scores."""
    scores = {
        "clarity": 9.0,
        "value_prop": 9.0,
        "cta_effectiveness": 9.0,
        "brand_voice": 4.0,  # Below hard gate
        "emotional_resonance": 9.0,
    }

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert result.passed_threshold is False
    assert "brand_voice" in result.hard_gate_failures
    # Weighted average should still be high (it's only the hard gate that fails)
    assert result.weighted_average > PASSING_THRESHOLD


@patch("src.evaluate.engine.log_decision")
def test_hard_gate_brand_voice_at_5_passes(mock_log):
    """brand_voice == 5 (at the gate) does NOT trigger hard gate."""
    scores = {
        "clarity": 7.5,
        "value_prop": 7.5,
        "cta_effectiveness": 7.5,
        "brand_voice": 5.0,  # At the gate — should pass
        "emotional_resonance": 7.5,
    }

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert result.hard_gate_failures == []
    assert result.passed_threshold is True


# ---------------------------------------------------------------------------
# Tests: passing threshold
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_passing_threshold_at_7_0_passes(mock_log):
    """Weighted average == 7.0 passes the threshold."""
    # Engineer scores so weighted average == 7.0
    # All dimensions at 7.0 gives weighted avg of 7.0
    scores = {d: 7.0 for d in DIMENSIONS}

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert result.weighted_average == pytest.approx(PASSING_THRESHOLD)
    assert result.passed_threshold is True


@patch("src.evaluate.engine.log_decision")
def test_passing_threshold_at_6_9_fails(mock_log):
    """Weighted average == 6.9 fails the threshold."""
    scores = {d: 6.9 for d in DIMENSIONS}

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert result.weighted_average < PASSING_THRESHOLD
    assert result.passed_threshold is False


# ---------------------------------------------------------------------------
# Tests: retry
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_retry_on_api_error(mock_log):
    """Engine retries on API error then succeeds."""
    mock_client = MagicMock()
    good_response = _mock_response(_make_all_dimensions_response())
    mock_client.models.generate_content.side_effect = [
        Exception("429 rate limit"),
        good_response,
    ]

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert len(result.scores) == 5
    assert mock_client.models.generate_content.call_count == 2


# ---------------------------------------------------------------------------
# Tests: CoT rationale
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_cot_rationale_extracted(mock_log):
    """Rationale field is populated from model response."""
    response_data = {
        dim: {
            "rationale": f"Detailed reasoning about {dim} quality.",
            "score": 7.0,
            "confidence": 0.9,
        }
        for dim in DIMENSIONS
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    for s in result.scores:
        assert s.rationale.startswith("Detailed reasoning about")
        assert len(s.rationale) > 10


# ---------------------------------------------------------------------------
# Tests: calibration (mocked)
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_calibration_great_ad_scores_above_7(mock_log):
    """A 'great' reference ad (mocked high scores) gets weighted avg >= 7.0."""
    scores = {
        "clarity": 8.0,
        "value_prop": 9.0,
        "cta_effectiveness": 8.0,
        "brand_voice": 8.0,
        "emotional_resonance": 7.0,
    }

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    ad = _make_ad(
        id="ref-great-001",
        primary_text="Your child's SAT score shouldn't be limited by access to great teaching.",
        headline="Average 160-Point SAT Score Improvement",
    )
    result = engine.evaluate_iteration(ad)

    assert result.weighted_average >= 7.0
    assert result.passed_threshold is True


@patch("src.evaluate.engine.log_decision")
def test_calibration_bad_ad_scores_below_5(mock_log):
    """A 'bad' reference ad (mocked low scores) gets weighted avg < 5.0."""
    scores = {
        "clarity": 4.0,
        "value_prop": 2.0,
        "cta_effectiveness": 3.0,
        "brand_voice": 3.0,
        "emotional_resonance": 2.0,
    }

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    ad = _make_ad(
        id="ref-bad-001",
        primary_text="Looking for SAT help? We offer tutoring services.",
        headline="SAT Tutoring Available Now",
    )
    result = engine.evaluate_iteration(ad)

    assert result.weighted_average < 5.0
    assert result.passed_threshold is False


# ---------------------------------------------------------------------------
# Tests: decision logging
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_log_decision_called(mock_log):
    """log_decision is called for scoring decisions."""
    mock_client = MagicMock()
    response_data = _make_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    engine.evaluate_iteration(_make_ad())

    # Should have: evaluate_iteration_start + 5 dimension_scores + evaluation_complete = 7
    assert mock_log.call_count >= 7

    # Check that evaluator component was used
    components = [call.args[0] for call in mock_log.call_args_list]
    assert all(c == "evaluator" for c in components)

    # Check that dimension scores were logged
    actions = [call.args[1] for call in mock_log.call_args_list]
    assert actions.count("dimension_score") == 5


# ---------------------------------------------------------------------------
# Tests: token count
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_token_count_captured(mock_log):
    """Token count from usage_metadata flows into EvaluationResult."""
    mock_client = MagicMock()
    response_data = _make_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(
        response_data, total_tokens=1500
    )

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert result.token_count == 1500


@patch("src.evaluate.engine.log_decision")
def test_final_mode_accumulates_tokens(mock_log):
    """Final mode accumulates token counts from all 5 calls."""
    mock_client = MagicMock()
    responses = []
    for dim in DIMENSIONS:
        responses.append(
            _mock_response(_make_single_dimension_response(dim, 7.0), total_tokens=200)
        )
    mock_client.models.generate_content.side_effect = responses

    engine = _make_engine(mock_client)
    result = engine.evaluate_final(_make_ad())

    assert result.token_count == 1000  # 200 * 5


# ---------------------------------------------------------------------------
# Tests: evaluator model ID
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_evaluator_model_set(mock_log):
    """EvaluationResult records the evaluator model ID."""
    mock_client = MagicMock()
    response_data = _make_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert result.evaluator_model == "gemini-2.5-pro"
