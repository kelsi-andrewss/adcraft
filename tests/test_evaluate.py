"""Tests for the evaluation engine.

All Gemini API calls are mocked — no API keys required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError, ServerError

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
        "learner_benefit": 8.0,
        "cta_effectiveness": 7.0,
        "brand_voice": 8.0,
        "student_empathy": 7.0,
        "pedagogical_integrity": 7.0,
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
    """Iteration mode returns scores for all 6 dimensions."""
    mock_client = MagicMock()
    response_data = _make_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert len(result.scores) == 6
    dims_returned = {s.dimension for s in result.scores}
    assert dims_returned == set(DIMENSIONS)
    # Single API call in iteration mode
    assert mock_client.models.generate_content.call_count == 1


# ---------------------------------------------------------------------------
# Tests: final mode
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_final_mode_makes_separate_calls(mock_log):
    """Final mode makes 6 separate API calls, one per dimension."""
    mock_client = MagicMock()

    responses = []
    for dim in DIMENSIONS:
        responses.append(_mock_response(_make_single_dimension_response(dim, 7.5)))
    mock_client.models.generate_content.side_effect = responses

    engine = _make_engine(mock_client)
    result = engine.evaluate_final(_make_ad())

    assert len(result.scores) == 6
    assert mock_client.models.generate_content.call_count == 6


# ---------------------------------------------------------------------------
# Tests: weighted average
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_weighted_average_calculation(mock_log):
    """Weighted average matches manual calculation."""
    scores = {
        "clarity": 8.0,
        "learner_benefit": 6.0,
        "cta_effectiveness": 7.0,
        "brand_voice": 9.0,
        "student_empathy": 5.0,
        "pedagogical_integrity": 7.0,
    }
    expected = sum(scores[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert abs(result.weighted_average - expected) < 0.01


# ---------------------------------------------------------------------------
# Tests: hard gate — brand_voice
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_hard_gate_brand_voice_below_5_fails(mock_log):
    """brand_voice < 5 triggers hard gate failure regardless of other scores."""
    scores = {
        "clarity": 9.0,
        "learner_benefit": 9.0,
        "cta_effectiveness": 9.0,
        "brand_voice": 4.0,  # Below hard gate
        "student_empathy": 9.0,
        "pedagogical_integrity": 9.0,
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
        "learner_benefit": 7.5,
        "cta_effectiveness": 7.5,
        "brand_voice": 5.0,  # At the gate — should pass
        "student_empathy": 7.5,
        "pedagogical_integrity": 7.5,
    }

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert result.hard_gate_failures == []
    assert result.passed_threshold is True


# ---------------------------------------------------------------------------
# Tests: hard gate — pedagogical_integrity
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_hard_gate_pedagogical_integrity_below_6_fails(mock_log):
    """pedagogical_integrity < 6 triggers hard gate failure."""
    scores = {
        "clarity": 9.0,
        "learner_benefit": 9.0,
        "cta_effectiveness": 9.0,
        "brand_voice": 9.0,
        "student_empathy": 9.0,
        "pedagogical_integrity": 5.9,  # Below hard gate of 6
    }

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert result.passed_threshold is False
    assert "pedagogical_integrity" in result.hard_gate_failures
    assert result.weighted_average > PASSING_THRESHOLD


@patch("src.evaluate.engine.log_decision")
def test_hard_gate_pedagogical_integrity_at_6_passes(mock_log):
    """pedagogical_integrity == 6 (at the gate) does NOT trigger hard gate."""
    scores = {
        "clarity": 7.5,
        "learner_benefit": 7.5,
        "cta_effectiveness": 7.5,
        "brand_voice": 7.5,
        "student_empathy": 7.5,
        "pedagogical_integrity": 6.0,
    }

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert "pedagogical_integrity" not in result.hard_gate_failures
    assert result.passed_threshold is True


@patch("src.evaluate.engine.log_decision")
def test_dual_hard_gate_failure(mock_log):
    """Both brand_voice and pedagogical_integrity can fail simultaneously."""
    scores = {
        "clarity": 9.0,
        "learner_benefit": 9.0,
        "cta_effectiveness": 9.0,
        "brand_voice": 4.0,  # Below brand_voice gate of 5
        "student_empathy": 9.0,
        "pedagogical_integrity": 5.9,  # Below pedagogical_integrity gate of 6
    }

    mock_client = MagicMock()
    response_data = _make_all_dimensions_response(scores)
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert result.passed_threshold is False
    assert "brand_voice" in result.hard_gate_failures
    assert "pedagogical_integrity" in result.hard_gate_failures
    assert len(result.hard_gate_failures) == 2


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
    """Engine retries on transient API error then succeeds."""
    mock_client = MagicMock()
    good_response = _mock_response(_make_all_dimensions_response())
    mock_client.models.generate_content.side_effect = [
        ServerError(500, {"error": {"message": "Internal error", "status": "INTERNAL"}}),
        good_response,
    ]

    engine = _make_engine(mock_client)
    result = engine.evaluate_iteration(_make_ad())

    assert len(result.scores) == 6
    assert mock_client.models.generate_content.call_count == 2


@patch("src.evaluate.engine.log_decision")
def test_no_retry_on_client_error(mock_log):
    """Engine does NOT retry on non-transient client errors (e.g., 400)."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ClientError(
        400, {"error": {"message": "Invalid request", "status": "INVALID_ARGUMENT"}}
    )

    engine = _make_engine(mock_client)
    with pytest.raises(ClientError):
        engine.evaluate_iteration(_make_ad())

    assert mock_client.models.generate_content.call_count == 1


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

    # Should have: evaluate_iteration_start + 6 dimension_scores + evaluation_complete = 8
    assert mock_log.call_count >= 8

    # Check that evaluator component was used
    components = [call.args[0] for call in mock_log.call_args_list]
    assert all(c == "evaluator" for c in components)

    # Check that dimension scores were logged
    actions = [call.args[1] for call in mock_log.call_args_list]
    assert actions.count("dimension_score") == 6


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
    """Final mode accumulates token counts from all 6 calls."""
    mock_client = MagicMock()
    responses = []
    for dim in DIMENSIONS:
        responses.append(
            _mock_response(_make_single_dimension_response(dim, 7.0), total_tokens=200)
        )
    mock_client.models.generate_content.side_effect = responses

    engine = _make_engine(mock_client)
    result = engine.evaluate_final(_make_ad())

    assert result.token_count == 1200  # 200 * 6


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
