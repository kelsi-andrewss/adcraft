"""Tests for composed ad evaluation (copy + image as a single unit).

All Gemini API calls are mocked — no API keys required.
Uses a 1x1 red PIL.Image as the test image.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from src.evaluate.composed import (
    COMPOSED_EVAL_PROMPT,
    COMPOSED_EVAL_SCHEMA,
    PUBLISHABLE_THRESHOLD,
    ComposedEvaluator,
)
from src.models.ad import AdCopy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_image() -> PILImage.Image:
    """1x1 red PIL.Image — no real image file needed."""
    return PILImage.new("RGB", (1, 1), color=(255, 0, 0))


@pytest.fixture()
def test_ad_copy() -> AdCopy:
    return AdCopy(
        id="composed-test-001",
        primary_text="Unlock your child's potential with expert SAT tutoring.",
        headline="Score Higher with Varsity Tutors",
        description="Personalized 1-on-1 SAT prep from expert tutors.",
        cta_button="Book a Free Session",
    )


def _make_evaluator(mock_client: MagicMock | None = None) -> ComposedEvaluator:
    client = mock_client or MagicMock()
    return ComposedEvaluator(client=client)


def _mock_response(data: dict, total_tokens: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.total_token_count = total_tokens
    return resp


def _make_composed_response(score: float = 7.5) -> dict:
    return {
        "rationale": (
            "The ad unit delivers a cohesive message. The image of a confident "
            "student reinforces the headline's promise of higher scores. Visual "
            "and textual elements are well-balanced for a feed context."
        ),
        "composed_score": score,
    }


# ---------------------------------------------------------------------------
# Tests: basic composed evaluation
# ---------------------------------------------------------------------------


@patch("src.evaluate.composed.log_decision")
def test_evaluate_composed_returns_expected_keys(mock_log, test_image, test_ad_copy):
    """Result dict contains composed_score, rationale, publishable, token_count."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response(8.0))

    evaluator = _make_evaluator(mock_client)
    result = evaluator.evaluate_composed(test_image, test_ad_copy)

    assert "composed_score" in result
    assert "rationale" in result
    assert "publishable" in result
    assert "token_count" in result


@patch("src.evaluate.composed.log_decision")
def test_evaluate_composed_score_extraction(mock_log, test_image, test_ad_copy):
    """Composed score is correctly extracted as a float."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response(8.2))

    evaluator = _make_evaluator(mock_client)
    result = evaluator.evaluate_composed(test_image, test_ad_copy)

    assert result["composed_score"] == 8.2
    assert isinstance(result["composed_score"], float)


@patch("src.evaluate.composed.log_decision")
def test_evaluate_composed_rationale_is_string(mock_log, test_image, test_ad_copy):
    """Rationale is a non-empty string."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response(7.0))

    evaluator = _make_evaluator(mock_client)
    result = evaluator.evaluate_composed(test_image, test_ad_copy)

    assert isinstance(result["rationale"], str)
    assert len(result["rationale"]) > 0


# ---------------------------------------------------------------------------
# Tests: publishable threshold logic
# ---------------------------------------------------------------------------


@patch("src.evaluate.composed.log_decision")
def test_publishable_true_above_threshold(mock_log, test_image, test_ad_copy):
    """Score above 7.0 is publishable."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response(8.5))

    evaluator = _make_evaluator(mock_client)
    result = evaluator.evaluate_composed(test_image, test_ad_copy)

    assert result["publishable"] is True


@patch("src.evaluate.composed.log_decision")
def test_publishable_true_at_threshold(mock_log, test_image, test_ad_copy):
    """Score exactly at 7.0 is publishable (>= threshold)."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(
        _make_composed_response(PUBLISHABLE_THRESHOLD)
    )

    evaluator = _make_evaluator(mock_client)
    result = evaluator.evaluate_composed(test_image, test_ad_copy)

    assert result["publishable"] is True


@patch("src.evaluate.composed.log_decision")
def test_publishable_false_below_threshold(mock_log, test_image, test_ad_copy):
    """Score below 7.0 is not publishable."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response(6.9))

    evaluator = _make_evaluator(mock_client)
    result = evaluator.evaluate_composed(test_image, test_ad_copy)

    assert result["publishable"] is False


@patch("src.evaluate.composed.log_decision")
def test_publishable_false_low_score(mock_log, test_image, test_ad_copy):
    """Very low score is not publishable."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response(3.0))

    evaluator = _make_evaluator(mock_client)
    result = evaluator.evaluate_composed(test_image, test_ad_copy)

    assert result["publishable"] is False
    assert result["composed_score"] == 3.0


# ---------------------------------------------------------------------------
# Tests: single API call
# ---------------------------------------------------------------------------


@patch("src.evaluate.composed.log_decision")
def test_single_api_call(mock_log, test_image, test_ad_copy):
    """Composed eval uses exactly 1 API call."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response())

    evaluator = _make_evaluator(mock_client)
    evaluator.evaluate_composed(test_image, test_ad_copy)

    assert mock_client.models.generate_content.call_count == 1


# ---------------------------------------------------------------------------
# Tests: multimodal content assembly
# ---------------------------------------------------------------------------


@patch("src.evaluate.composed.log_decision")
def test_content_list_structure(mock_log, test_image, test_ad_copy):
    """Content list passed to Gemini is [str, PILImage.Image, str]."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response())

    evaluator = _make_evaluator(mock_client)
    evaluator.evaluate_composed(test_image, test_ad_copy)

    call_args = mock_client.models.generate_content.call_args
    contents = call_args.kwargs.get("contents")

    assert len(contents) == 3
    assert isinstance(contents[0], str), "First element should be prompt text"
    assert isinstance(contents[1], PILImage.Image), "Second element should be PIL Image"
    assert isinstance(contents[2], str), "Third element should be ad copy text"


@patch("src.evaluate.composed.log_decision")
def test_content_includes_all_copy_fields(mock_log, test_image, test_ad_copy):
    """Ad copy text passed to Gemini includes headline, primary text, description, CTA."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response())

    evaluator = _make_evaluator(mock_client)
    evaluator.evaluate_composed(test_image, test_ad_copy)

    call_args = mock_client.models.generate_content.call_args
    ad_copy_text = call_args.kwargs["contents"][2]

    assert "Headline:" in ad_copy_text
    assert "Primary text:" in ad_copy_text
    assert "Description:" in ad_copy_text
    assert "CTA:" in ad_copy_text
    assert test_ad_copy.headline in ad_copy_text
    assert test_ad_copy.primary_text in ad_copy_text
    assert test_ad_copy.cta_button in ad_copy_text


# ---------------------------------------------------------------------------
# Tests: cost tracking via token count
# ---------------------------------------------------------------------------


@patch("src.evaluate.composed.log_decision")
def test_token_count_tracked(mock_log, test_image, test_ad_copy):
    """Token count from usage_metadata is returned in the result."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(
        _make_composed_response(), total_tokens=350
    )

    evaluator = _make_evaluator(mock_client)
    result = evaluator.evaluate_composed(test_image, test_ad_copy)

    assert result["token_count"] == 350


@patch("src.evaluate.composed.log_decision")
def test_token_count_zero_when_no_metadata(mock_log, test_image, test_ad_copy):
    """Token count defaults to 0 when usage_metadata is absent."""
    mock_client = MagicMock()
    resp = _mock_response(_make_composed_response())
    resp.usage_metadata = None
    mock_client.models.generate_content.return_value = resp

    evaluator = _make_evaluator(mock_client)
    result = evaluator.evaluate_composed(test_image, test_ad_copy)

    assert result["token_count"] == 0


# ---------------------------------------------------------------------------
# Tests: decision logging
# ---------------------------------------------------------------------------


@patch("src.evaluate.composed.log_decision")
def test_decision_logging_called(mock_log, test_image, test_ad_copy):
    """log_decision is called for start and publishable decision."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(_make_composed_response())

    evaluator = _make_evaluator(mock_client)
    evaluator.evaluate_composed(test_image, test_ad_copy)

    actions = [call.args[1] for call in mock_log.call_args_list]
    assert "evaluate_composed_start" in actions
    assert "publishable_decision" in actions


# ---------------------------------------------------------------------------
# Tests: prompt and schema constants
# ---------------------------------------------------------------------------


def test_prompt_covers_evaluation_criteria():
    """Prompt text covers message coherence, visual-text balance, publishability."""
    prompt_lower = COMPOSED_EVAL_PROMPT.lower()
    assert "message coherence" in prompt_lower
    assert "visual-text balance" in prompt_lower
    assert "publishability" in prompt_lower


def test_schema_requires_rationale_and_score():
    """Schema requires rationale and composed_score fields."""
    assert "rationale" in COMPOSED_EVAL_SCHEMA["required"]
    assert "composed_score" in COMPOSED_EVAL_SCHEMA["required"]


def test_publishable_threshold_is_seven():
    """Publishable threshold is 7.0 as specified."""
    assert PUBLISHABLE_THRESHOLD == 7.0
