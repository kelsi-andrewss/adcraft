"""Tests for visual evaluation dimensions.

All Gemini API calls are mocked — no API keys required.
Uses a 1x1 red PIL.Image as the test image (no real image file needed).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from src.evaluate.engine import EvaluationEngine
from src.evaluate.visual_rubrics import (
    VISUAL_DIMENSION_WEIGHTS,
    VISUAL_DIMENSIONS,
    build_visual_all_dimensions_prompt,
    build_visual_single_dimension_prompt,
)
from src.models.ad import AdCopy
from src.theme import THEME

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
        id="visual-test-001",
        primary_text="Unlock your child's potential with expert SAT tutoring.",
        headline="Score Higher with Nerdy",
        description="Personalized 1-on-1 SAT prep from expert tutors.",
        cta_button="Book a Free Session",
    )


def _make_engine(mock_client: MagicMock | None = None) -> EvaluationEngine:
    client = mock_client or MagicMock()
    return EvaluationEngine(client=client)


def _mock_response(data: dict, total_tokens: int = 150) -> MagicMock:
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.total_token_count = total_tokens
    return resp


def _make_visual_all_dimensions_response(
    scores: dict[str, float] | None = None,
) -> dict:
    default_scores = {
        "brand_consistency": 7.5,
        "composition_quality": 6.0,
        "text_image_synergy": 8.0,
        "instructional_clarity": 7.0,
    }
    if scores:
        default_scores.update(scores)
    return {
        dim: {
            "rationale": f"Detailed visual analysis of {dim}.",
            "score": default_scores[dim],
            "confidence": 0.85,
        }
        for dim in VISUAL_DIMENSIONS
    }


def _make_visual_single_dimension_response(dim: str, score: float) -> dict:
    return {
        "rationale": f"Focused analysis of {dim} for this image.",
        "score": score,
        "confidence": 0.9,
    }


# ---------------------------------------------------------------------------
# Tests: rubric prompt construction
# ---------------------------------------------------------------------------


def test_build_visual_single_dimension_prompt_contains_rubric():
    """Each single-dimension builder includes the dimension name and key rubric phrases."""
    for dim in VISUAL_DIMENSIONS:
        prompt = build_visual_single_dimension_prompt(dim)
        assert dim in prompt
        # Each rubric should contain its scoring bands
        assert "1-3" in prompt
        assert "9-10" in prompt
        # CoT instruction present
        assert "describe what you observe" in prompt.lower()
        # Anti-inflation present
        assert "Do NOT inflate scores" in prompt


def test_build_visual_all_dimensions_prompt_contains_all_rubrics():
    """All-dimensions builder includes all 4 dimension names."""
    prompt = build_visual_all_dimensions_prompt()
    for dim in VISUAL_DIMENSIONS:
        assert dim.upper() in prompt
    assert "Do NOT inflate scores" in prompt


def test_visual_dimension_weights_sum_to_one():
    """Visual dimension weights must sum to 1.0."""
    assert sum(VISUAL_DIMENSION_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: multimodal Gemini response (mocked)
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_evaluate_visual_iteration_mode(mock_log, test_image, test_ad_copy):
    """Iteration mode returns 4 DimensionScore objects with correct dimension names."""
    mock_client = MagicMock()
    response_data = _make_visual_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    scores = engine.evaluate_visual(test_image, test_ad_copy, eval_mode="iteration")

    assert len(scores) == 4
    dims_returned = {s.dimension for s in scores}
    assert dims_returned == set(VISUAL_DIMENSIONS)


@patch("src.evaluate.engine.log_decision")
def test_evaluate_visual_final_mode(mock_log, test_image, test_ad_copy):
    """Final mode makes 4 separate API calls, one per visual dimension."""
    mock_client = MagicMock()
    responses = [
        _mock_response(_make_visual_single_dimension_response(dim, 7.0))
        for dim in VISUAL_DIMENSIONS
    ]
    mock_client.models.generate_content.side_effect = responses

    engine = _make_engine(mock_client)
    scores = engine.evaluate_visual(test_image, test_ad_copy, eval_mode="final")

    assert len(scores) == 4
    assert mock_client.models.generate_content.call_count == 4
    dims_returned = {s.dimension for s in scores}
    assert dims_returned == set(VISUAL_DIMENSIONS)


@patch("src.evaluate.engine.log_decision")
def test_evaluate_visual_iteration_single_api_call(mock_log, test_image, test_ad_copy):
    """Iteration mode uses exactly 1 API call for all 4 dimensions."""
    mock_client = MagicMock()
    response_data = _make_visual_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    engine.evaluate_visual(test_image, test_ad_copy, eval_mode="iteration")

    assert mock_client.models.generate_content.call_count == 1


# ---------------------------------------------------------------------------
# Tests: DimensionScore output format
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_visual_dimension_scores_have_rationale(mock_log, test_image, test_ad_copy):
    """Each returned DimensionScore has a non-empty rationale string."""
    mock_client = MagicMock()
    response_data = _make_visual_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    scores = engine.evaluate_visual(test_image, test_ad_copy, eval_mode="iteration")

    for s in scores:
        assert isinstance(s.rationale, str)
        assert len(s.rationale) > 0


@patch("src.evaluate.engine.log_decision")
def test_visual_dimension_scores_range(mock_log, test_image, test_ad_copy):
    """Each score is between 1.0 and 10.0."""
    mock_client = MagicMock()
    response_data = _make_visual_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    scores = engine.evaluate_visual(test_image, test_ad_copy, eval_mode="iteration")

    for s in scores:
        assert 1.0 <= s.score <= 10.0


@patch("src.evaluate.engine.log_decision")
def test_visual_dimension_names_match_constants(mock_log, test_image, test_ad_copy):
    """Dimension names in returned scores match VISUAL_DIMENSIONS."""
    mock_client = MagicMock()
    response_data = _make_visual_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    scores = engine.evaluate_visual(test_image, test_ad_copy, eval_mode="iteration")

    returned_dims = [s.dimension for s in scores]
    assert returned_dims == VISUAL_DIMENSIONS


# ---------------------------------------------------------------------------
# Tests: content assembly
# ---------------------------------------------------------------------------


@patch("src.evaluate.engine.log_decision")
def test_multimodal_content_list_structure(mock_log, test_image, test_ad_copy):
    """Content list passed to Gemini is [str, PILImage.Image, str]."""
    mock_client = MagicMock()
    response_data = _make_visual_all_dimensions_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    engine.evaluate_visual(test_image, test_ad_copy, eval_mode="iteration")

    call_args = mock_client.models.generate_content.call_args
    contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
    # Contents may also be positional — check kwargs first, then positional
    if contents is None:
        # generate_content(model=..., contents=...) — contents is second positional arg
        contents = call_args[1] if len(call_args[0]) <= 1 else call_args[0][1]

    assert len(contents) == 3
    assert isinstance(contents[0], str), "First element should be prompt text"
    assert isinstance(contents[1], PILImage.Image), "Second element should be PIL Image"
    assert isinstance(contents[2], str), "Third element should be ad copy text"

    # Verify ad copy text contains expected fields
    assert "Headline:" in contents[2]
    assert "Primary text:" in contents[2]
    assert "CTA:" in contents[2]


# ---------------------------------------------------------------------------
# Tests: brand theme assertions
# ---------------------------------------------------------------------------


def test_brand_consistency_rubric_contains_theme_brand():
    """brand_consistency rubric references THEME.brand_name, not Varsity Tutors."""
    from src.evaluate.visual_rubrics import VISUAL_RUBRICS

    rubric = VISUAL_RUBRICS["brand_consistency"]
    assert THEME.brand_name in rubric
    assert "Varsity Tutors" not in rubric
    assert THEME.primary_color in rubric
    assert "#003057" not in rubric
    assert "#00B4D8" not in rubric


def test_single_dimension_prompt_contains_theme_brand():
    """Single-dimension prompt references THEME.brand_name."""
    prompt = build_visual_single_dimension_prompt("brand_consistency")
    assert THEME.brand_name in prompt
    assert "Varsity Tutors" not in prompt


def test_all_dimensions_prompt_contains_theme_brand():
    """All-dimensions prompt references THEME.brand_name."""
    prompt = build_visual_all_dimensions_prompt()
    assert THEME.brand_name in prompt
    assert "Varsity Tutors" not in prompt


def test_few_shot_examples_reference_nerdy_palette():
    """Few-shot brand_consistency examples reference Nerdy palette."""
    from src.evaluate.visual_rubrics import VISUAL_FEW_SHOT_EXAMPLES

    examples = VISUAL_FEW_SHOT_EXAMPLES["brand_consistency"]
    assert THEME.brand_name in examples
    assert "Varsity Tutors" not in examples
    assert "cyan" in examples.lower()
    assert "dark navy" in examples.lower()
