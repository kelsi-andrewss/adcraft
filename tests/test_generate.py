"""Tests for the generation engine.

All Gemini API calls are mocked — no API keys required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.generate.engine import GENERATION_SCHEMA, GenerationEngine
from src.models.ad import AdCopy
from src.models.brief import AdBrief

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_brief(**kwargs) -> AdBrief:
    defaults = {
        "audience_segment": "parents of high school juniors",
        "product_offer": "1-on-1 SAT tutoring with expert instructors",
        "campaign_goal": "lead generation",
        "tone": "supportive, results-oriented",
        "competitive_context": (
            "Princeton Review emphasizes classroom format; we differentiate with 1-on-1."
        ),
    }
    defaults.update(kwargs)
    return AdBrief(**defaults)


def _make_generation_response(
    primary_text: str = "Great SAT prep starts here.",
    headline: str = "Boost Your SAT Score",
    description: str = "Expert 1-on-1 tutoring for real results.",
    cta_button: str = "Get Started",
) -> dict:
    return {
        "primary_text": primary_text,
        "headline": headline,
        "description": description,
        "cta_button": cta_button,
    }


def _mock_response(data: dict, total_tokens: int = 250) -> MagicMock:
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.total_token_count = total_tokens
    return resp


def _make_engine(mock_client: MagicMock | None = None) -> GenerationEngine:
    client = mock_client or MagicMock()
    with patch("src.generate.engine.log_decision"):
        return GenerationEngine(client=client)


# ---------------------------------------------------------------------------
# Tests: successful generation
# ---------------------------------------------------------------------------


@patch("src.generate.engine.log_decision")
def test_generate_returns_valid_adcopy(mock_log):
    """Generate returns a valid AdCopy instance."""
    mock_client = MagicMock()
    response_data = _make_generation_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.generate(_make_brief())

    assert isinstance(result, AdCopy)
    assert result.primary_text == "Great SAT prep starts here."
    assert result.headline == "Boost Your SAT Score"
    assert result.description == "Expert 1-on-1 tutoring for real results."
    assert result.cta_button == "Get Started"


# ---------------------------------------------------------------------------
# Tests: structured output schema
# ---------------------------------------------------------------------------


@patch("src.generate.engine.log_decision")
def test_structured_output_schema_passed(mock_log):
    """response_json_schema is passed to the API call."""
    mock_client = MagicMock()
    response_data = _make_generation_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    engine.generate(_make_brief())

    call_kwargs = mock_client.models.generate_content.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == GENERATION_SCHEMA


# ---------------------------------------------------------------------------
# Tests: safety filter retry
# ---------------------------------------------------------------------------


@patch("src.generate.engine.log_decision")
def test_safety_filter_retry(mock_log):
    """Safety filter block triggers retry, then succeeds."""
    mock_client = MagicMock()
    good_response = _mock_response(_make_generation_response())
    mock_client.models.generate_content.side_effect = [
        Exception("Safety filter blocked"),
        good_response,
    ]

    engine = _make_engine(mock_client)
    result = engine.generate(_make_brief())

    assert isinstance(result, AdCopy)
    assert mock_client.models.generate_content.call_count == 2


@patch("src.generate.engine.log_decision")
def test_safety_filter_max_retries_raises(mock_log):
    """3 consecutive safety blocks raises exception."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        Exception("Safety filter blocked"),
        Exception("Safety filter blocked"),
        Exception("Safety filter blocked"),
    ]

    engine = _make_engine(mock_client)
    with pytest.raises(Exception, match="Safety filter blocked"):
        engine.generate(_make_brief())

    assert mock_client.models.generate_content.call_count == 3


# ---------------------------------------------------------------------------
# Tests: decision logging
# ---------------------------------------------------------------------------


@patch("src.generate.engine.log_decision")
def test_log_decision_called_on_generate(mock_log):
    """log_decision called with component='generator' during generation."""
    mock_client = MagicMock()
    response_data = _make_generation_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    engine.generate(_make_brief())

    # Should have: engine_init + generation_start + generation_complete = 3
    # (engine_init is patched out during _make_engine, so we get generation_start + complete)
    assert mock_log.call_count >= 2

    components = [c.args[0] for c in mock_log.call_args_list]
    assert all(c == "generator" for c in components)


# ---------------------------------------------------------------------------
# Tests: token count
# ---------------------------------------------------------------------------


@patch("src.generate.engine.log_decision")
def test_token_count_captured(mock_log):
    """Token count from usage_metadata flows into AdCopy."""
    mock_client = MagicMock()
    response_data = _make_generation_response()
    mock_client.models.generate_content.return_value = _mock_response(
        response_data, total_tokens=350
    )

    engine = _make_engine(mock_client)
    result = engine.generate(_make_brief())

    assert result.token_count == 350


# ---------------------------------------------------------------------------
# Tests: prompt content
# ---------------------------------------------------------------------------


@patch("src.generate.engine.log_decision")
def test_prompt_includes_brief_details(mock_log):
    """Brief audience, offer, and tone appear in the constructed prompt."""
    mock_client = MagicMock()
    response_data = _make_generation_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    brief = _make_brief(
        audience_segment="high school seniors",
        product_offer="intensive SAT boot camp",
        tone="energetic, motivating",
    )
    engine.generate(brief)

    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get("contents") or call_args[1].get("contents")
    assert "high school seniors" in prompt
    assert "intensive SAT boot camp" in prompt
    assert "energetic, motivating" in prompt


@patch("src.generate.engine.log_decision")
def test_prompt_includes_brand_voice_examples(mock_log):
    """Few-shot brand voice examples are present in the prompt."""
    mock_client = MagicMock()
    response_data = _make_generation_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    engine.generate(_make_brief())

    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get("contents") or call_args[1].get("contents")
    # Should contain few-shot examples
    assert "160-Point SAT Score Improvement" in prompt
    assert "Break Through Your SAT Score Ceiling" in prompt
    # Should contain brand voice guidelines
    assert "Supportive and encouraging" in prompt


# ---------------------------------------------------------------------------
# Tests: model ID
# ---------------------------------------------------------------------------


@patch("src.generate.engine.log_decision")
def test_uses_gemini_flash_model(mock_log):
    """Generator uses gemini-2.5-flash, not gemini-2.5-pro."""
    mock_client = MagicMock()
    response_data = _make_generation_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    result = engine.generate(_make_brief())

    assert result.model_id == "gemini-2.5-flash"
    call_args = mock_client.models.generate_content.call_args
    model_arg = call_args.kwargs.get("model") or call_args[1].get("model")
    assert model_arg == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Tests: competitive context in prompt
# ---------------------------------------------------------------------------


@patch("src.generate.engine.log_decision")
def test_prompt_includes_competitive_context(mock_log):
    """Competitive context from brief appears in the prompt when provided."""
    mock_client = MagicMock()
    response_data = _make_generation_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    brief = _make_brief(
        competitive_context="Kaplan focuses on classroom; we differentiate with 1-on-1 tutoring."
    )
    engine.generate(brief)

    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get("contents") or call_args[1].get("contents")
    assert "Kaplan focuses on classroom" in prompt


@patch("src.generate.engine.log_decision")
def test_prompt_omits_competitive_context_when_empty(mock_log):
    """No competitive context section when brief.competitive_context is empty."""
    mock_client = MagicMock()
    response_data = _make_generation_response()
    mock_client.models.generate_content.return_value = _mock_response(response_data)

    engine = _make_engine(mock_client)
    brief = _make_brief(competitive_context="")
    engine.generate(brief)

    call_args = mock_client.models.generate_content.call_args
    prompt = call_args.kwargs.get("contents") or call_args[1].get("contents")
    assert "COMPETITIVE CONTEXT:" not in prompt
