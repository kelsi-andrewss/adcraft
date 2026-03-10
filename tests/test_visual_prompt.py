"""Tests for visual prompt engineering (story-595).

Tests cover:
- Pydantic model validation for VisualBrief, ImageResult, VisualEvaluationResult
- Aspect ratio mapping from placement
- Visual prompt generation with mocked Gemini
- Decision logging at all branch points
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.generate.visual_prompt import (
    PLACEMENT_ASPECT_RATIOS,
    VisualPromptGenerator,
)
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.creative import ImageResult, VisualBrief, VisualEvaluationResult

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def sample_ad() -> AdCopy:
    return AdCopy(
        id="ad-001",
        primary_text=(
            "Your child's SAT score shouldn't be limited by access to great teaching. "
            "Varsity Tutors connects students with expert SAT tutors who've helped "
            "families see an average 160-point score improvement."
        ),
        headline="Average 160-Point SAT Improvement",
        description="Personalized 1-on-1 SAT tutoring from expert instructors.",
        cta_button="Get Started",
        brief_id="brief-001",
        model_id="gemini-2.5-flash",
    )


@pytest.fixture()
def sample_brief() -> AdBrief:
    return AdBrief(
        audience_segment="Parents of high school juniors",
        product_offer="1-on-1 SAT tutoring",
        campaign_goal="Lead generation",
        tone="Supportive and results-oriented",
    )


@pytest.fixture()
def mock_gemini_response() -> MagicMock:
    """Create a mock Gemini response with prompt and negative_prompt."""
    response = MagicMock()
    response.text = json.dumps(
        {
            "prompt": (
                "A confident high school student celebrating with raised fist "
                "after seeing improved SAT score on laptop screen. Warm natural "
                "lighting, blue and green color palette, modern clean study room "
                "with bookshelves. Soft bokeh background. Diverse student, "
                "genuine smile of achievement."
            ),
            "negative_prompt": (
                "No anxiety, stress, frustrated expressions, dark lighting, "
                "cluttered background, text overlays, stock photo poses, "
                "thumbs up, staged smiles, gloomy atmosphere"
            ),
        }
    )
    usage = MagicMock()
    usage.total_token_count = 342
    response.usage_metadata = usage
    return response


# ── Model validation tests ────────────────────────────────────────────


class TestVisualBrief:
    def test_defaults_applied(self):
        brief = VisualBrief(prompt="test prompt", negative_prompt="test negative")
        assert brief.aspect_ratio == "1:1"
        assert brief.resolution == "1K"
        assert brief.style_refs == []
        assert brief.placement == "feed"

    def test_required_fields_enforced(self):
        with pytest.raises(Exception):
            VisualBrief()  # type: ignore[call-arg]

    def test_prompt_required(self):
        with pytest.raises(Exception):
            VisualBrief(negative_prompt="no bad stuff")  # type: ignore[call-arg]

    def test_negative_prompt_required(self):
        with pytest.raises(Exception):
            VisualBrief(prompt="good stuff")  # type: ignore[call-arg]

    def test_custom_values(self):
        brief = VisualBrief(
            prompt="custom prompt",
            negative_prompt="custom negative",
            aspect_ratio="9:16",
            resolution="2K",
            style_refs=["/path/to/ref.png"],
            placement="stories",
        )
        assert brief.aspect_ratio == "9:16"
        assert brief.resolution == "2K"
        assert brief.style_refs == ["/path/to/ref.png"]
        assert brief.placement == "stories"


class TestImageResult:
    def test_bytes_accepts_none(self):
        result = ImageResult(
            image_bytes=None,
            file_path="/data/images/ad-001.png",
            model_id="gemini-2.5-flash-image",
            cost_usd=0.0,
        )
        assert result.image_bytes is None

    def test_bytes_accepts_bytes(self):
        result = ImageResult(
            image_bytes=b"\x89PNG\r\n",
            file_path="/data/images/ad-001.png",
            model_id="gemini-2.5-flash-image",
            cost_usd=0.0,
        )
        assert result.image_bytes == b"\x89PNG\r\n"

    def test_required_fields_enforced(self):
        with pytest.raises(Exception):
            ImageResult()  # type: ignore[call-arg]

    def test_file_path_required(self):
        with pytest.raises(Exception):
            ImageResult(model_id="test", cost_usd=0.0)  # type: ignore[call-arg]

    def test_model_id_required(self):
        with pytest.raises(Exception):
            ImageResult(file_path="/test.png", cost_usd=0.0)  # type: ignore[call-arg]

    def test_cost_usd_required(self):
        with pytest.raises(Exception):
            ImageResult(file_path="/test.png", model_id="test")  # type: ignore[call-arg]

    def test_default_generation_config(self):
        result = ImageResult(
            file_path="/data/images/ad-001.png",
            model_id="gemini-2.5-flash-image",
            cost_usd=0.04,
        )
        assert result.generation_config == {}


class TestVisualEvaluationResult:
    def test_all_scores_required(self):
        with pytest.raises(Exception):
            VisualEvaluationResult()  # type: ignore[call-arg]

    def test_valid_result(self):
        result = VisualEvaluationResult(
            brand_consistency_score=8.5,
            composition_score=7.0,
            synergy_score=9.0,
            rationales={
                "brand_consistency": "Strong blue/green palette match",
                "composition": "Clean layout but focal point slightly off-center",
                "synergy": "Image perfectly reinforces the achievement message",
            },
            overall_visual_score=8.2,
        )
        assert result.brand_consistency_score == 8.5
        assert result.composition_score == 7.0
        assert result.synergy_score == 9.0
        assert len(result.rationales) == 3
        assert result.overall_visual_score == 8.2

    def test_rationales_is_dict_str_str(self):
        result = VisualEvaluationResult(
            brand_consistency_score=8.0,
            composition_score=7.0,
            synergy_score=8.0,
            rationales={"key": "value"},
            overall_visual_score=7.5,
        )
        assert isinstance(result.rationales, dict)
        for k, v in result.rationales.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ── Aspect ratio mapping tests ────────────────────────────────────────


class TestAspectRatioMapping:
    def test_feed_maps_to_square(self):
        assert PLACEMENT_ASPECT_RATIOS["feed"] == "1:1"

    def test_stories_maps_to_portrait(self):
        assert PLACEMENT_ASPECT_RATIOS["stories"] == "9:16"

    def test_banner_maps_to_landscape(self):
        assert PLACEMENT_ASPECT_RATIOS["banner"] == "16:9"

    def test_unknown_placement_defaults_to_square(self):
        assert PLACEMENT_ASPECT_RATIOS.get("reel", "1:1") == "1:1"
        assert PLACEMENT_ASPECT_RATIOS.get("carousel", "1:1") == "1:1"


# ── Prompt generator integration test (mocked Gemini) ─────────────────


class TestVisualPromptGenerator:
    @patch("src.generate.visual_prompt.log_decision")
    def test_generate_returns_valid_visual_brief(
        self,
        mock_log,
        sample_ad,
        sample_brief,
        mock_gemini_response,
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_gemini_response

        generator = VisualPromptGenerator(client=mock_client)
        result = generator.generate(sample_ad, sample_brief, placement="feed")

        assert isinstance(result, VisualBrief)
        assert len(result.prompt) > 0
        assert len(result.negative_prompt) > 0
        assert result.aspect_ratio == "1:1"
        assert result.resolution == "1K"
        assert result.placement == "feed"

    @patch("src.generate.visual_prompt.log_decision")
    def test_generate_stories_placement(
        self,
        mock_log,
        sample_ad,
        sample_brief,
        mock_gemini_response,
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_gemini_response

        generator = VisualPromptGenerator(client=mock_client)
        result = generator.generate(sample_ad, sample_brief, placement="stories")

        assert result.aspect_ratio == "9:16"
        assert result.placement == "stories"

    @patch("src.generate.visual_prompt.log_decision")
    def test_generate_banner_placement(
        self,
        mock_log,
        sample_ad,
        sample_brief,
        mock_gemini_response,
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_gemini_response

        generator = VisualPromptGenerator(client=mock_client)
        result = generator.generate(sample_ad, sample_brief, placement="banner")

        assert result.aspect_ratio == "16:9"
        assert result.placement == "banner"

    @patch("src.generate.visual_prompt.log_decision")
    def test_generate_unknown_placement_defaults(
        self,
        mock_log,
        sample_ad,
        sample_brief,
        mock_gemini_response,
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_gemini_response

        generator = VisualPromptGenerator(client=mock_client)
        result = generator.generate(sample_ad, sample_brief, placement="reel")

        assert result.aspect_ratio == "1:1"

    @patch("src.generate.visual_prompt.log_decision")
    def test_gemini_called_with_correct_model(
        self,
        mock_log,
        sample_ad,
        sample_brief,
        mock_gemini_response,
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_gemini_response

        generator = VisualPromptGenerator(client=mock_client)
        generator.generate(sample_ad, sample_brief)

        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-2.5-flash"


# ── Decision logging tests ────────────────────────────────────────────


class TestDecisionLogging:
    @patch("src.generate.visual_prompt.log_decision")
    def test_logs_at_all_branch_points(
        self,
        mock_log,
        sample_ad,
        sample_brief,
        mock_gemini_response,
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_gemini_response

        generator = VisualPromptGenerator(client=mock_client)
        generator.generate(sample_ad, sample_brief)

        logged_actions = [call.args[1] for call in mock_log.call_args_list]
        assert "engine_init" in logged_actions
        assert "generation_start" in logged_actions
        assert "aspect_ratio_selection" in logged_actions
        assert "generation_complete" in logged_actions

    @patch("src.generate.visual_prompt.log_decision")
    def test_init_logs_model(self, mock_log):
        mock_client = MagicMock()
        VisualPromptGenerator(client=mock_client)

        init_call = mock_log.call_args_list[0]
        assert init_call.args[0] == "visual_prompt"
        assert init_call.args[1] == "engine_init"
        assert "gemini-2.5-flash" in init_call.args[2]
