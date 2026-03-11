"""Tests for BatchPipeline visual pipeline wiring (story-605).

Covers _run_visual_pipeline() gating, variant generation, composed evaluation,
DB persistence, and run() integration with circuit breaker and no_images flag.

All external APIs are mocked — zero real calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.iterate.controller import IterationController
from src.iterate.healing import SelfHealer
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.creative import ImageResult
from src.models.evaluation import DimensionScore, EvaluationResult
from src.pipeline.main import BatchPipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_BRIEF = AdBrief(
    audience_segment="parent",
    product_offer="SAT Prep 1-on-1",
    campaign_goal="Drive sign-ups",
    tone="authoritative",
    competitive_context="Kaplan",
)

SAMPLE_AD = AdCopy(
    id="ad-001",
    primary_text="Great SAT prep.",
    headline="Boost SAT Scores",
    description="1-on-1 tutoring.",
    cta_button="Get Started",
    model_id="gemini-2.5-flash",
    token_count=100,
)


def _make_image_result(**overrides) -> ImageResult:
    defaults = {
        "file_path": "/data/images/ad-001.png",
        "model_id": "gemini-2.5-flash-image",
        "cost_usd": 0.04,
        "generation_config": {"negative_prompt": "no text overlays"},
        "variant_group_id": "vg-001",
        "variant_type": "lifestyle",
    }
    defaults.update(overrides)
    return ImageResult(**defaults)


def _passing_eval(ad_id: str = "") -> EvaluationResult:
    scores = [
        DimensionScore(dimension="clarity", score=8.0, rationale="clear"),
        DimensionScore(dimension="value_prop", score=7.5, rationale="strong"),
        DimensionScore(dimension="cta_effectiveness", score=7.0, rationale="good"),
        DimensionScore(dimension="brand_voice", score=7.0, rationale="on brand"),
        DimensionScore(dimension="emotional_resonance", score=7.0, rationale="resonant"),
    ]
    return EvaluationResult(
        ad_id=ad_id,
        scores=scores,
        weighted_average=7.4,
        passed_threshold=True,
        evaluator_model="gemini-2.5-pro",
        token_count=50,
    )


def _build_pipeline(db_conn):
    """Build a BatchPipeline with all external dependencies mocked.

    Returns (pipeline, mocks_dict) where mocks_dict contains all mock objects
    keyed by name for easy assertion access.
    """
    mock_gen = MagicMock(name="GenerationEngine")
    mock_eval = MagicMock(name="EvaluationEngine")
    mock_image_engine = MagicMock(name="ImageGenerationEngine")
    mock_composed = MagicMock(name="ComposedEvaluator")
    mock_prompt_gen = MagicMock(name="VisualPromptGenerator")
    mock_variant_gen = MagicMock(name="VariantGenerator")
    mock_circuit_breaker = MagicMock(name="VisualCircuitBreaker")

    # Default: generate and pass iteration on first try
    mock_gen.generate.return_value = SAMPLE_AD
    mock_eval.evaluate_iteration.return_value = _passing_eval()

    mock_circuit_breaker.check_batch_health.return_value = "continue"
    mock_circuit_breaker.check_ad_status.return_value = "has_candidates"

    pipeline = BatchPipeline(
        db_path=":memory:",
        generator=mock_gen,
        evaluator=mock_eval,
        image_engine=mock_image_engine,
        composed_evaluator=mock_composed,
        prompt_generator=mock_prompt_gen,
        variant_generator=mock_variant_gen,
        circuit_breaker=mock_circuit_breaker,
    )
    # Replace controller and conn to use test in-memory DB
    pipeline._controller = IterationController(mock_gen, mock_eval, SelfHealer(), db_conn)
    pipeline._conn = db_conn

    mocks = {
        "gen": mock_gen,
        "eval": mock_eval,
        "image_engine": mock_image_engine,
        "composed": mock_composed,
        "prompt_gen": mock_prompt_gen,
        "variant_gen": mock_variant_gen,
        "circuit_breaker": mock_circuit_breaker,
    }
    return pipeline, mocks


# ---------------------------------------------------------------------------
# _run_visual_pipeline tests
# ---------------------------------------------------------------------------


class TestRunVisualPipeline:
    """Direct tests on _run_visual_pipeline() method."""

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_no_variants_returns_none(self, _heal_log, _ctrl_log, _pipe_log, db_conn):
        """When variant generator returns empty list, result is None."""
        pipeline, mocks = _build_pipeline(db_conn)
        mocks["variant_gen"].generate_variants.return_value = []

        result, cost = pipeline._run_visual_pipeline(SAMPLE_AD, SAMPLE_BRIEF)

        assert result is None
        assert cost == 0.0

    @patch("src.pipeline.main.update_ad_image")
    @patch("src.pipeline.main.PILImage")
    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_not_publishable_returns_none_no_db_write(
        self, _heal_log, _ctrl_log, _pipe_log, mock_pil, mock_update_image, db_conn
    ):
        """When composed eval says not publishable, no DB write occurs."""
        pipeline, mocks = _build_pipeline(db_conn)
        image_result = _make_image_result()
        mocks["variant_gen"].generate_variants.return_value = [image_result]
        mocks["composed"].evaluate_composed.return_value = {
            "composed_score": 4.5,
            "rationale": "Image does not match copy tone.",
            "publishable": False,
            "token_count": 30,
        }

        result, cost = pipeline._run_visual_pipeline(SAMPLE_AD, SAMPLE_BRIEF)

        assert result is None
        mock_update_image.assert_not_called()

    @patch("src.pipeline.main.update_ad_image")
    @patch("src.pipeline.main.PILImage")
    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_happy_path_publishable_persists_and_returns(
        self, _heal_log, _ctrl_log, _pipe_log, mock_pil, mock_update_image, db_conn
    ):
        """Publishable image persists to DB and returns ImageResult."""
        pipeline, mocks = _build_pipeline(db_conn)
        image_result = _make_image_result()
        mocks["variant_gen"].generate_variants.return_value = [image_result]
        mocks["composed"].evaluate_composed.return_value = {
            "composed_score": 8.5,
            "rationale": "Strong visual-copy alignment.",
            "publishable": True,
            "token_count": 40,
        }

        result, cost = pipeline._run_visual_pipeline(SAMPLE_AD, SAMPLE_BRIEF)

        assert result is not None
        assert result.file_path == "/data/images/ad-001.png"
        assert result.cost_usd == 0.04
        assert result.model_id == "gemini-2.5-flash-image"

        mock_update_image.assert_called_once_with(
            db_conn,
            "ad-001",
            image_path="/data/images/ad-001.png",
            visual_prompt="no text overlays",
            image_model="gemini-2.5-flash-image",
            image_cost_usd=0.04,
            variant_group_id="vg-001",
            variant_type="lifestyle",
        )

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_exception_caught_returns_none(self, _heal_log, _ctrl_log, _pipe_log, db_conn):
        """Any exception in the visual pipeline is caught and returns None."""
        pipeline, mocks = _build_pipeline(db_conn)
        mocks["prompt_gen"].generate.side_effect = RuntimeError("API down")

        result, cost = pipeline._run_visual_pipeline(SAMPLE_AD, SAMPLE_BRIEF)

        assert result is None
        # Verify error was logged
        logged_actions = [c.args[1] for c in _pipe_log.call_args_list]
        assert "visual_pipeline_error" in logged_actions


# ---------------------------------------------------------------------------
# run() visual integration tests
# ---------------------------------------------------------------------------


class TestRunVisualIntegration:
    """Tests for visual pipeline wiring inside run()."""

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_no_images_flag_skips_visual(self, _heal_log, _ctrl_log, _pipe_log, db_conn):
        """With no_images=True, visual pipeline is never invoked."""
        pipeline, mocks = _build_pipeline(db_conn)

        result = pipeline.run([SAMPLE_BRIEF], no_images=True)

        assert result.passed == 1
        assert result.images_generated == 0
        mocks["variant_gen"].generate_variants.assert_not_called()

    @patch("src.pipeline.main.update_ad_image")
    @patch("src.pipeline.main.PILImage")
    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_successful_image_updates_batch_result(
        self, _heal_log, _ctrl_log, _pipe_log, mock_pil, mock_update_image, db_conn
    ):
        """Successful image gen increments images_generated and visual_cost_usd."""
        pipeline, mocks = _build_pipeline(db_conn)
        image_result = _make_image_result(cost_usd=0.05)
        mocks["variant_gen"].generate_variants.return_value = [image_result]
        mocks["composed"].evaluate_composed.return_value = {
            "composed_score": 8.0,
            "rationale": "Good.",
            "publishable": True,
            "token_count": 30,
        }

        result = pipeline.run([SAMPLE_BRIEF])

        assert result.images_generated == 1
        assert result.visual_cost_usd == 0.05
        assert result.visual_failures == 0

    @patch("src.pipeline.main.update_ad_image")
    @patch("src.pipeline.main.PILImage")
    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_visual_failure_increments_failures(
        self, _heal_log, _ctrl_log, _pipe_log, mock_pil, mock_update_image, db_conn
    ):
        """When image gen is attempted but fails, visual_failures increments."""
        pipeline, mocks = _build_pipeline(db_conn)
        mocks["variant_gen"].generate_variants.return_value = []

        result = pipeline.run([SAMPLE_BRIEF])

        assert result.passed == 1
        assert result.images_generated == 0
        assert result.visual_failures == 1

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_circuit_breaker_halt_skips_visual(self, _heal_log, _ctrl_log, _pipe_log, db_conn):
        """Circuit breaker 'halt' skips visual pipeline and sets tripped flag."""
        pipeline, mocks = _build_pipeline(db_conn)
        mocks["circuit_breaker"].check_batch_health.return_value = "halt"

        result = pipeline.run([SAMPLE_BRIEF])

        assert result.passed == 1
        assert result.circuit_breaker_tripped is True
        assert result.images_generated == 0
        mocks["variant_gen"].generate_variants.assert_not_called()

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_reset_batch_called_at_start(self, _heal_log, _ctrl_log, _pipe_log, db_conn):
        """reset_batch() is called at the beginning of each run()."""
        pipeline, mocks = _build_pipeline(db_conn)

        pipeline.run([SAMPLE_BRIEF])

        mocks["circuit_breaker"].reset_batch.assert_called_once()

    @patch("src.pipeline.main.update_ad_image")
    @patch("src.pipeline.main.PILImage")
    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_visual_cost_accumulates_across_briefs(
        self, _heal_log, _ctrl_log, _pipe_log, mock_pil, mock_update_image, db_conn
    ):
        """Visual cost sums across multiple successful image generations."""
        pipeline, mocks = _build_pipeline(db_conn)

        # Return unique ads for each brief so iterate works
        ads = [
            AdCopy(
                id=f"ad-{i}",
                primary_text=f"Ad {i}",
                headline=f"Head {i}",
                description=f"Desc {i}",
                cta_button="Learn More",
                model_id="gemini-2.5-flash",
                token_count=80,
            )
            for i in range(10)
        ]
        mocks["gen"].generate.side_effect = ads

        img1 = _make_image_result(cost_usd=0.03, file_path="/data/images/ad-0.png")
        img2 = _make_image_result(cost_usd=0.05, file_path="/data/images/ad-1.png")
        mocks["variant_gen"].generate_variants.side_effect = [[img1], [img2]]
        mocks["composed"].evaluate_composed.return_value = {
            "composed_score": 8.0,
            "rationale": "Good.",
            "publishable": True,
            "token_count": 30,
        }

        result = pipeline.run([SAMPLE_BRIEF, SAMPLE_BRIEF])

        assert result.images_generated == 2
        assert abs(result.visual_cost_usd - 0.08) < 1e-9

    @patch("src.pipeline.main.update_ad_image")
    @patch("src.pipeline.main.PILImage")
    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_circuit_breaker_check_ad_status_called_on_image_result(
        self, _heal_log, _ctrl_log, _pipe_log, mock_pil, mock_update_image, db_conn
    ):
        """Circuit breaker's check_ad_status is called when image result is returned."""
        pipeline, mocks = _build_pipeline(db_conn)
        image_result = _make_image_result()
        mocks["variant_gen"].generate_variants.return_value = [image_result]
        mocks["composed"].evaluate_composed.return_value = {
            "composed_score": 8.0,
            "rationale": "Good.",
            "publishable": True,
            "token_count": 30,
        }

        pipeline.run([SAMPLE_BRIEF])

        mocks["circuit_breaker"].check_ad_status.assert_called_once()
        # The ad id is generated by the iterate loop -- verify the arg is a string
        actual_ad_id = mocks["circuit_breaker"].check_ad_status.call_args[0][0]
        assert isinstance(actual_ad_id, str) and len(actual_ad_id) > 0

