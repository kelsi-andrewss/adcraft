"""Tests for BatchPipeline with mocked API calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.db.queries import list_quality_snapshots
from src.iterate.controller import IterationController
from src.iterate.healing import SelfHealer
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.evaluation import DimensionScore, EvaluationResult
from src.pipeline.main import BatchPipeline, RateLimiter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BRIEF = AdBrief(
    audience_segment="parent",
    product_offer="SAT Prep 1-on-1",
    campaign_goal="Drive sign-ups",
    tone="authoritative",
    competitive_context="Kaplan",
)

SAMPLE_AD = AdCopy(
    id="ad-1",
    primary_text="Great SAT prep.",
    headline="Boost SAT Scores",
    description="1-on-1 tutoring.",
    cta_button="Get Started",
    model_id="gemini-2.5-flash",
    token_count=100,
)


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


def _failing_eval(ad_id: str = "") -> EvaluationResult:
    scores = [
        DimensionScore(dimension="clarity", score=4.0, rationale="unclear"),
        DimensionScore(dimension="value_prop", score=4.0, rationale="weak"),
        DimensionScore(dimension="cta_effectiveness", score=4.0, rationale="poor"),
        DimensionScore(dimension="brand_voice", score=4.0, rationale="off"),
        DimensionScore(dimension="emotional_resonance", score=4.0, rationale="flat"),
    ]
    return EvaluationResult(
        ad_id=ad_id,
        scores=scores,
        weighted_average=4.0,
        passed_threshold=False,
        evaluator_model="gemini-2.5-pro",
        token_count=50,
    )


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestBatchPipeline:
    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_full_pipeline_loop(
        self, mock_heal_log, mock_ctrl_log, mock_pipe_log, db_conn
    ):
        """Single brief -> gen -> eval -> pass (happy path)."""
        mock_gen = MagicMock()
        mock_eval = MagicMock()
        mock_gen.generate.return_value = SAMPLE_AD
        mock_eval.evaluate_iteration.return_value = _passing_eval()

        pipeline = BatchPipeline(
            db_path=":memory:", generator=mock_gen, evaluator=mock_eval
        )
        # Replace the controller's conn with our test db_conn
        pipeline._controller = IterationController(
            mock_gen, mock_eval, SelfHealer(), db_conn
        )
        pipeline._conn = db_conn

        result = pipeline.run([SAMPLE_BRIEF])

        assert result.total_briefs == 1
        assert result.passed == 1
        assert result.failed == 0

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_pipeline_with_iteration(
        self, mock_heal_log, mock_ctrl_log, mock_pipe_log, db_conn
    ):
        """Brief that needs 2 cycles to pass."""
        mock_gen = MagicMock()
        mock_eval = MagicMock()

        improved_ad = AdCopy(
            id="ad-2",
            primary_text="Improved SAT prep.",
            headline="Better Headline",
            description="Better desc.",
            cta_button="Get Started",
            model_id="gemini-2.5-flash",
            token_count=90,
        )

        mock_gen.generate.side_effect = [SAMPLE_AD, improved_ad]
        # First eval fails, coherence check after fix passes
        mock_eval.evaluate_iteration.side_effect = [
            _failing_eval(),  # initial eval
            _passing_eval(),  # coherence check passes
            _passing_eval(),  # pipeline's final eval call
        ]

        pipeline = BatchPipeline(
            db_path=":memory:", generator=mock_gen, evaluator=mock_eval
        )
        pipeline._controller = IterationController(
            mock_gen, mock_eval, SelfHealer(), db_conn
        )
        pipeline._conn = db_conn

        result = pipeline.run([SAMPLE_BRIEF])

        assert result.passed == 1
        assert result.total_cycles >= 2

    @patch("src.pipeline.main.log_decision")
    def test_rate_limiter_throttles(self, mock_log):
        """Verify rate limiter inserts delays when RPM is exceeded."""
        limiter = RateLimiter(rpm_limit=3, rpd_limit=100)

        # First 3 should go through immediately
        for _ in range(3):
            limiter.acquire()

        # 4th should trigger a throttle (sleep).
        # We patch time.sleep to avoid actual delays.
        with patch("src.pipeline.main.time.sleep") as mock_sleep:
            limiter.acquire()
            mock_sleep.assert_called_once()

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_force_fail_doesnt_crash_batch(
        self, mock_heal_log, mock_ctrl_log, mock_pipe_log, db_conn
    ):
        """One brief force-fails, next brief still runs."""
        mock_gen = MagicMock()
        mock_eval = MagicMock()

        failing_eval = _failing_eval()
        passing_eval = _passing_eval()

        # Generate enough ads for the force-fail path plus the passing path
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
            for i in range(20)
        ]
        mock_gen.generate.side_effect = ads

        # First brief: all evals fail. Second brief: first eval passes.
        eval_sequence = [failing_eval] * 10 + [passing_eval] * 5
        mock_eval.evaluate_iteration.side_effect = eval_sequence

        pipeline = BatchPipeline(
            db_path=":memory:", generator=mock_gen, evaluator=mock_eval
        )
        pipeline._controller = IterationController(
            mock_gen, mock_eval, SelfHealer(), db_conn
        )
        pipeline._conn = db_conn

        result = pipeline.run([SAMPLE_BRIEF, SAMPLE_BRIEF])

        assert result.total_briefs == 2
        # At least one should have failed and one passed
        assert result.failed >= 1
        # The batch completed without crashing
        assert result.passed + result.failed == result.total_briefs

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_quality_snapshot_persisted(
        self, mock_heal_log, mock_ctrl_log, mock_pipe_log, db_conn
    ):
        """After batch, query DB for quality_snapshot row."""
        mock_gen = MagicMock()
        mock_eval = MagicMock()
        mock_gen.generate.return_value = SAMPLE_AD
        mock_eval.evaluate_iteration.return_value = _passing_eval()

        pipeline = BatchPipeline(
            db_path=":memory:", generator=mock_gen, evaluator=mock_eval
        )
        pipeline._controller = IterationController(
            mock_gen, mock_eval, SelfHealer(), db_conn
        )
        pipeline._conn = db_conn

        pipeline.run([SAMPLE_BRIEF])

        snapshots = list_quality_snapshots(db_conn)
        assert len(snapshots) >= 1
        assert snapshots[0]["total_ads"] == 1
        assert snapshots[0]["ads_above_threshold"] == 1

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_batch_result_accuracy(
        self, mock_heal_log, mock_ctrl_log, mock_pipe_log, db_conn
    ):
        """Verify passed/failed/total counts match actual outcomes."""
        mock_gen = MagicMock()
        mock_eval = MagicMock()

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
            for i in range(20)
        ]
        mock_gen.generate.side_effect = ads
        mock_eval.evaluate_iteration.return_value = _passing_eval()

        pipeline = BatchPipeline(
            db_path=":memory:", generator=mock_gen, evaluator=mock_eval
        )
        pipeline._controller = IterationController(
            mock_gen, mock_eval, SelfHealer(), db_conn
        )
        pipeline._conn = db_conn

        briefs = [SAMPLE_BRIEF, SAMPLE_BRIEF, SAMPLE_BRIEF]
        result = pipeline.run(briefs)

        assert result.total_briefs == 3
        assert result.passed + result.failed == result.total_briefs
        assert result.duration_seconds > 0

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_decision_log_coverage(
        self, mock_heal_log, mock_ctrl_log, mock_pipe_log, db_conn
    ):
        """Verify log_decision called for batch_start and batch_complete."""
        mock_gen = MagicMock()
        mock_eval = MagicMock()
        mock_gen.generate.return_value = SAMPLE_AD
        mock_eval.evaluate_iteration.return_value = _passing_eval()

        pipeline = BatchPipeline(
            db_path=":memory:", generator=mock_gen, evaluator=mock_eval
        )
        pipeline._controller = IterationController(
            mock_gen, mock_eval, SelfHealer(), db_conn
        )
        pipeline._conn = db_conn

        pipeline.run([SAMPLE_BRIEF])

        pipe_actions = [
            call.args[1] for call in mock_pipe_log.call_args_list
        ]

        assert "batch_start" in pipe_actions
        assert "batch_complete" in pipe_actions

    @patch("src.pipeline.main.log_decision")
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_brief_error_handled(
        self, mock_heal_log, mock_ctrl_log, mock_pipe_log, db_conn
    ):
        """Inject exception in one brief, verify pipeline continues and logs the error."""
        mock_gen = MagicMock()
        mock_eval = MagicMock()

        # First call raises, second succeeds
        mock_gen.generate.side_effect = [
            RuntimeError("API exploded"),
            SAMPLE_AD,
        ]
        mock_eval.evaluate_iteration.return_value = _passing_eval()

        pipeline = BatchPipeline(
            db_path=":memory:", generator=mock_gen, evaluator=mock_eval
        )
        pipeline._controller = IterationController(
            mock_gen, mock_eval, SelfHealer(), db_conn
        )
        pipeline._conn = db_conn

        result = pipeline.run([SAMPLE_BRIEF, SAMPLE_BRIEF])

        assert result.total_briefs == 2
        assert result.errors >= 1
        # Pipeline didn't crash — batch_complete should be logged
        pipe_actions = [
            call.args[1] for call in mock_pipe_log.call_args_list
        ]
        assert "brief_error" in pipe_actions
        assert "batch_complete" in pipe_actions
