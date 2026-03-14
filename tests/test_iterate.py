"""Tests for IterationController and SelfHealer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.iterate.controller import IterationController
from src.iterate.healing import INTERVENTION_STRATEGIES, InterventionPlan, SelfHealer
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.evaluation import DimensionScore, EvaluationResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BRIEF = AdBrief(
    audience_segment="parent",
    product_offer="SAT Prep 1-on-1",
    campaign_goal="Drive sign-ups",
    tone="authoritative",
    competitive_context="Kaplan, Princeton Review",
)

PASSING_AD = AdCopy(
    id="ad-pass",
    primary_text="Great SAT prep for your child.",
    headline="Boost SAT Scores",
    description="1-on-1 tutoring that works.",
    cta_button="Get Started",
    model_id="gemini-2.5-flash",
    token_count=100,
)

FAILING_AD = AdCopy(
    id="ad-fail",
    primary_text="Tutoring available.",
    headline="SAT Help",
    description="Sign up today.",
    cta_button="Learn More",
    model_id="gemini-2.5-flash",
    token_count=80,
)


def _make_eval(
    ad_id: str,
    *,
    clarity: float = 7.0,
    value_prop: float = 7.0,
    cta_effectiveness: float = 7.0,
    brand_voice: float = 7.0,
    emotional_resonance: float = 7.0,
    passed: bool = True,
) -> EvaluationResult:
    """Build an EvaluationResult with controllable per-dimension scores."""
    scores = [
        DimensionScore(dimension="clarity", score=clarity, rationale="ok"),
        DimensionScore(dimension="value_prop", score=value_prop, rationale="ok"),
        DimensionScore(dimension="cta_effectiveness", score=cta_effectiveness, rationale="ok"),
        DimensionScore(dimension="brand_voice", score=brand_voice, rationale="ok"),
        DimensionScore(dimension="emotional_resonance", score=emotional_resonance, rationale="ok"),
    ]
    # Compute actual weighted average
    weights = {
        "clarity": 0.25,
        "value_prop": 0.25,
        "cta_effectiveness": 0.20,
        "brand_voice": 0.15,
        "emotional_resonance": 0.15,
    }
    w_avg = sum(s.score * weights[s.dimension] for s in scores)
    return EvaluationResult(
        ad_id=ad_id,
        scores=scores,
        weighted_average=round(w_avg, 4),
        passed_threshold=passed,
        evaluator_model="gemini-2.5-pro",
        token_count=50,
    )


def _build_controller(
    db_conn,
    gen_side_effect=None,
    eval_side_effect=None,
) -> tuple[IterationController, MagicMock, MagicMock]:
    """Build a controller with mocked generator and evaluator."""
    mock_gen = MagicMock()
    mock_eval = MagicMock()
    healer = SelfHealer()

    if gen_side_effect is not None:
        mock_gen.generate.side_effect = gen_side_effect
    if eval_side_effect is not None:
        mock_eval.evaluate_iteration.side_effect = eval_side_effect

    controller = IterationController(mock_gen, mock_eval, healer, db_conn)
    return controller, mock_gen, mock_eval


# ---------------------------------------------------------------------------
# SelfHealer tests
# ---------------------------------------------------------------------------


class TestSelfHealer:
    @patch("src.iterate.healing.log_decision")
    def test_quality_ratchet_only_goes_up(self, mock_log):
        healer = SelfHealer()
        healer.update_ratchet(5.0)
        assert healer.running_avg == 5.0
        healer.update_ratchet(7.0)
        assert healer.running_avg == 7.0
        healer.update_ratchet(6.0)
        assert healer.running_avg == 7.0  # ratchet held
        healer.update_ratchet(7.0)
        assert healer.running_avg == 7.0  # same value, no increase

    @patch("src.iterate.healing.log_decision")
    def test_regression_detected(self, mock_log):
        healer = SelfHealer()
        healer.update_ratchet(7.0)
        assert healer.detect_regression(6.5) is True
        assert healer.detect_regression(7.0) is False
        assert healer.detect_regression(8.0) is False

    @patch("src.iterate.healing.log_decision")
    def test_intervention_selection_per_dimension(self, mock_log):
        healer = SelfHealer()
        seen_strategies = set()
        for dim in INTERVENTION_STRATEGIES:
            plan = healer.select_intervention(dim)
            assert isinstance(plan, InterventionPlan)
            assert plan.dimension == dim
            assert plan.strategy_text == INTERVENTION_STRATEGIES[dim]
            assert plan.severity in ("minor", "major")
            seen_strategies.add(plan.strategy_text)

        # All 5 dimensions produce distinct strategies
        assert len(seen_strategies) == 5

    @patch("src.iterate.healing.log_decision")
    def test_diagnose_weakest(self, mock_log):
        healer = SelfHealer()
        evaluation = _make_eval("ad-1", clarity=3.0, value_prop=7.0)
        assert healer.diagnose(evaluation) == "clarity"

    @patch("src.iterate.healing.log_decision")
    def test_diagnose_tie_prefers_hard_gated(self, mock_log):
        healer = SelfHealer()
        # Both clarity and brand_voice at 4.0 — prefer brand_voice (hard-gated)
        evaluation = _make_eval("ad-1", clarity=4.0, brand_voice=4.0)
        assert healer.diagnose(evaluation) == "brand_voice"

    @patch("src.iterate.healing.log_decision")
    def test_original_brief_in_every_prompt(self, mock_log):
        """Verify the feedback prompt always contains the original brief fields."""
        healer = SelfHealer()
        evaluation = _make_eval("ad-1", clarity=3.0, passed=False)
        prompt = healer.build_feedback_prompt(SAMPLE_BRIEF, FAILING_AD, evaluation)

        # Must contain original brief fields
        assert SAMPLE_BRIEF.audience_segment in prompt
        assert SAMPLE_BRIEF.product_offer in prompt
        assert SAMPLE_BRIEF.campaign_goal in prompt
        assert SAMPLE_BRIEF.tone in prompt
        assert SAMPLE_BRIEF.competitive_context in prompt

        # Must contain current ad content
        assert FAILING_AD.primary_text in prompt
        assert FAILING_AD.headline in prompt

        # Must contain intervention guidance
        assert "clarity" in prompt.lower() or "Simplify" in prompt


# ---------------------------------------------------------------------------
# IterationController tests
# ---------------------------------------------------------------------------


class TestIterationController:
    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_accept_on_first_pass(self, mock_heal_log, mock_ctrl_log, db_conn):
        """Generator produces passing ad, controller returns it after 1 cycle."""
        passing_eval = _make_eval("", passed=True)

        controller, mock_gen, mock_eval = _build_controller(
            db_conn,
            gen_side_effect=[PASSING_AD],
            eval_side_effect=[passing_eval],
        )

        ad, records = controller.iterate(SAMPLE_BRIEF)

        assert ad is not None
        assert ad.headline == PASSING_AD.headline
        assert len(records) == 0  # No iteration records for first-pass accept
        mock_gen.generate.assert_called_once()

    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_component_fix_improves_score(self, mock_heal_log, mock_ctrl_log, db_conn):
        """First eval fails on clarity, component fix raises clarity, second eval passes."""
        failing_eval = _make_eval("", clarity=3.0, passed=False)
        # After component fix, coherence check shows improvement
        improved_eval = _make_eval("", clarity=7.0, passed=True)

        improved_ad = AdCopy(
            id="ad-improved",
            primary_text="Improved SAT prep ad.",
            headline="Better Headline",
            description="Improved description.",
            cta_button="Get Started",
            model_id="gemini-2.5-flash",
            token_count=90,
        )

        controller, mock_gen, mock_eval = _build_controller(
            db_conn,
            gen_side_effect=[FAILING_AD, improved_ad],
            eval_side_effect=[failing_eval, improved_eval],
        )

        ad, records = controller.iterate(SAMPLE_BRIEF)

        assert ad is not None
        assert len(records) == 1
        assert records[0].action_type == "component_fix"
        assert records[0].weak_dimension == "clarity"

    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_coherence_fail_triggers_full_regen(self, mock_heal_log, mock_ctrl_log, db_conn):
        """Component fix drops weighted_avg by >0.5, controller falls back to full regen."""
        # Initial eval: failing but decent
        initial_eval = _make_eval(
            "",
            clarity=5.0,
            value_prop=5.0,
            cta_effectiveness=5.0,
            brand_voice=5.0,
            emotional_resonance=5.0,
            passed=False,
        )
        # After component fix: coherence broken (big drop)
        broken_eval = _make_eval(
            "",
            clarity=6.0,
            value_prop=2.0,
            cta_effectiveness=2.0,
            brand_voice=2.0,
            emotional_resonance=2.0,
            passed=False,
        )
        # After full regen: passes
        regen_eval = _make_eval("", passed=True)

        regen_ad = AdCopy(
            id="ad-regen",
            primary_text="Fully regenerated ad.",
            headline="Regen Headline",
            description="Regen desc.",
            cta_button="Get Started",
            model_id="gemini-2.5-flash",
            token_count=95,
        )

        controller, mock_gen, mock_eval = _build_controller(
            db_conn,
            gen_side_effect=[FAILING_AD, FAILING_AD, regen_ad],
            eval_side_effect=[initial_eval, broken_eval, regen_eval],
        )

        ad, records = controller.iterate(SAMPLE_BRIEF)

        assert ad is not None
        # Should have component_fix then full_regen records
        action_types = [r.action_type for r in records]
        assert "component_fix" in action_types
        assert "full_regen" in action_types

    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_max_cycles_force_fail(self, mock_heal_log, mock_ctrl_log, db_conn):
        """3 consecutive failures -> controller returns None."""
        failing_eval = _make_eval("", clarity=3.0, passed=False)
        # Coherence passes but still failing
        still_failing_eval = _make_eval("", clarity=4.0, passed=False)

        ads = [
            AdCopy(
                id=f"ad-{i}",
                primary_text=f"Ad version {i}",
                headline=f"Headline {i}",
                description=f"Desc {i}",
                cta_button="Learn More",
                model_id="gemini-2.5-flash",
                token_count=80,
            )
            for i in range(10)
        ]

        # Pattern: generate -> eval(fail) -> fix -> coherence(pass but still fail)
        # -> eval(fail) -> ... until max cycles
        eval_results = [failing_eval, still_failing_eval] * 5

        controller, mock_gen, mock_eval = _build_controller(
            db_conn,
            gen_side_effect=ads,
            eval_side_effect=eval_results,
        )

        ad, records = controller.iterate(SAMPLE_BRIEF)

        assert ad is None
        assert len(records) > 0  # At least some iteration attempts

    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_decision_logging_at_every_branch(self, mock_heal_log, mock_ctrl_log, db_conn):
        """Verify log_decision called for each state transition during multi-cycle iteration."""
        failing_eval = _make_eval("", clarity=3.0, passed=False)
        passing_eval = _make_eval("", passed=True)

        improved_ad = AdCopy(
            id="ad-improved",
            primary_text="Improved.",
            headline="Better",
            description="Better desc.",
            cta_button="Get Started",
            model_id="gemini-2.5-flash",
            token_count=90,
        )

        controller, mock_gen, mock_eval = _build_controller(
            db_conn,
            gen_side_effect=[FAILING_AD, improved_ad],
            eval_side_effect=[failing_eval, passing_eval],
        )

        controller.iterate(SAMPLE_BRIEF)

        # Extract all log_decision actions from controller calls
        ctrl_actions = [call.args[1] for call in mock_ctrl_log.call_args_list]

        # Must have loop_start, generate_cycle, and at least one terminal state
        assert "loop_start" in ctrl_actions
        assert "generate_cycle" in ctrl_actions

        # Should have either accept or reject_retry
        has_terminal = "accept" in ctrl_actions or "reject_retry" in ctrl_actions
        assert has_terminal

    @patch("src.iterate.controller.log_decision")
    @patch("src.iterate.healing.log_decision")
    def test_diminishing_returns_early_exit(self, mock_heal_log, mock_ctrl_log, db_conn):
        """Controller exits early when score improvement falls below CONVERGENCE_THRESHOLD."""
        # Cycle 1: eval fails with score ~5.0
        eval_cycle1 = _make_eval(
            "",
            clarity=5.0,
            value_prop=5.0,
            cta_effectiveness=5.0,
            brand_voice=5.0,
            emotional_resonance=5.0,
            passed=False,
        )
        # Coherence check: tiny improvement (5.0 -> 5.1), delta 0.1 < 0.3 threshold
        coherence_eval = _make_eval(
            "",
            clarity=5.1,
            value_prop=5.1,
            cta_effectiveness=5.1,
            brand_voice=5.1,
            emotional_resonance=5.1,
            passed=False,
        )

        fixed_ad = AdCopy(
            id="ad-fixed",
            primary_text="Slightly improved ad.",
            headline="Better Headline",
            description="Better desc.",
            cta_button="Get Started",
            model_id="gemini-2.5-flash",
            token_count=85,
        )

        controller, mock_gen, mock_eval = _build_controller(
            db_conn,
            gen_side_effect=[FAILING_AD, fixed_ad],
            eval_side_effect=[eval_cycle1, coherence_eval],
        )

        ad, records = controller.iterate(SAMPLE_BRIEF)

        assert ad is None
        assert len(records) == 1  # One component_fix before early exit

        ctrl_actions = [call.args[1] for call in mock_ctrl_log.call_args_list]
        assert "diminishing_returns" in ctrl_actions
