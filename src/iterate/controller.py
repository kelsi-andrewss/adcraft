"""Iteration controller — procedural state machine for gen/eval/fix cycles.

Orchestrates: generate -> evaluate -> (if fail) component fix -> coherence check
-> re-evaluate, with full regeneration fallback when coherence breaks. Max 3 cycles.
"""

from __future__ import annotations

import sqlite3
from enum import Enum

from src.db.queries import insert_ad, insert_evaluation, insert_iteration
from src.decisions.logger import log_decision
from src.evaluate.engine import EvaluationEngine
from src.evaluate.rubrics import PASSING_THRESHOLD
from src.generate.engine import GenerationEngine
from src.iterate.healing import SelfHealer
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.evaluation import EvaluationResult
from src.models.iteration import IterationRecord

MAX_CYCLES = 3
COHERENCE_DROP_THRESHOLD = 0.5


class State(str, Enum):
    GENERATE = "generate"
    EVALUATE = "evaluate"
    FIX = "fix"
    COHERENCE_CHECK = "coherence_check"
    REGEN = "regen"
    ACCEPT = "accept"
    FAIL = "fail"


class IterationController:
    """Procedural state machine driving the gen -> eval -> fix loop."""

    def __init__(
        self,
        generator: GenerationEngine,
        evaluator: EvaluationEngine,
        healer: SelfHealer,
        conn: sqlite3.Connection,
    ) -> None:
        self._gen = generator
        self._eval = evaluator
        self._healer = healer
        self._conn = conn

    def iterate(
        self, brief: AdBrief
    ) -> tuple[AdCopy | None, list[IterationRecord]]:
        """Run the iteration loop until the ad passes or max cycles reached.

        Returns (passing_ad_or_None, list_of_iteration_records).
        """
        state = State.GENERATE
        cycle = 0
        ad: AdCopy | None = None
        evaluation: EvaluationResult | None = None
        pre_fix_score: float = 0.0
        records: list[IterationRecord] = []
        feedback_context: list[str] = []

        log_decision(
            "iterate",
            "loop_start",
            f"Starting iteration loop for brief: audience={brief.audience_segment}, "
            f"offer={brief.product_offer[:60]}",
            {
                "audience": brief.audience_segment,
                "tone": brief.tone,
                "max_cycles": MAX_CYCLES,
            },
        )

        while state not in (State.ACCEPT, State.FAIL):
            match state:
                case State.GENERATE:
                    cycle += 1

                    if cycle > MAX_CYCLES:
                        last_avg = (
                            evaluation.weighted_average
                            if evaluation
                            else 0.0
                        )
                        log_decision(
                            "iterate",
                            "force_fail",
                            f"Max cycles ({MAX_CYCLES}) reached. "
                            f"Last weighted_avg={last_avg:.2f}",
                            {
                                "cycles": cycle - 1,
                                "max_cycles": MAX_CYCLES,
                                "last_score": last_avg,
                            },
                        )
                        state = State.FAIL
                        continue

                    log_decision(
                        "iterate",
                        "generate_cycle",
                        f"Cycle {cycle}/{MAX_CYCLES}: generating ad from brief",
                        {"cycle": cycle, "max_cycles": MAX_CYCLES},
                    )

                    ad = self._gen.generate(brief)
                    ad = self._persist_ad(ad, brief)
                    state = State.EVALUATE

                case State.EVALUATE:
                    assert ad is not None
                    evaluation = self._eval.evaluate_iteration(ad)
                    self._persist_evaluation(evaluation, ad.id)
                    self._healer.update_ratchet(evaluation.weighted_average)

                    if evaluation.passed_threshold:
                        log_decision(
                            "iterate",
                            "accept",
                            f"Ad passed on cycle {cycle}: "
                            f"weighted_avg="
                            f"{evaluation.weighted_average:.2f} "
                            f">= {PASSING_THRESHOLD}",
                            {
                                "cycle": cycle,
                                "weighted_avg": evaluation.weighted_average,
                                "threshold": PASSING_THRESHOLD,
                            },
                        )
                        state = State.ACCEPT
                    else:
                        log_decision(
                            "iterate",
                            "reject_retry",
                            f"Ad failed cycle {cycle}: "
                            f"weighted_avg="
                            f"{evaluation.weighted_average:.2f} "
                            f"< {PASSING_THRESHOLD}",
                            {
                                "cycle": cycle,
                                "weighted_avg": evaluation.weighted_average,
                                "threshold": PASSING_THRESHOLD,
                                "hard_gate_failures": evaluation.hard_gate_failures,
                            },
                        )
                        pre_fix_score = evaluation.weighted_average
                        state = State.FIX

                case State.FIX:
                    assert ad is not None
                    assert evaluation is not None

                    # Check if we'd exceed max cycles with another generate
                    if cycle >= MAX_CYCLES:
                        log_decision(
                            "iterate",
                            "force_fail",
                            f"Max cycles ({MAX_CYCLES}) reached without passing. "
                            f"Last weighted_avg={evaluation.weighted_average:.2f}",
                            {
                                "cycles": cycle,
                                "max_cycles": MAX_CYCLES,
                                "last_score": evaluation.weighted_average,
                            },
                        )
                        state = State.FAIL
                        continue

                    weak_dim = self._healer.diagnose(evaluation)
                    feedback = self._healer.build_feedback_prompt(
                        brief, ad, evaluation
                    )
                    feedback_context.append(feedback)

                    log_decision(
                        "iterate",
                        "component_fix",
                        f"Attempting component fix on {weak_dim} "
                        f"(pre-fix weighted_avg={pre_fix_score:.2f})",
                        {
                            "cycle": cycle,
                            "weak_dimension": weak_dim,
                            "pre_fix_score": pre_fix_score,
                        },
                    )

                    source_ad = ad
                    # Generate a new ad using the feedback-enriched brief
                    fix_brief = self._build_fix_brief(brief, feedback)
                    ad = self._gen.generate(fix_brief)
                    ad = self._persist_ad(ad, brief)

                    # Record the iteration
                    record = IterationRecord(
                        source_ad_id=source_ad.id,
                        target_ad_id=ad.id,
                        cycle_number=cycle,
                        action_type="component_fix",
                        weak_dimension=weak_dim,
                        token_cost=float(ad.token_count),
                    )
                    self._persist_iteration(record, feedback)
                    records.append(record)

                    state = State.COHERENCE_CHECK

                case State.COHERENCE_CHECK:
                    assert ad is not None

                    coherence_eval = self._eval.evaluate_iteration(ad)
                    self._persist_evaluation(coherence_eval, ad.id)

                    score_delta = coherence_eval.weighted_average - pre_fix_score

                    if score_delta < -COHERENCE_DROP_THRESHOLD:
                        log_decision(
                            "iterate",
                            "coherence_fail_full_regen",
                            f"Coherence broken: weighted_avg dropped {score_delta:.2f} "
                            f"(from {pre_fix_score:.2f} to {coherence_eval.weighted_average:.2f}), "
                            f"exceeds -{COHERENCE_DROP_THRESHOLD} threshold. "
                            f"Falling back to full regen.",
                            {
                                "pre_fix_score": pre_fix_score,
                                "post_fix_score": coherence_eval.weighted_average,
                                "delta": score_delta,
                                "threshold": COHERENCE_DROP_THRESHOLD,
                            },
                        )
                        state = State.REGEN
                    else:
                        log_decision(
                            "iterate",
                            "coherence_pass",
                            f"Coherence maintained: delta={score_delta:+.2f} "
                            f"(from {pre_fix_score:.2f} to {coherence_eval.weighted_average:.2f})",
                            {
                                "pre_fix_score": pre_fix_score,
                                "post_fix_score": coherence_eval.weighted_average,
                                "delta": score_delta,
                            },
                        )
                        evaluation = coherence_eval
                        self._healer.update_ratchet(coherence_eval.weighted_average)

                        if coherence_eval.passed_threshold:
                            log_decision(
                                "iterate",
                                "accept",
                                f"Ad passed after component fix on cycle {cycle}: "
                                f"weighted_avg={coherence_eval.weighted_average:.2f}",
                                {
                                    "cycle": cycle,
                                    "weighted_avg": coherence_eval.weighted_average,
                                },
                            )
                            state = State.ACCEPT
                        else:
                            # Loop back to generate for next cycle
                            cycle += 1
                            if cycle > MAX_CYCLES:
                                log_decision(
                                    "iterate",
                                    "force_fail",
                                    f"Max cycles ({MAX_CYCLES}) reached after coherence pass. "
                                    f"Last weighted_avg={coherence_eval.weighted_average:.2f}",
                                    {
                                        "cycles": cycle - 1,
                                        "max_cycles": MAX_CYCLES,
                                        "last_score": coherence_eval.weighted_average,
                                    },
                                )
                                state = State.FAIL
                            else:
                                state = State.EVALUATE

                case State.REGEN:
                    assert ad is not None
                    assert evaluation is not None

                    cycle += 1
                    if cycle > MAX_CYCLES:
                        log_decision(
                            "iterate",
                            "force_fail",
                            f"Max cycles ({MAX_CYCLES}) reached during regen fallback. "
                            f"Last weighted_avg={evaluation.weighted_average:.2f}",
                            {
                                "cycles": cycle - 1,
                                "max_cycles": MAX_CYCLES,
                                "last_score": evaluation.weighted_average,
                            },
                        )
                        state = State.FAIL
                        continue

                    log_decision(
                        "iterate",
                        "full_regen",
                        f"Full regeneration from original brief on cycle {cycle} "
                        f"with accumulated feedback context",
                        {
                            "cycle": cycle,
                            "feedback_context_count": len(feedback_context),
                        },
                    )

                    source_ad = ad
                    # Full regen: use original brief with accumulated feedback
                    regen_brief = self._build_regen_brief(brief, feedback_context)
                    ad = self._gen.generate(regen_brief)
                    ad = self._persist_ad(ad, brief)

                    weak_dim = self._healer.diagnose(evaluation)
                    record = IterationRecord(
                        source_ad_id=source_ad.id,
                        target_ad_id=ad.id,
                        cycle_number=cycle,
                        action_type="full_regen",
                        weak_dimension=weak_dim,
                        token_cost=float(ad.token_count),
                    )
                    self._persist_iteration(
                        record, "\n---\n".join(feedback_context)
                    )
                    records.append(record)

                    state = State.EVALUATE

        if state == State.ACCEPT:
            return ad, records
        return None, records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist_ad(self, ad: AdCopy, brief: AdBrief) -> AdCopy:
        """Persist ad to DB and return it with its assigned ID."""
        ad_id = insert_ad(
            self._conn,
            primary_text=ad.primary_text,
            headline=ad.headline,
            description=ad.description,
            cta_button=ad.cta_button,
            model_id=ad.model_id,
            temperature=ad.generation_config.get("temperature"),
        )
        ad.id = ad_id
        return ad

    def _persist_evaluation(
        self, evaluation: EvaluationResult, ad_id: str
    ) -> None:
        """Persist all dimension scores to DB.

        Uses the explicit ad_id (from the DB-persisted ad) rather than
        evaluation.ad_id, which may not match after mock or re-assignment.
        """
        for s in evaluation.scores:
            insert_evaluation(
                self._conn,
                ad_id=ad_id,
                dimension=s.dimension,
                score=s.score,
                rationale=s.rationale,
                confidence=s.confidence,
                evaluator_model=evaluation.evaluator_model,
                eval_mode="iteration",
            )

    def _persist_iteration(
        self, record: IterationRecord, feedback_prompt: str
    ) -> None:
        """Persist an iteration record to DB."""
        insert_iteration(
            self._conn,
            source_ad_id=record.source_ad_id,
            target_ad_id=record.target_ad_id,
            cycle_number=record.cycle_number,
            action_type=record.action_type,
            weak_dimension=record.weak_dimension,
            feedback_prompt=feedback_prompt,
            delta_weighted_avg=None,
            token_cost=record.token_cost,
        )

    @staticmethod
    def _build_fix_brief(brief: AdBrief, feedback: str) -> AdBrief:
        """Create a brief that includes feedback for a component fix.

        Appends the feedback to the competitive_context field so the
        generator sees both original brief and iteration feedback.
        """
        return AdBrief(
            audience_segment=brief.audience_segment,
            product_offer=brief.product_offer,
            campaign_goal=brief.campaign_goal,
            tone=brief.tone,
            competitive_context=(
                brief.competitive_context + "\n\n" + feedback
            ),
        )

    @staticmethod
    def _build_regen_brief(
        brief: AdBrief, feedback_context: list[str]
    ) -> AdBrief:
        """Create a brief for full regeneration with accumulated feedback."""
        combined_feedback = (
            "PREVIOUS ITERATION FEEDBACK (use as guidance, "
            "but regenerate the ad from scratch):\n\n"
            + "\n---\n".join(feedback_context)
        )
        return AdBrief(
            audience_segment=brief.audience_segment,
            product_offer=brief.product_offer,
            campaign_goal=brief.campaign_goal,
            tone=brief.tone,
            competitive_context=(
                brief.competitive_context + "\n\n" + combined_feedback
            ),
        )
