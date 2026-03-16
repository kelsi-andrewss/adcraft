"""Self-healing and quality ratchet for the iteration loop.

SelfHealer maintains a running quality floor (ratchet) that only goes up,
detects regressions when new scores fall below it, and selects
dimension-specific intervention strategies to guide targeted fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.decisions.logger import log_decision
from src.evaluate.rubrics import (
    BRAND_VOICE_HARD_GATE,
    DIMENSION_WEIGHTS,
    DIMENSIONS,
    PASSING_THRESHOLD,
    PEDAGOGICAL_INTEGRITY_HARD_GATE,
)
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.evaluation import EvaluationResult
from src.theme import THEME

PASSING_TARGETS: dict[str, float] = {
    "brand_voice": float(BRAND_VOICE_HARD_GATE),
    "pedagogical_integrity": float(PEDAGOGICAL_INTEGRITY_HARD_GATE),
}
DEFAULT_PASSING_TARGET = 7.0


@dataclass
class InterventionPlan:
    """Targeted feedback strategy for a specific weak dimension."""

    dimension: str
    strategy_text: str
    severity: str  # "minor" or "major"


# Dimension-specific intervention strategies keyed by actual dimension names
INTERVENTION_STRATEGIES: dict[str, str] = {
    "clarity": ("Simplify sentence structure. Remove jargon. One idea per sentence."),
    "learner_benefit": (
        "Sharpen the specific learning outcome. What transformation does the student "
        "experience that competitors can't deliver?"
    ),
    "cta_effectiveness": (
        "The CTA button is platform-constrained (Learn More, Get Started, Sign Up, "
        "Book Now, etc). To score 7+, the ad copy BEFORE the CTA must build toward "
        "the action — establish what the reader gets (free diagnostic, custom plan, "
        "tutor match) so the button feels like a natural next step, not a generic ask. "
        "Frame the CTA as accessing something valuable, not just clicking a button."
    ),
    "brand_voice": (
        f"Match {THEME.brand_name}'s tone: confident, warm, expert but approachable. "
        "Avoid corporate stiffness."
    ),
    "student_empathy": (
        "Connect to the parent's anxiety about their child's future or the "
        "student's desire to succeed. Be specific, not generic."
    ),
    "pedagogical_integrity": (
        "Ensure claims align with sound educational principles. Remove any guaranteed "
        "outcomes, shortcut promises, or learning-without-effort language. Emphasize "
        "personalized, expert-guided learning processes."
    ),
}


class SelfHealer:
    """Detects quality regressions and prescribes targeted interventions."""

    def __init__(self) -> None:
        self.running_avg: float = 0.0

    def update_ratchet(self, score: float) -> None:
        """Update the quality ratchet. Only goes up."""
        previous = self.running_avg
        self.running_avg = max(self.running_avg, score)
        if self.running_avg > previous:
            log_decision(
                "healer",
                "ratchet_raised",
                f"Quality ratchet raised from {previous:.2f} to {self.running_avg:.2f}",
                {"previous": previous, "new": self.running_avg, "score": score},
            )
        else:
            log_decision(
                "healer",
                "ratchet_held",
                f"Quality ratchet held at {self.running_avg:.2f} "
                f"(score {score:.2f} did not exceed)",
                {"running_avg": self.running_avg, "score": score},
            )

    def detect_regression(self, current_score: float) -> bool:
        """Return True if current_score is below the running average."""
        is_regression = current_score < self.running_avg
        if is_regression:
            log_decision(
                "healer",
                "regression_detected",
                f"Score {current_score:.2f} regressed below ratchet {self.running_avg:.2f}",
                {"current_score": current_score, "running_avg": self.running_avg},
            )
        else:
            log_decision(
                "healer",
                "no_regression",
                f"Score {current_score:.2f} at or above ratchet {self.running_avg:.2f}",
                {"current_score": current_score, "running_avg": self.running_avg},
            )
        return is_regression

    def diagnose(self, evaluation: EvaluationResult) -> str:
        """Identify the highest-impact dimension to fix.

        Impact score = DIMENSION_WEIGHTS[dim] * max(0, passing_target - current_score).
        Passing target is 7 for normal dimensions, 5 for brand_voice (hard gate),
        6 for pedagogical_integrity (hard gate). Ties broken by DIMENSIONS order.
        """
        scores_by_dim = {s.dimension: s.score for s in evaluation.scores}

        impact_scores: dict[str, float] = {}
        for dim, score in scores_by_dim.items():
            target = PASSING_TARGETS.get(dim, DEFAULT_PASSING_TARGET)
            gap = max(0.0, target - score)
            impact_scores[dim] = DIMENSION_WEIGHTS[dim] * gap

        max_impact = max(impact_scores.values())

        if max_impact > 0:
            # Pick dimension with highest impact; break ties by DIMENSIONS order
            weakest = next(d for d in DIMENSIONS if impact_scores.get(d, 0.0) == max_impact)
        else:
            # All dimensions at or above target — fall back to lowest absolute score
            weakest = min(scores_by_dim, key=scores_by_dim.get)

        log_decision(
            "healer",
            "diagnose_weakest",
            f"Highest-impact dimension: {weakest} "
            f"(score={scores_by_dim[weakest]:.1f}, "
            f"impact={impact_scores.get(weakest, 0.0):.3f})",
            {
                "weakest": weakest,
                "score": scores_by_dim[weakest],
                "impact_scores": impact_scores,
                "all_scores": scores_by_dim,
            },
        )

        return weakest

    def select_intervention(self, dimension: str) -> InterventionPlan:
        """Return a targeted intervention plan for the given dimension."""
        strategy = INTERVENTION_STRATEGIES.get(
            dimension,
            f"Improve the {dimension} dimension of this ad.",
        )

        # Severity: major if the dimension is hard-gated (brand_voice, pedagogical_integrity)
        # or if it's one of the high-weight dimensions
        severity = (
            "major"
            if dimension in ("brand_voice", "clarity", "learner_benefit", "pedagogical_integrity")
            else "minor"
        )

        plan = InterventionPlan(
            dimension=dimension,
            strategy_text=strategy,
            severity=severity,
        )

        log_decision(
            "healer",
            "intervention_selected",
            f"Intervention for {dimension} ({severity}): {strategy[:80]}",
            {
                "dimension": dimension,
                "severity": severity,
                "strategy": strategy,
            },
        )

        return plan

    def build_feedback_prompt(
        self,
        brief: AdBrief,
        ad: AdCopy,
        evaluation: EvaluationResult,
    ) -> str:
        """Build a structured feedback prompt for the generator.

        Always includes the original brief to prevent context drift.
        """
        weakest = self.diagnose(evaluation)
        plan = self.select_intervention(weakest)

        scores_summary = "\n".join(
            f"  - {s.dimension}: {s.score:.1f} — {s.rationale[:120]}" for s in evaluation.scores
        )

        weak_score = next(s.score for s in evaluation.scores if s.dimension == weakest)

        prompt = (
            "ITERATION FEEDBACK — Fix the weakest dimension "
            "while preserving strengths.\n\n"
            "ORIGINAL BRIEF (always reference this for context):\n"
            f"- Audience: {brief.audience_segment}\n"
            f"- Offer: {brief.product_offer}\n"
            f"- Goal: {brief.campaign_goal}\n"
            f"- Tone: {brief.tone}\n"
            f"- Competitive context: {brief.competitive_context}\n\n"
            "CURRENT AD:\n"
            f"- Primary text: {ad.primary_text}\n"
            f"- Headline: {ad.headline}\n"
            f"- Description: {ad.description}\n"
            f"- CTA: {ad.cta_button}\n\n"
            "EVALUATION SCORES:\n"
            f"{scores_summary}\n"
            f"Weighted average: {evaluation.weighted_average:.2f}\n"
            f"Passing threshold: {PASSING_THRESHOLD}\n\n"
            f"WEAKEST DIMENSION: {weakest} "
            f"(score: {weak_score:.1f})\n\n"
            f"INTERVENTION STRATEGY ({plan.severity} severity):\n"
            f"{plan.strategy_text}\n\n"
            "INSTRUCTIONS:\n"
            f"1. Focus on improving {weakest} "
            "using the strategy above.\n"
            "2. Keep the other dimensions at their current "
            "quality or better.\n"
            "3. Stay true to the original brief — "
            "audience, offer, goal, and tone.\n"
            "4. Do NOT introduce new claims not supported "
            "by the brief.\n"
            "5. Maintain brand voice: supportive, "
            "knowledgeable, warm, results-oriented.\n"
        )

        log_decision(
            "healer",
            "feedback_prompt_built",
            f"Built feedback prompt targeting {weakest} ({plan.severity}), "
            f"includes original brief fields",
            {
                "target_dimension": weakest,
                "severity": plan.severity,
                "brief_audience": brief.audience_segment,
                "weighted_avg": evaluation.weighted_average,
                "passing_threshold": PASSING_THRESHOLD,
            },
        )

        return prompt
