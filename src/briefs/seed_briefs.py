"""Seed ad briefs for the Varsity Tutors SAT prep campaign.

Covers parent and student audience segments across a range of tones.
These briefs drive the initial generation pipeline before the system
learns to generate its own brief variations.
"""

import copy

from src.models.brief import AdBrief

COMPETITIVE_CONTEXT = (
    "Princeton Review, Kaplan, and Khan Academy all offer SAT prep products. "
    "Varsity Tutors differentiates with live 1-on-1 tutoring and adaptive learning."
)

SEED_BRIEFS: list[AdBrief] = [
    # --- Parent-focused briefs ---
    AdBrief(
        audience_segment="parent",
        product_offer=(
            "Varsity Tutors SAT Prep — 1-on-1 live tutoring with score improvement guarantee"
        ),
        campaign_goal=(
            "Drive sign-ups by framing SAT prep as a high-ROI investment in the child's future"
        ),
        tone="authoritative",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="parent",
        product_offer="Varsity Tutors SAT Prep — personalized study plan with progress tracking",
        campaign_goal="Reduce perceived risk with score guarantee and money-back messaging",
        tone="reassuring",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="parent",
        product_offer="Varsity Tutors SAT Prep — expert tutors matched to student learning style",
        campaign_goal=(
            "Create urgency around college admissions timelines and competitive advantage"
        ),
        tone="urgent",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="parent",
        product_offer="Varsity Tutors SAT Prep — flexible scheduling with on-demand sessions",
        campaign_goal="Position SAT prep as essential for college admissions edge, not optional",
        tone="authoritative",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    # --- Student-focused briefs ---
    AdBrief(
        audience_segment="student",
        product_offer="Varsity Tutors SAT Prep — adaptive practice that focuses on your weak spots",
        campaign_goal="Motivate students to own their score improvement and take action now",
        tone="motivational",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors SAT Prep — real tutors, not just videos, available when you need them"
        ),
        campaign_goal="Build confidence by showing SAT prep can be low-stress and even engaging",
        tone="casual",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors SAT Prep — proven score boosts that open doors to top colleges"
        ),
        campaign_goal="Connect higher SAT scores to college independence and future opportunities",
        tone="aspirational",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors SAT Prep — study smarter with AI-powered practice + live tutor support"
        ),
        campaign_goal="Leverage peer comparison to drive sign-ups — don't fall behind classmates",
        tone="motivational",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
]


def get_seed_briefs() -> list[AdBrief]:
    """Return a copy of the seed briefs list.

    Returns a deep copy so callers can mutate without affecting the originals.
    """
    return copy.deepcopy(SEED_BRIEFS)
