"""Seed ad briefs for the Varsity Tutors SAT prep campaign.

Covers parent, student, and family audience segments across a range of
tones, product offers, and campaign goals. These briefs drive the
generation pipeline and produce a diverse ad library.
"""

import copy

from src.models.brief import AdBrief

COMPETITIVE_CONTEXT = (
    "Princeton Review, Kaplan, and Khan Academy all offer SAT prep products. "
    "Varsity Tutors differentiates with live 1-on-1 tutoring and adaptive learning."
)

SEED_BRIEFS: list[AdBrief] = [
    # =================================================================
    # PARENT — Awareness / Consideration
    # =================================================================
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
    AdBrief(
        audience_segment="parent",
        product_offer="Varsity Tutors SAT Prep — free diagnostic test to find score gaps",
        campaign_goal="Lower barrier to entry with a free first step that demonstrates value",
        tone="reassuring",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="parent",
        product_offer=(
            "Varsity Tutors SAT Prep — summer intensive program before senior year"
        ),
        campaign_goal="Create seasonal urgency: summer is the last window before fall applications",
        tone="urgent",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="parent",
        product_offer="Varsity Tutors SAT Prep — weekly progress reports sent directly to parents",
        campaign_goal="Appeal to involved parents who want visibility into their child's prep",
        tone="data-driven",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="parent",
        product_offer=(
            "Varsity Tutors SAT Prep — tutors who scored in the 99th percentile on the SAT"
        ),
        campaign_goal="Build trust through tutor credentials and expertise proof points",
        tone="authoritative",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    # =================================================================
    # PARENT — Conversion / Retargeting
    # =================================================================
    AdBrief(
        audience_segment="parent",
        product_offer="Varsity Tutors SAT Prep — limited-time bundle: 10 sessions + diagnostic",
        campaign_goal="Drive conversion with a time-limited package deal for retargeted visitors",
        tone="urgent",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="parent",
        product_offer=(
            "Varsity Tutors SAT Prep — families see an average 160-point improvement"
        ),
        campaign_goal="Use social proof and outcome data to convert consideration into action",
        tone="testimonial",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    # =================================================================
    # STUDENT — Awareness / Motivation
    # =================================================================
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
    AdBrief(
        audience_segment="student",
        product_offer="Varsity Tutors SAT Prep — free practice test so you know where you stand",
        campaign_goal="Curiosity-driven: find out your baseline score, no commitment required",
        tone="casual",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors SAT Prep — flexible sessions that fit around school and activities"
        ),
        campaign_goal="Remove the 'too busy' objection by emphasizing schedule flexibility",
        tone="reassuring",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors SAT Prep — master the SAT math section with targeted drills"
        ),
        campaign_goal="Speak to students who know their weak section and want focused help",
        tone="data-driven",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors SAT Prep — crush test anxiety with strategies that actually work"
        ),
        campaign_goal="Address the emotional barrier: test anxiety, not lack of knowledge",
        tone="empathetic",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    # =================================================================
    # STUDENT — Conversion / Urgency
    # =================================================================
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors SAT Prep — your friends are prepping, are you?"
        ),
        campaign_goal="Social proof + FOMO to push undecided students toward sign-up",
        tone="motivational",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors SAT Prep — 8 weeks to your best SAT score"
        ),
        campaign_goal="Create urgency with a concrete timeline and achievable goal",
        tone="urgent",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    # =================================================================
    # FAMILY — Decision-makers together
    # =================================================================
    AdBrief(
        audience_segment="family",
        product_offer=(
            "Varsity Tutors SAT Prep — a college prep investment the whole family can track"
        ),
        campaign_goal="Position SAT prep as a family decision with shared visibility and goals",
        tone="reassuring",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="family",
        product_offer=(
            "Varsity Tutors SAT Prep — from first diagnostic to test day, we're with you"
        ),
        campaign_goal="Tell the full journey story: diagnose, plan, practice, score, celebrate",
        tone="aspirational",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    # =================================================================
    # STUDENT — AP Prep (adjacent product)
    # =================================================================
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors AP Exam Prep — 1-on-1 tutoring for AP Calculus, English, and more"
        ),
        campaign_goal="Expand beyond SAT to AP exams — same quality, same approach",
        tone="authoritative",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    AdBrief(
        audience_segment="student",
        product_offer=(
            "Varsity Tutors AP Exam Prep — boost your GPA and earn college credit early"
        ),
        campaign_goal="Connect AP scores to tangible college benefits: credit, placement, savings",
        tone="aspirational",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
    # =================================================================
    # PARENT — AP Prep
    # =================================================================
    AdBrief(
        audience_segment="parent",
        product_offer=(
            "Varsity Tutors AP Exam Prep — help your student earn college credit in high school"
        ),
        campaign_goal="Frame AP prep as a cost-saving investment: college credits before college",
        tone="data-driven",
        competitive_context=COMPETITIVE_CONTEXT,
    ),
]


def get_seed_briefs() -> list[AdBrief]:
    """Return a copy of the seed briefs list.

    Returns a deep copy so callers can mutate without affecting the originals.
    """
    return copy.deepcopy(SEED_BRIEFS)
