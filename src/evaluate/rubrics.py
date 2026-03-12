"""Evaluation rubrics, prompts, and scoring constants for AdCraft.

Defines the six evaluation dimensions, their rubric text, CoT prompt
templates, few-shot examples, weights, and hard gate thresholds.
"""

from __future__ import annotations

from src.theme import THEME

# ---------------------------------------------------------------------------
# Dimension weights — must sum to 1.0
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: dict[str, float] = {
    "clarity": 0.20,
    "learner_benefit": 0.20,
    "cta_effectiveness": 0.16,
    "brand_voice": 0.12,
    "student_empathy": 0.12,
    "pedagogical_integrity": 0.20,
}

DIMENSIONS: list[str] = list(DIMENSION_WEIGHTS.keys())

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

PASSING_THRESHOLD: float = 7.0
BRAND_VOICE_HARD_GATE: int = 5
PEDAGOGICAL_INTEGRITY_HARD_GATE: int = 6

# ---------------------------------------------------------------------------
# Rubric text per dimension
# ---------------------------------------------------------------------------

RUBRICS: dict[str, str] = {
    "clarity": """CLARITY — How clearly does the ad communicate its message?

1-3 (Poor): Confusing structure, jargon-heavy, unclear what the product is or does.
  The reader finishes and doesn't know what's being offered or what to do next.
  Example markers: run-on sentences, undefined acronyms, mixed messages, no clear subject.

4-6 (Adequate): Message is understandable but requires effort. Some filler or redundancy.
  The reader gets the gist but specific claims or next steps are fuzzy.
  Example markers: explains obvious things (e.g., what the SAT is), buries the point, verbose.

7-8 (Strong): Clear, logical flow. Each sentence builds toward the CTA. Specific and concise.
  The reader immediately understands the offer, the benefit, and the action.
  Example markers: strong opening hook, specific numbers/claims, no wasted words.

9-10 (Exceptional): Crystal clear at a glance. Could be understood in a 2-second scroll.
  Every word earns its place. Structure guides the eye from problem to solution to action.""",
    "learner_benefit": f"""LEARNER BENEFIT — How compelling is the learning transformation promised?

1-3 (Poor): No learning outcome articulated. Generic claims like "we help students" or
  "improve your scores" with no specifics. Could be any tutoring company.
  Example markers: no measurable outcomes, no before/after framing, feature lists without benefits.

4-6 (Adequate): Mentions learning but vaguely. "Better grades" or "score improvement" without
  specifics on how the learning happens or what transformation occurs.
  Example markers: generic improvement language, no methodology detail, no student journey.

7-8 (Strong): Specific learning outcomes — score improvement numbers, concept mastery,
  skill gaps closed. Clear connection between {THEME.brand_name}'s approach and the result.
  Example markers: 160-point improvement, diagnostic-first methodology, personalized plans.

9-10 (Exceptional): Transformation narrative — the reader can envision the before/after
  learning journey. Measurable impact on the student's academic trajectory with
  specific proof points and methodology that makes the outcome believable.""",
    "cta_effectiveness": """CTA EFFECTIVENESS — How well does the call-to-action drive the next step?

1-3 (Poor): Generic or missing CTA. No urgency, no value framing.
  "Sign Up" or "Click Here" with no context on what happens next.
  Example markers: transactional language, no benefit attached to the action.

4-6 (Adequate): CTA exists and is clear, but doesn't compel action.
  The reader knows what to click but has no strong reason to do it NOW.
  Example markers: standard "Learn More" without supporting context, no risk reduction.

7-8 (Strong): CTA is clear, action-oriented, and connected to a benefit.
  Lowers barrier to entry (free consultation, no commitment) or creates appropriate urgency.
  Example markers: "Book a Free Consultation", "Get Your Custom Plan", benefit-linked.

9-10 (Exceptional): CTA feels like the natural next step — not a sales push.
  Perfectly aligned with the ad's emotional arc. Reader WANTS to click.""",
    "brand_voice": f"""BRAND VOICE — Does this sound like {THEME.brand_name}?

{THEME.brand_name}'s brand voice is: supportive, knowledgeable, encouraging, results-oriented,
warm but professional. They position themselves as a partner in the student's journey,
not a fear-based pressure seller. They're confident in their results without being arrogant.

1-3 (Poor): Completely off-brand. Fear-based, overly casual, corporate jargon, or generic.
  Could not be identified as {THEME.brand_name}. Damages brand perception.
  Example markers: ALL CAPS, "DON'T FAIL", "lol", "leverage synergies", no personality.

4-6 (Adequate): Somewhat neutral but not distinctly {THEME.brand_name}.
  Not offensive to the brand but interchangeable with any education company.
  Example markers: generic professional tone, no warmth, no personality, template-feeling.

7-8 (Strong): Recognizably {THEME.brand_name}. Supportive, knowledgeable, encouraging.
  Balances professionalism with warmth. Results-focused without being pushy.
  Example markers: specific to {THEME.brand_name}'s approach (1-on-1, personalized, expert tutors).

9-10 (Exceptional): Could only be {THEME.brand_name}. Distinctive voice that builds brand equity.
  Perfect balance of warmth, expertise, and confidence. Memorable.""",
    "student_empathy": """STUDENT EMPATHY — Does the ad connect with the student/parent experience?

1-3 (Poor): Emotionally flat or emotionally manipulative.
  Either reads like a product spec with no feeling, or uses fear/guilt to manipulate.
  Example markers: "services available", pure feature lists, "RUIN THEIR FUTURE".

4-6 (Adequate): Attempts emotional connection but doesn't land.
  Generic empathy that could apply to any learning context.
  Example markers: generic "success" language, obvious emotional buttons without depth.

7-8 (Strong): Genuine connection with the student or parent experience.
  Understands real feelings — the "Aha!" moment of understanding a concept,
  reduction of learning anxiety, confidence building after mastering material.
  Example markers: specific pain points (score ceiling, test anxiety, access to opportunity).

9-10 (Exceptional): Deeply resonant — reader feels understood as a learner.
  Taps into core motivations (parent: wanting the best for their child; student: proving
  themselves) without manipulation. Reader feels the ad was written by someone who
  truly understands the learning journey.""",
    "pedagogical_integrity": f"""PEDAGOGICAL INTEGRITY — Does the ad accurately represent learning outcomes?

Evaluate whether the ad's claims align with sound educational principles.
{THEME.brand_name} is committed to honest, evidence-based education marketing.

1-3 (Poor): Misleading claims that misrepresent how learning works.
  Promises guaranteed outcomes, exam shortcuts, or learning-without-effort.
  Example markers: "100% score guarantee", "learn SAT tricks in 24 hours",
  "no studying required", miracle transformation claims.

4-6 (Adequate): Neutral — no misleading claims but no pedagogical depth.
  Makes standard marketing claims without engaging with how learning actually works.
  Example markers: generic "improve your scores", no methodology detail, surface-level.

7-8 (Strong): Demonstrates genuine educational value.
  References personalized learning, expert instruction, evidence-based methods.
  The ad conveys that improvement comes through guided, structured work.
  Example markers: diagnostic assessments, mastery-based progression, adaptive learning.

9-10 (Exceptional): Embodies pedagogical best practices.
  Reflects growth mindset, mastery-based learning, intrinsic motivation alongside results.
  The ad itself serves as an example of honest, inspiring education communication.""",
}

# ---------------------------------------------------------------------------
# Few-shot examples per dimension (extracted from reference ads)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: dict[str, str] = {
    "clarity": f"""Example 1 (Score: 9):
Ad: "Your child's SAT score shouldn't be limited by access to great teaching. {THEME.brand_name} connects students with expert SAT tutors who've helped families like yours see an average 160-point score improvement. Our personalized 1-on-1 approach means your student gets a study plan built around their specific strengths and weaknesses -- not a one-size-fits-all program."
Rationale: Every sentence has a clear purpose — problem (access), solution (expert tutors + 160-point improvement), differentiator (personalized plans). No wasted words. The reader instantly knows what's offered and why it matters.
Score: 9

Example 2 (Score: 2):
Ad: "{THEME.brand_name} leverages proprietary adaptive learning technology and a curated network of credentialed educators to deliver optimized SAT preparation outcomes. Our pedagogical methodology synthesizes evidence-based instructional strategies with real-time performance analytics to maximize score trajectory."
Rationale: Jargon-laden and impenetrable. "Pedagogical methodology" and "score trajectory" are corporate speak that obscures the actual offer. A parent reading this would have no idea what they're buying or why it's better. Fails the 2-second scroll test completely.
Score: 2""",
    "learner_benefit": """Example 1 (Score: 9):
Ad: "The SAT is 3 months away. Is your student ready? Our expert tutors have helped over 3,000 students boost their scores by an average of 160 points. We start with a diagnostic assessment to find exactly where your child needs help -- then build a custom study plan that targets those gaps."
Rationale: Triple proof points — 3,000 students (social proof), 160-point average (specific result), diagnostic-first approach (unique methodology). Each claim is specific and credible. Clear differentiation from generic "we help with SAT" competitors. The learning transformation is concrete and measurable.
Score: 9

Example 2 (Score: 2):
Ad: "Looking for SAT help? We offer tutoring services for students preparing for the SAT exam. Our tutors are experienced and ready to help your child succeed."
Rationale: Zero specifics about learning outcomes. "Experienced tutors" and "help succeed" could describe any tutoring company in existence. No learning transformation articulated — no numbers, no methodology, no sense of what the student will gain.
Score: 2""",
    "cta_effectiveness": """Example 1 (Score: 9):
Ad CTA: "Book a Free Consultation" (supporting context: diagnostic assessment, custom study plan)
Rationale: Lowers the barrier to zero cost. "Free consultation" frames the next step as getting expert advice, not making a purchase. Supported by the ad's diagnostic-first positioning — the consultation IS the value. Natural next step that doesn't feel salesy.
Score: 9

Example 2 (Score: 3):
Ad CTA: "Sign Up" (supporting context: generic tutoring services description)
Rationale: "Sign Up" is transactional — it implies commitment without establishing value. No indication of what happens after signing up. No risk reduction. Combined with the generic ad copy, there's no compelling reason to take this action over doing nothing.
Score: 3""",
    "brand_voice": f"""Example 1 (Score: 9):
Ad: "Why do 94% of {THEME.brand_name} SAT students recommend us? Because we don't believe in cookie-cutter prep. Every session is 1-on-1 with an expert tutor who adapts to how your child learns. Math anxiety? We address it. Reading comprehension gaps? We target them."
Rationale: Perfectly captures {THEME.brand_name}'s voice — confident but not arrogant, warm but professional. "Cookie-cutter prep" is natural language that shows personality. The Q&A about specific struggles (math anxiety, reading gaps) demonstrates genuine understanding. Could only be {THEME.brand_name}.
Score: 9

Example 2 (Score: 2):
Ad: "DON'T LET YOUR CHILD FAIL THE SAT! The SAT is the most important test of their life and a bad score could ruin their future."
Rationale: Fear-based manipulation that directly contradicts {THEME.brand_name}'s supportive brand. ALL CAPS screams desperation. "Ruin their future" is anxiety-inducing hyperbole that would damage brand trust. This is the opposite of the encouraging, partnership-oriented voice {THEME.brand_name} cultivates.
Score: 2""",
    "student_empathy": """Example 1 (Score: 9):
Ad: "Your SAT score opens doors -- to your dream school, to scholarships, to opportunities. Don't leave it to chance. Whether you're aiming for 1400+ or trying to crack 1200, we meet you where you are and get you where you want to be."
Rationale: Connects SAT scores to tangible life outcomes (dream school, scholarships) without manipulation. The inclusive range (1400+ to 1200) makes every student feel seen. "Meet you where you are" is emotionally warm and reduces anxiety about being judged. Aspirational without being pressure-heavy.
Score: 9

Example 2 (Score: 2):
Ad: "Every year thousands of students take the SAT. The SAT was created by the College Board and is used by colleges for admissions decisions. A good SAT score can help with college applications."
Rationale: Emotionally dead. States facts that everyone already knows instead of connecting with feelings. No acknowledgment of the parent's concern or the student's ambition. Reads like a Wikipedia entry, not communication from a company that cares about its students.
Score: 2""",
    "pedagogical_integrity": f"""Example 1 (Score: 9):
Ad: "Every {THEME.brand_name} student starts with a diagnostic assessment that maps their unique strengths and learning gaps. From there, your tutor builds a mastery-based plan — progressing through concepts only when true understanding is demonstrated, not just memorized. Because real SAT improvement comes from deep learning, not shortcuts."
Rationale: Embodies pedagogical best practices — diagnostic assessment, mastery-based progression, emphasis on understanding over memorization. Explicitly rejects shortcuts. Frames improvement as the result of genuine learning. This is education marketing done right.
Score: 9

Example 2 (Score: 2):
Ad: "Guaranteed 200-point SAT score increase or your money back! Our secret formula cracks the SAT code in just 3 sessions. No hard work required — just learn our tricks and watch your score skyrocket!"
Rationale: Every claim violates pedagogical integrity. Guaranteed outcomes ignore individual learning differences. "Secret formula" and "tricks" misrepresent how learning works. "No hard work required" is fundamentally dishonest about the learning process. This ad would damage trust with any education-savvy parent.
Score: 2""",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SINGLE_DIMENSION_PROMPT = f"""You are an expert advertising evaluator assessing a Facebook/Instagram ad for {THEME.brand_name} SAT test prep.

You are evaluating the dimension: {{dimension_name}}

RUBRIC:
{{rubric_text}}

FEW-SHOT EXAMPLES:
{{examples}}

AD TO EVALUATE:
Primary text: {{primary_text}}
Headline: {{headline}}
Description: {{description}}
CTA: {{cta_button}}

INSTRUCTIONS:
1. First, explain your reasoning about this ad's {{dimension_name}} in 2-3 sentences. Be specific about what works or doesn't work. Reference concrete elements of the ad.
2. Then provide your score (1-10) based strictly on the rubric above.
3. Be rigorous. A score of 7+ means the ad is genuinely strong on this dimension. A score of 5-6 means adequate but unremarkable. Below 5 means clear problems.
4. Do NOT inflate scores. Generic, vague, or templated ads should score 4-6 at most. Only specific, compelling, differentiated ads deserve 7+."""

ALL_DIMENSIONS_PROMPT = f"""You are an expert advertising evaluator assessing a Facebook/Instagram ad for {THEME.brand_name} SAT test prep.

Evaluate this ad across ALL 6 dimensions below. For EACH dimension, first explain your reasoning in 2-3 sentences, then provide a score (1-10).

Be rigorous. Generic or vague ads should score 4-6 at most. Only specific, compelling, differentiated ads deserve 7+. Do NOT inflate scores.

DIMENSIONS AND RUBRICS:

{{combined_rubrics}}

FEW-SHOT EXAMPLES (one per dimension):

{{combined_examples}}

AD TO EVALUATE:
Primary text: {{primary_text}}
Headline: {{headline}}
Description: {{description}}
CTA: {{cta_button}}

For each dimension, provide your rationale FIRST, then your score."""

# ---------------------------------------------------------------------------
# Response schemas for structured output
# ---------------------------------------------------------------------------

SINGLE_DIMENSION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": "2-3 sentence explanation of the reasoning behind the score",
        },
        "score": {
            "type": "number",
            "description": "Score from 1-10 based on the rubric",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this score from 0.0 to 1.0",
        },
    },
    "required": ["rationale", "score", "confidence"],
}

ALL_DIMENSIONS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        dim: {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": f"2-3 sentence reasoning for {dim} score",
                },
                "score": {
                    "type": "number",
                    "description": f"Score from 1-10 for {dim}",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in this score from 0.0 to 1.0",
                },
            },
            "required": ["rationale", "score", "confidence"],
        }
        for dim in DIMENSIONS
    },
    "required": DIMENSIONS,
}


# ---------------------------------------------------------------------------
# Helper: build prompts from templates
# ---------------------------------------------------------------------------


def build_single_dimension_prompt(
    dimension: str,
    primary_text: str,
    headline: str,
    description: str,
    cta_button: str,
) -> str:
    """Build a prompt for evaluating a single dimension."""
    return SINGLE_DIMENSION_PROMPT.format(
        dimension_name=dimension,
        rubric_text=RUBRICS[dimension],
        examples=FEW_SHOT_EXAMPLES[dimension],
        primary_text=primary_text,
        headline=headline,
        description=description,
        cta_button=cta_button,
    )


def build_all_dimensions_prompt(
    primary_text: str,
    headline: str,
    description: str,
    cta_button: str,
) -> str:
    """Build a prompt for evaluating all dimensions in one call."""
    combined_rubrics = "\n\n".join(f"--- {dim.upper()} ---\n{RUBRICS[dim]}" for dim in DIMENSIONS)

    # Use abbreviated examples for all-dimensions to manage prompt length
    combined_examples = "\n\n".join(
        f"--- {dim.upper()} ---\n{FEW_SHOT_EXAMPLES[dim]}" for dim in DIMENSIONS
    )

    return ALL_DIMENSIONS_PROMPT.format(
        combined_rubrics=combined_rubrics,
        combined_examples=combined_examples,
        primary_text=primary_text,
        headline=headline,
        description=description,
        cta_button=cta_button,
    )
