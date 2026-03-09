"""Evaluation rubrics, prompts, and scoring constants for AdCraft.

Defines the five evaluation dimensions, their rubric text, CoT prompt
templates, few-shot examples, weights, and hard gate thresholds.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Dimension weights — must sum to 1.0
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: dict[str, float] = {
    "clarity": 0.25,
    "value_prop": 0.25,
    "cta_effectiveness": 0.20,
    "brand_voice": 0.15,
    "emotional_resonance": 0.15,
}

DIMENSIONS: list[str] = list(DIMENSION_WEIGHTS.keys())

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

PASSING_THRESHOLD: float = 6.5
BRAND_VOICE_HARD_GATE: int = 5

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
    "value_prop": """VALUE PROPOSITION — How compelling and differentiated is the offer?

1-3 (Poor): Generic claims with no specifics. Could be any tutoring company.
  No evidence, no results data, no differentiator from competitors.
  Example markers: "experienced tutors", "professional services", no numbers.

4-6 (Adequate): Has a value claim but it's vague or unsubstantiated.
  Mentions some benefit but doesn't prove it or distinguish from competitors.
  Example markers: "great results" without numbers, lists features without benefits.

7-8 (Strong): Specific, credible value claims backed by evidence.
  Clear differentiation — the reader understands why THIS service over alternatives.
  Example markers: specific score improvements, social proof numbers, named methodology.

9-10 (Exceptional): Irresistible value prop with multiple proof points.
  Combines specific results + social proof + unique methodology. Hard to say no.""",
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
    "brand_voice": """BRAND VOICE — Does this sound like Varsity Tutors?

Varsity Tutors' brand voice is: supportive, knowledgeable, encouraging, results-oriented,
warm but professional. They position themselves as a partner in the student's journey,
not a fear-based pressure seller. They're confident in their results without being arrogant.

1-3 (Poor): Completely off-brand. Fear-based, overly casual, corporate jargon, or generic.
  Could not be identified as Varsity Tutors. Damages brand perception.
  Example markers: ALL CAPS, "DON'T FAIL", "lol", "leverage synergies", no personality.

4-6 (Adequate): Somewhat neutral but not distinctly Varsity Tutors.
  Not offensive to the brand but interchangeable with any education company.
  Example markers: generic professional tone, no warmth, no personality, template-feeling.

7-8 (Strong): Recognizably Varsity Tutors. Supportive, knowledgeable, encouraging.
  Balances professionalism with warmth. Results-focused without being pushy.
  Example markers: specific to VT's approach (1-on-1, personalized, expert tutors).

9-10 (Exceptional): Could only be Varsity Tutors. Distinctive voice that builds brand equity.
  Perfect balance of warmth, expertise, and confidence. Memorable.""",
    "emotional_resonance": """EMOTIONAL RESONANCE — Does the ad connect with the target audience's feelings?

1-3 (Poor): Emotionally flat or emotionally manipulative.
  Either reads like a product spec with no feeling, or uses fear/guilt to manipulate.
  Example markers: "services available", pure feature lists, "RUIN THEIR FUTURE".

4-6 (Adequate): Attempts emotional connection but doesn't land.
  The feeling is there but surface-level or cliched.
  Example markers: generic "success" language, obvious emotional buttons without depth.

7-8 (Strong): Genuine emotional connection with the target audience.
  Understands and addresses real concerns (parent worry, student ambition) authentically.
  Example markers: specific pain points (score ceiling, test anxiety, access to opportunity).

9-10 (Exceptional): Deeply resonant — reader feels understood.
  Taps into core motivations (parent: wanting the best for their child; student: proving
  themselves) without manipulation. Leaves the reader emotionally moved to act.""",
}

# ---------------------------------------------------------------------------
# Few-shot examples per dimension (extracted from reference ads)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: dict[str, str] = {
    "clarity": """Example 1 (Score: 9):
Ad: "Your child's SAT score shouldn't be limited by access to great teaching. Varsity Tutors connects students with expert SAT tutors who've helped families like yours see an average 160-point score improvement. Our personalized 1-on-1 approach means your student gets a study plan built around their specific strengths and weaknesses -- not a one-size-fits-all program."
Rationale: Every sentence has a clear purpose — problem (access), solution (expert tutors + 160-point improvement), differentiator (personalized plans). No wasted words. The reader instantly knows what's offered and why it matters.
Score: 9

Example 2 (Score: 2):
Ad: "Varsity Tutors leverages proprietary adaptive learning technology and a curated network of credentialed educators to deliver optimized SAT preparation outcomes. Our pedagogical methodology synthesizes evidence-based instructional strategies with real-time performance analytics to maximize score trajectory."
Rationale: Jargon-laden and impenetrable. "Pedagogical methodology" and "score trajectory" are corporate speak that obscures the actual offer. A parent reading this would have no idea what they're buying or why it's better. Fails the 2-second scroll test completely.
Score: 2""",
    "value_prop": """Example 1 (Score: 9):
Ad: "The SAT is 3 months away. Is your student ready? Our expert tutors have helped over 3,000 students boost their scores by an average of 160 points. We start with a diagnostic assessment to find exactly where your child needs help -- then build a custom study plan that targets those gaps."
Rationale: Triple proof points — 3,000 students (social proof), 160-point average (specific result), diagnostic-first approach (unique methodology). Each claim is specific and credible. Clear differentiation from generic "we help with SAT" competitors.
Score: 9

Example 2 (Score: 2):
Ad: "Looking for SAT help? We offer tutoring services for students preparing for the SAT exam. Our tutors are experienced and ready to help your child succeed."
Rationale: Zero specifics. "Experienced tutors" and "help succeed" could describe any tutoring company in existence. No numbers, no proof, no methodology, no differentiation. The reader has no reason to choose this over any alternative.
Score: 2""",
    "cta_effectiveness": """Example 1 (Score: 9):
Ad CTA: "Book a Free Consultation" (supporting context: diagnostic assessment, custom study plan)
Rationale: Lowers the barrier to zero cost. "Free consultation" frames the next step as getting expert advice, not making a purchase. Supported by the ad's diagnostic-first positioning — the consultation IS the value. Natural next step that doesn't feel salesy.
Score: 9

Example 2 (Score: 3):
Ad CTA: "Sign Up" (supporting context: generic tutoring services description)
Rationale: "Sign Up" is transactional — it implies commitment without establishing value. No indication of what happens after signing up. No risk reduction. Combined with the generic ad copy, there's no compelling reason to take this action over doing nothing.
Score: 3""",
    "brand_voice": """Example 1 (Score: 9):
Ad: "Why do 94% of Varsity Tutors SAT students recommend us? Because we don't believe in cookie-cutter prep. Every session is 1-on-1 with an expert tutor who adapts to how your child learns. Math anxiety? We address it. Reading comprehension gaps? We target them."
Rationale: Perfectly captures Varsity Tutors' voice — confident but not arrogant, warm but professional. "Cookie-cutter prep" is natural language that shows personality. The Q&A about specific struggles (math anxiety, reading gaps) demonstrates genuine understanding. Could only be Varsity Tutors.
Score: 9

Example 2 (Score: 2):
Ad: "DON'T LET YOUR CHILD FAIL THE SAT! The SAT is the most important test of their life and a bad score could ruin their future."
Rationale: Fear-based manipulation that directly contradicts Varsity Tutors' supportive brand. ALL CAPS screams desperation. "Ruin their future" is anxiety-inducing hyperbole that would damage brand trust. This is the opposite of the encouraging, partnership-oriented voice VT cultivates.
Score: 2""",
    "emotional_resonance": """Example 1 (Score: 9):
Ad: "Your SAT score opens doors -- to your dream school, to scholarships, to opportunities. Don't leave it to chance. Whether you're aiming for 1400+ or trying to crack 1200, we meet you where you are and get you where you want to be."
Rationale: Connects SAT scores to tangible life outcomes (dream school, scholarships) without manipulation. The inclusive range (1400+ to 1200) makes every student feel seen. "Meet you where you are" is emotionally warm and reduces anxiety about being judged. Aspirational without being pressure-heavy.
Score: 9

Example 2 (Score: 2):
Ad: "Every year thousands of students take the SAT. The SAT was created by the College Board and is used by colleges for admissions decisions. A good SAT score can help with college applications."
Rationale: Emotionally dead. States facts that everyone already knows instead of connecting with feelings. No acknowledgment of the parent's concern or the student's ambition. Reads like a Wikipedia entry, not communication from a company that cares about its students.
Score: 2""",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SINGLE_DIMENSION_PROMPT = """You are an expert advertising evaluator assessing a Facebook/Instagram ad for Varsity Tutors SAT test prep.

You are evaluating the dimension: {dimension_name}

RUBRIC:
{rubric_text}

FEW-SHOT EXAMPLES:
{examples}

AD TO EVALUATE:
Primary text: {primary_text}
Headline: {headline}
Description: {description}
CTA: {cta_button}

INSTRUCTIONS:
1. First, explain your reasoning about this ad's {dimension_name} in 2-3 sentences. Be specific about what works or doesn't work. Reference concrete elements of the ad.
2. Then provide your score (1-10) based strictly on the rubric above.
3. Be rigorous. A score of 7+ means the ad is genuinely strong on this dimension. A score of 5-6 means adequate but unremarkable. Below 5 means clear problems.
4. Do NOT inflate scores. Generic, vague, or templated ads should score 4-6 at most. Only specific, compelling, differentiated ads deserve 7+."""

ALL_DIMENSIONS_PROMPT = """You are an expert advertising evaluator assessing a Facebook/Instagram ad for Varsity Tutors SAT test prep.

Evaluate this ad across ALL 5 dimensions below. For EACH dimension, first explain your reasoning in 2-3 sentences, then provide a score (1-10).

Be rigorous. Generic or vague ads should score 4-6 at most. Only specific, compelling, differentiated ads deserve 7+. Do NOT inflate scores.

DIMENSIONS AND RUBRICS:

{combined_rubrics}

FEW-SHOT EXAMPLES (one per dimension):

{combined_examples}

AD TO EVALUATE:
Primary text: {primary_text}
Headline: {headline}
Description: {description}
CTA: {cta_button}

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
