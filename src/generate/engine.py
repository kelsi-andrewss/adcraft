"""Ad copy generation engine for AdCraft.

Uses Gemini 2.5 Flash with structured JSON output to generate ad copy
from an AdBrief. Includes brand voice guidelines, few-shot examples,
platform constraints, and retry logic for safety filter blocks.
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.decisions.logger import log_decision
from src.evaluate.utils import gemini_retry, is_retriable
from src.intel.analyzer import CompetitorPatterns
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.theme import THEME

GENERATOR_MODEL = "gemini-2.5-flash"
GENERATION_TEMPERATURE = 0.7

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
]

# Schema for structured output — matches AdCopy fields
GENERATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "primary_text": {
            "type": "string",
            "description": "Main ad body text (max 125 chars visible, up to 500 total)",
        },
        "headline": {
            "type": "string",
            "description": "Ad headline (max 40 characters)",
        },
        "description": {
            "type": "string",
            "description": "Ad description/link description (max 125 characters)",
        },
        "cta_button": {
            "type": "string",
            "description": "CTA button text. Must be one of: Learn More, Get Started, Sign Up, Book Now, Contact Us, Get Offer, Apply Now",
        },
    },
    "required": ["primary_text", "headline", "description", "cta_button"],
}

BRAND_VOICE_GUIDELINES = f"""{THEME.brand_name.upper()} BRAND VOICE:
- Supportive and encouraging — we're a partner in the student's journey
- Knowledgeable and confident — we have the expertise and results to prove it
- Warm but professional — approachable without being unprofessional
- Results-oriented — specific numbers and proof points, not vague claims
- Never fear-based — no "DON'T FAIL", no anxiety manipulation, no false urgency
- Authentic — real language that parents and students relate to, no corporate jargon"""

FEW_SHOT_EXAMPLES = f"""EXAMPLE 1 (Parent-focused, strong):
Primary text: "Your child's SAT score shouldn't be limited by access to great teaching. {THEME.brand_name} connects students with expert SAT tutors who've helped families like yours see an average 160-point score improvement. Our personalized 1-on-1 approach means your student gets a study plan built around their specific strengths and weaknesses -- not a one-size-fits-all program."
Headline: "Average 160-Point SAT Score Improvement"
Description: "Personalized 1-on-1 SAT tutoring from expert instructors. Join 3,000+ families who've seen real results."
CTA: "Get Started"

EXAMPLE 2 (Student-focused, strong):
Primary text: "You've put in the work. You've taken the practice tests. But are you scoring where you want to be? {THEME.brand_name} matches you with an SAT expert who's scored in the 99th percentile and knows exactly how to help you break through your score ceiling. Real tutors, real strategies, real results."
Headline: "Break Through Your SAT Score Ceiling"
Description: "Work with 99th-percentile SAT tutors who know every trick in the book. Personalized strategies for your target score."
CTA: "Find Your Tutor"
"""

PLATFORM_CONSTRAINTS = """FACEBOOK/INSTAGRAM AD CONSTRAINTS:
- Primary text: 125 characters visible in feed (up to 500 total, remainder behind "See More")
- Headline: max 40 characters for full display
- Description: max 125 characters
- CTA button: must be one of: Learn More, Get Started, Sign Up, Book Now, Contact Us, Get Offer, Apply Now
- Write for mobile-first — short paragraphs, scannable, punchy"""


class GenerationEngine:
    """Generates ad copy using Gemini 2.5 Flash with structured output."""

    def __init__(self, client: genai.Client | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self._client = genai.Client(api_key=api_key)
        self._model = GENERATOR_MODEL

        log_decision(
            "generator",
            "engine_init",
            f"GenerationEngine initialized: model={self._model}, temp={GENERATION_TEMPERATURE}",
            {"model": self._model, "temperature": GENERATION_TEMPERATURE},
        )

    def generate(
        self, brief: AdBrief, competitor_patterns: CompetitorPatterns | None = None
    ) -> AdCopy:
        """Generate ad copy from a brief. Returns a validated AdCopy instance.

        When competitor_patterns is provided, a competitive context section is
        injected into the prompt with top hook patterns, CTA buttons, and
        emotional angles from competitor analysis.
        """
        log_decision(
            "generator",
            "generation_start",
            f"Generating ad for audience='{brief.audience_segment}', "
            f"offer='{brief.product_offer}', goal='{brief.campaign_goal}'",
            {
                "audience": brief.audience_segment,
                "offer": brief.product_offer,
                "goal": brief.campaign_goal,
                "tone": brief.tone,
                "has_competitor_context": competitor_patterns is not None,
            },
        )

        if competitor_patterns is not None:
            brief = self._inject_competitor_context(brief, competitor_patterns)

        prompt = self._build_prompt(brief)
        raw, token_count = self._call_gemini(prompt)

        ad = AdCopy(
            primary_text=raw["primary_text"],
            headline=raw["headline"],
            description=raw["description"],
            cta_button=raw["cta_button"],
            model_id=self._model,
            generation_config={
                "temperature": GENERATION_TEMPERATURE,
                "safety_settings": "BLOCK_ONLY_HIGH",
            },
            token_count=token_count,
        )

        log_decision(
            "generator",
            "generation_complete",
            f"Generated ad: headline='{ad.headline[:50]}', tokens={token_count}",
            {
                "headline": ad.headline,
                "primary_text_len": len(ad.primary_text),
                "token_count": token_count,
                "pedagogical_context": "Pedagogical North Star principles applied",
            },
        )

        return ad

    def _inject_competitor_context(self, brief: AdBrief, patterns: CompetitorPatterns) -> AdBrief:
        """Build competitive context string and attach it to the brief."""
        lines = ["Competitive Intelligence (use as inspiration, do NOT copy):"]

        if patterns.top_hooks:
            hook_strs = [f"{hook} ({count} ads)" for hook, count in patterns.top_hooks]
            lines.append(f"- Top hook patterns in category: {', '.join(hook_strs)}")

        if patterns.top_angles:
            angle_strs = [f"{angle} ({count} ads)" for angle, count in patterns.top_angles]
            lines.append(f"- Emotional angles that resonate: {', '.join(angle_strs)}")

        if patterns.cta_buttons:
            lines.append(f"- Common CTA buttons: {', '.join(patterns.cta_buttons)}")

        if patterns.sample_headlines:
            lines.append("- Competitor headline examples:")
            for hl in patterns.sample_headlines:
                lines.append(f'  * "{hl}"')

        competitive_text = "\n".join(lines)

        log_decision(
            "generator",
            "competitor_context_injected",
            f"Injected competitive context: {len(patterns.top_hooks)} hooks, "
            f"{len(patterns.top_angles)} angles, {len(patterns.cta_buttons)} CTAs",
            {
                "top_hooks": [h for h, _ in patterns.top_hooks],
                "top_angles": [a for a, _ in patterns.top_angles],
            },
        )

        return brief.model_copy(update={"competitive_context": competitive_text})

    def _build_prompt(self, brief: AdBrief) -> str:
        """Construct prompt with THEME.brand_name, PEDAGOGICAL_NORTH_STAR section,
        brand voice guidelines, and few-shot examples."""
        competitive_section = ""
        if brief.competitive_context:
            competitive_section = f"\nCOMPETITIVE CONTEXT:\n{brief.competitive_context}\n"

        pedagogical_north_star = f"""PEDAGOGICAL NORTH STAR -- {THEME.brand_name}'s Pedagogical Principles:
- Personalization at Scale: Every learner's path is unique. Ads must reflect individualized learning, not one-size-fits-all.
- Expert-Led Instruction: Tutors are subject matter experts and skilled educators. Ads should convey credentialed expertise.
- Active Learning: Learning is a process, not a transaction. Ads should frame education as a journey of growth, not a product to purchase.
- Never promise outcomes without effort. No "guaranteed scores" or "learn tricks." Frame improvement as the result of guided, personalized work."""

        return f"""You are an expert advertising copywriter for {THEME.brand_name}, creating a Facebook/Instagram ad for SAT test prep.

{BRAND_VOICE_GUIDELINES}

{pedagogical_north_star}

{PLATFORM_CONSTRAINTS}

{FEW_SHOT_EXAMPLES}

AD BRIEF:
- Target audience: {brief.audience_segment}
- Product/offer: {brief.product_offer}
- Campaign goal: {brief.campaign_goal}
- Tone: {brief.tone}
{competitive_section}
INSTRUCTIONS:
1. Write a compelling ad that speaks directly to the target audience.
2. Include specific, credible claims (score improvements, student counts, tutor qualifications).
3. Make every word count — mobile users scroll fast.
4. The CTA should feel like a natural next step, not a hard sell.
5. Stay on-brand: supportive, knowledgeable, warm, results-oriented.
6. Do NOT use fear-based messaging, ALL CAPS, or false urgency.
7. Align with the Pedagogical North Star — no guaranteed outcomes or shortcut promises.

Generate the ad copy now."""

    def _call_gemini(self, prompt: str) -> tuple[dict, int]:
        """Call Gemini with structured output and retry logic.

        Returns (parsed_response_dict, total_token_count).
        """
        return self._call_gemini_with_retry(prompt)

    @gemini_retry
    def _call_gemini_with_retry(self, prompt: str) -> tuple[dict, int]:
        """Inner call with gemini_retry decorator.

        Retries only transient APIErrors (408/429/500/502/503).
        Non-transient APIErrors and non-API exceptions propagate immediately.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=GENERATION_SCHEMA,
                    temperature=GENERATION_TEMPERATURE,
                    safety_settings=SAFETY_SETTINGS,
                ),
            )
        except APIError as exc:
            retriable = is_retriable(exc)
            log_decision(
                "generator",
                "api_retry" if retriable else "api_error",
                f"Gemini call failed ({exc.code}), {'will retry' if retriable else 'non-retriable'}: {exc}",
                {"model": self._model, "error": str(exc), "code": exc.code, "retriable": retriable},
            )
            raise

        token_count = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            token_count = getattr(meta, "total_token_count", 0) or 0

        parsed = json.loads(response.text)
        return parsed, token_count
