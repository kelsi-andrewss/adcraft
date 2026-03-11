"""Visual prompt generator for AdCraft.

Uses Gemini 2.5 Flash to synthesize image generation prompts from approved
ad copy + brand style guide. Extracts the emotional hook, key visual elements,
and brand constraints. Outputs a structured VisualBrief.
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.decisions.logger import log_decision
from src.evaluate.utils import gemini_retry, is_retriable
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.creative import VisualBrief
from src.theme import THEME

VISUAL_PROMPT_MODEL = "gemini-2.5-flash"

PLACEMENT_ASPECT_RATIOS: dict[str, str] = {
    "feed": "1:1",
    "stories": "9:16",
    "banner": "16:9",
}

# Schema for structured output — only LLM-generated fields.
# aspect_ratio, resolution, style_refs, placement are deterministic and set by code.
VISUAL_PROMPT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "Detailed image generation prompt capturing "
                "emotional hook, visual elements, and brand style"
            ),
        },
        "negative_prompt": {
            "type": "string",
            "description": "Negative prompt listing visual elements to avoid",
        },
    },
    "required": ["prompt", "negative_prompt"],
}

SYNTHESIS_PROMPT_TEMPLATE = """\
You are a visual creative director designing an image to accompany a Facebook/Instagram ad.

APPROVED AD COPY:
- Headline: {{headline}}
- Primary text: {{primary_text}}
- Description: {{description}}
- CTA: {{cta_button}}

AD BRIEF CONTEXT:
- Target audience: {{audience_segment}}
- Product/offer: {{product_offer}}
- Campaign goal: {{campaign_goal}}
- Tone: {{tone}}
- Placement: {{placement}}

BRAND VISUAL CONSTRAINTS:
- Color palette: {primary_color}, {secondary_color}, {accent_color} brand colors, \
{text_color} text
- Lighting: warm, inviting, natural lighting
- People: {people_descriptors}
- Aesthetic: {visual_tone}
- Setting: {setting_descriptors}

YOUR TASK:
1. Extract the emotional hook from the ad copy (what feeling should the image evoke?)
2. Identify key visual elements implied by the copy \
(e.g., "score improvement" -> celebratory imagery, \
"expert tutors" -> mentoring scene)
3. Compose a detailed image generation prompt that reinforces the ad's message
4. Generate a negative prompt to prevent off-brand imagery

NEGATIVE PROMPT MUST INCLUDE:
{negative_constraints}

Generate the visual prompt now.""".format(
    primary_color=THEME.primary_color,
    secondary_color=THEME.secondary_color,
    accent_color=THEME.accent_color,
    text_color=THEME.text_color,
    people_descriptors=", ".join(THEME.people_descriptors),
    visual_tone=", ".join(THEME.visual_tone),
    setting_descriptors=", ".join(THEME.setting_descriptors),
    negative_constraints="\n".join(f"- {c}" for c in THEME.negative_constraints),
)

REGENERATION_PROMPT_TEMPLATE = """\
You are a visual creative director redesigning an image for a Facebook/Instagram ad.
The previous visual prompt failed composed evaluation. Your job is to generate a \
completely new visual prompt that avoids the identified issues while preserving brand \
constraints and emotional alignment with the ad copy.

PREVIOUS FAILURE RATIONALE:
{{rationale}}

APPROVED AD COPY:
- Headline: {{headline}}
- Primary text: {{primary_text}}
- Description: {{description}}
- CTA: {{cta_button}}

AD BRIEF CONTEXT:
- Target audience: {{audience_segment}}
- Product/offer: {{product_offer}}
- Campaign goal: {{campaign_goal}}
- Tone: {{tone}}
- Placement: {{placement}}

BRAND VISUAL CONSTRAINTS:
- Color palette: {primary_color}, {secondary_color}, {accent_color} brand colors, \
{text_color} text
- Lighting: warm, inviting, natural lighting
- People: {people_descriptors}
- Aesthetic: {visual_tone}
- Setting: {setting_descriptors}

YOUR TASK:
1. Analyze the failure rationale — identify the specific visual shortcomings described
2. Design a completely new visual concept that addresses those shortcomings \
(do NOT patch the old prompt — start fresh)
3. Ensure the new prompt reinforces the ad copy's emotional message and campaign goal
4. Preserve all brand constraints (colors, aesthetic, people, setting)
5. Generate a negative prompt to prevent off-brand imagery and the previously identified issues

NEGATIVE PROMPT MUST INCLUDE:
{negative_constraints}

Generate the new visual prompt now.""".format(
    primary_color=THEME.primary_color,
    secondary_color=THEME.secondary_color,
    accent_color=THEME.accent_color,
    text_color=THEME.text_color,
    people_descriptors=", ".join(THEME.people_descriptors),
    visual_tone=", ".join(THEME.visual_tone),
    setting_descriptors=", ".join(THEME.setting_descriptors),
    negative_constraints="\n".join(f"- {c}" for c in THEME.negative_constraints),
)


class VisualPromptGenerator:
    """Synthesizes image generation prompts from approved ad copy.

    Uses Gemini 2.5 Flash to analyze ad copy and produce a structured
    VisualBrief with prompt text, negative prompts, and deterministic
    parameters (aspect ratio, resolution) based on placement.
    """

    def __init__(self, client: genai.Client | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self._client = genai.Client(api_key=api_key)
        self._model = VISUAL_PROMPT_MODEL

        log_decision(
            "visual_prompt",
            "engine_init",
            f"VisualPromptGenerator initialized: model={self._model}",
            {"model": self._model},
        )

    def generate(self, ad: AdCopy, brief: AdBrief, placement: str = "feed") -> VisualBrief:
        """Generate a visual prompt from approved ad copy and brief.

        Args:
            ad: Approved ad copy to create imagery for.
            brief: Original ad brief with audience/tone context.
            placement: Ad placement type (feed, stories, banner).

        Returns:
            VisualBrief with prompt, negative_prompt, and deterministic params.
        """
        log_decision(
            "visual_prompt",
            "generation_start",
            f"Generating visual prompt: headline='{ad.headline[:50]}', placement='{placement}'",
            {
                "headline": ad.headline,
                "placement": placement,
                "audience": brief.audience_segment,
            },
        )

        aspect_ratio = PLACEMENT_ASPECT_RATIOS.get(placement, "1:1")

        log_decision(
            "visual_prompt",
            "aspect_ratio_selection",
            f"Mapped placement '{placement}' to aspect_ratio '{aspect_ratio}'",
            {
                "placement": placement,
                "aspect_ratio": aspect_ratio,
                "is_default": placement not in PLACEMENT_ASPECT_RATIOS,
            },
        )

        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            headline=ad.headline,
            primary_text=ad.primary_text,
            description=ad.description,
            cta_button=ad.cta_button,
            audience_segment=brief.audience_segment,
            product_offer=brief.product_offer,
            campaign_goal=brief.campaign_goal,
            tone=brief.tone,
            placement=placement,
        )

        raw, token_count = self._call_gemini(prompt)

        visual_brief = VisualBrief(
            prompt=raw["prompt"],
            negative_prompt=raw["negative_prompt"],
            aspect_ratio=aspect_ratio,
            resolution="1K",
            placement=placement,
        )

        log_decision(
            "visual_prompt",
            "generation_complete",
            f"Visual prompt generated: {len(visual_brief.prompt)} chars, tokens={token_count}",
            {
                "prompt_length": len(visual_brief.prompt),
                "negative_prompt_length": len(visual_brief.negative_prompt),
                "token_count": token_count,
                "aspect_ratio": aspect_ratio,
            },
        )

        return visual_brief

    def regenerate(
        self,
        ad: AdCopy,
        brief: AdBrief,
        rationale: str,
        placement: str = "feed",
    ) -> VisualBrief:
        """Generate a fresh visual prompt informed by previous composed eval failure.

        Args:
            ad: Approved ad copy.
            brief: Original ad brief with audience/tone context.
            rationale: Most recent composed eval rationale (single string, not accumulated).
            placement: Ad placement type (feed, stories, banner).

        Returns:
            VisualBrief with new prompt, negative_prompt, and deterministic params.
        """
        log_decision(
            "visual_prompt",
            "regeneration_start",
            f"Regenerating visual prompt: headline='{ad.headline[:50]}', "
            f"placement='{placement}', rationale_len={len(rationale)}",
            {
                "headline": ad.headline,
                "placement": placement,
                "audience": brief.audience_segment,
                "rationale_length": len(rationale),
            },
        )

        aspect_ratio = PLACEMENT_ASPECT_RATIOS.get(placement, "1:1")

        log_decision(
            "visual_prompt",
            "aspect_ratio_selection",
            f"Mapped placement '{placement}' to aspect_ratio '{aspect_ratio}'",
            {
                "placement": placement,
                "aspect_ratio": aspect_ratio,
                "is_default": placement not in PLACEMENT_ASPECT_RATIOS,
            },
        )

        prompt = REGENERATION_PROMPT_TEMPLATE.format(
            rationale=rationale,
            headline=ad.headline,
            primary_text=ad.primary_text,
            description=ad.description,
            cta_button=ad.cta_button,
            audience_segment=brief.audience_segment,
            product_offer=brief.product_offer,
            campaign_goal=brief.campaign_goal,
            tone=brief.tone,
            placement=placement,
        )

        raw, token_count = self._call_gemini(prompt)

        visual_brief = VisualBrief(
            prompt=raw["prompt"],
            negative_prompt=raw["negative_prompt"],
            aspect_ratio=aspect_ratio,
            resolution="1K",
            placement=placement,
        )

        log_decision(
            "visual_prompt",
            "regeneration_complete",
            f"Visual prompt regenerated: {len(visual_brief.prompt)} chars, tokens={token_count}",
            {
                "prompt_length": len(visual_brief.prompt),
                "negative_prompt_length": len(visual_brief.negative_prompt),
                "token_count": token_count,
                "aspect_ratio": aspect_ratio,
            },
        )

        return visual_brief

    @gemini_retry
    def _call_gemini(self, prompt: str) -> tuple[dict, int]:
        """Call Gemini with structured output and retry logic.

        Returns (parsed_response_dict, total_token_count).
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=VISUAL_PROMPT_SCHEMA,
                ),
            )
        except APIError as exc:
            retriable = is_retriable(exc)
            log_decision(
                "visual_prompt",
                "api_retry" if retriable else "api_error",
                f"Gemini call failed ({exc.code} {exc.status}), "
                f"{'will retry' if retriable else 'non-retriable'}: {exc}",
                {
                    "model": self._model,
                    "error": str(exc),
                    "code": exc.code,
                    "retriable": retriable,
                },
            )
            raise
        except Exception as exc:
            log_decision(
                "visual_prompt",
                "api_error",
                f"Gemini call failed (non-API error, will not retry): {type(exc).__name__}: {exc}",
                {"model": self._model, "error": str(exc), "retriable": False},
            )
            raise

        token_count = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            token_count = getattr(meta, "total_token_count", 0) or 0

        parsed = json.loads(response.text)
        return parsed, token_count
