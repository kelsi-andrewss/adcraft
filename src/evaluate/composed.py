"""Composed ad evaluation — scores the complete ad unit (copy + image) as one.

Separate from individual text and visual scoring. This judges the ad as a whole:
does the copy + image combination work as a publishable Facebook/Instagram ad?

Single multimodal call to Gemini 2.5 Pro with the image and full ad copy.
Returns composed_score (1-10), rationale, and publishable boolean.
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image as PILImage

from src.decisions.logger import log_decision
from src.evaluate.utils import SAFETY_SETTINGS, gemini_retry, is_retriable
from src.models.ad import AdCopy

EVALUATOR_MODEL = "gemini-2.5-pro"
PUBLISHABLE_THRESHOLD = 7.0

COMPOSED_EVAL_PROMPT = (
    "You are an expert advertising creative director evaluating a complete\n"
    "Facebook/Instagram ad unit for an online learning platform.\n"
    "\n"
    "You are evaluating the COMPLETE AD — copy and image together as a single\n"
    "publishable unit. This is NOT about the image alone or the copy alone.\n"
    "It's about whether the combination works as a real ad that would perform\n"
    "on FB/IG.\n"
    "\n"
    "EVALUATION CRITERIA:\n"
    "\n"
    "1. MESSAGE COHERENCE — Does the image reinforce the copy's core message?\n"
    "   Does the viewer receive one unified message from image + headline +\n"
    "   primary text + CTA together? Or do the elements pull in different\n"
    "   directions, creating confusion?\n"
    "\n"
    "2. VISUAL-TEXT BALANCE — Is there appropriate interplay between the\n"
    "   visual and textual elements? Does the image carry visual weight\n"
    "   without overwhelming the copy? Does the copy complement the image\n"
    "   without being redundant? In a feed context, would a viewer's eye\n"
    "   naturally flow from image to headline to CTA?\n"
    "\n"
    "3. OVERALL AD PUBLISHABILITY — Would you approve this ad for a real\n"
    "   FB/IG campaign? Consider: scroll-stopping power,\n"
    "   brand alignment, professional polish, clarity of value proposition,\n"
    "   and CTA effectiveness as a complete unit.\n"
    "\n"
    "SCORING BANDS:\n"
    "\n"
    "1-3 (Reject): The ad unit doesn't work. Image contradicts the copy,\n"
    "   elements feel disconnected, or the combination looks unprofessional.\n"
    "   Would damage the brand if published. Clear deal-breakers present.\n"
    "\n"
    "4-6 (Needs Work): The ad is functional but unremarkable. Copy and image\n"
    "   are thematically related but don't amplify each other. Looks like a\n"
    "   generic educational ad — nothing distinctive about the brand. Would get\n"
    "   scrolled past without a second glance.\n"
    "\n"
    "7-8 (Publishable): The ad works as a cohesive unit. Image and copy\n"
    "   deliver a unified message. Brand-aligned, professional, and the CTA\n"
    "   feels natural. Would hold its own in a real FB/IG feed. Minor polish\n"
    "   opportunities but fundamentally sound.\n"
    "\n"
    "9-10 (Exceptional): The ad is scroll-stopping. Image and copy form an\n"
    "   inseparable unit where removing either element weakens the whole.\n"
    "   Distinctly branded, emotionally resonant, and the CTA feels\n"
    "   inevitable. Agency-quality creative.\n"
    "\n"
    "INSTRUCTIONS:\n"
    "The ad image and copy text follow this prompt.\n"
    "\n"
    "1. Describe what you see: the image, the headline, the primary text,\n"
    "   the CTA, and how they relate to each other visually and narratively.\n"
    "2. Assess each criterion above (message coherence, visual-text balance,\n"
    "   overall publishability) with specific references to the ad elements.\n"
    "3. Provide a single composed_score (1-10) reflecting the overall ad\n"
    "   unit quality.\n"
    "4. Be rigorous. A functional-but-generic ad is a 5-6, not a 7-8.\n"
    "   Only ads that genuinely work as publishable ad creatives\n"
    "   deserve 7+.\n"
    "\n"
    "Respond with JSON matching the schema provided."
)

COMPOSED_EVAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": (
                "Detailed assessment covering message coherence, visual-text "
                "balance, and overall publishability of the complete ad unit"
            ),
        },
        "composed_score": {
            "type": "number",
            "description": "Overall score from 1-10 for the complete ad unit",
        },
    },
    "required": ["rationale", "composed_score"],
}


class ComposedEvaluator:
    """Scores the complete ad unit (copy + image) as a single publishable piece."""

    def __init__(self, client: genai.Client | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self._client = genai.Client(api_key=api_key)
        self._model = EVALUATOR_MODEL

    def evaluate_composed(
        self,
        image: PILImage.Image,
        ad_copy: AdCopy,
    ) -> dict:
        """Score the complete ad unit (copy + image) in a single multimodal call.

        Args:
            image: PIL Image of the ad creative.
            ad_copy: The ad copy paired with this image.

        Returns:
            Dict with composed_score (float 1-10), rationale (str),
            publishable (bool, True if composed_score >= 7.0),
            and token_count (int).
        """
        log_decision(
            "composed_evaluator",
            "evaluate_composed_start",
            f"Scoring complete ad unit for ad '{ad_copy.id}' "
            f"(copy + image, single multimodal call)",
            {"ad_id": ad_copy.id},
        )

        ad_copy_text = (
            f"Ad copy:\n"
            f"Headline: {ad_copy.headline}\n"
            f"Primary text: {ad_copy.primary_text}\n"
            f"Description: {ad_copy.description}\n"
            f"CTA: {ad_copy.cta_button}"
        )

        contents: list = [COMPOSED_EVAL_PROMPT, image, ad_copy_text]
        parsed, token_count = self._call_gemini_multimodal(contents, COMPOSED_EVAL_SCHEMA)

        composed_score = float(parsed["composed_score"])
        rationale = parsed["rationale"]
        publishable = composed_score >= PUBLISHABLE_THRESHOLD

        log_decision(
            "composed_evaluator",
            "publishable_decision",
            f"Ad '{ad_copy.id}' composed_score={composed_score:.1f}, "
            f"{'PUBLISHABLE' if publishable else 'NOT PUBLISHABLE'} "
            f"(threshold={PUBLISHABLE_THRESHOLD})",
            {
                "ad_id": ad_copy.id,
                "composed_score": composed_score,
                "publishable": publishable,
                "threshold": PUBLISHABLE_THRESHOLD,
                "token_count": token_count,
            },
        )

        return {
            "composed_score": composed_score,
            "rationale": rationale,
            "publishable": publishable,
            "token_count": token_count,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @gemini_retry
    def _call_gemini_multimodal(self, contents: list, schema: dict) -> tuple[dict, int]:
        """Call Gemini with multimodal content and structured output.

        Returns (parsed_response_dict, total_token_count).
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    safety_settings=SAFETY_SETTINGS,
                    http_options=types.HttpOptions(timeout=180_000),
                ),
            )
        except APIError as exc:
            retriable = is_retriable(exc)
            log_decision(
                "composed_evaluator",
                "api_retry" if retriable else "api_error",
                f"Gemini multimodal call failed ({exc.code} {exc.status}), "
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
                "composed_evaluator",
                "api_error",
                f"Gemini multimodal call failed (non-API error, will not retry): "
                f"{type(exc).__name__}: {exc}",
                {"model": self._model, "error": str(exc), "retriable": False},
            )
            raise

        token_count = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            token_count = getattr(meta, "total_token_count", 0) or 0

        parsed = json.loads(response.text)
        return parsed, token_count
