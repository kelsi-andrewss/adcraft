"""Quality escalation image generation engine for AdCraft.

Implements a two-tier escalation ladder: Gemini 2.5 Flash Image (free tier)
generates first, escalating to Gemini 3 Pro Image Preview on failure.
Dynamic quality threshold gates entry into image generation.

Tenacity retry handles transient API errors within each model tier.
Escalation logic handles the tier switch between models.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.analytics.cost import IMAGE_PRICING
from src.db.queries import get_image_gen_threshold
from src.decisions.logger import log_decision
from src.evaluate.utils import gemini_retry, is_retriable
from src.models.creative import ImageResult, VisualBrief

FLASH_IMAGE_MODEL = "gemini-2.5-flash-image"
PRO_IMAGE_MODEL = "gemini-3-pro-image-preview"

IMAGE_COST_USD = IMAGE_PRICING

DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_IMAGE_SIZE = "1K"
IMAGES_DIR = Path("data/images")


class ImageGenerationError(Exception):
    """Raised when image generation fails after all retries and escalation."""


class ImageGenerationEngine:
    """Generates ad images with a two-tier quality escalation ladder.

    Flash Image generates first (free). On failure, escalates to Pro Image.
    Dynamic threshold from quality_snapshots gates whether image generation
    should happen at all.
    """

    def __init__(self, client: genai.Client | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self._client = genai.Client(api_key=api_key)

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        log_decision(
            "image_generator",
            "engine_init",
            f"ImageGenerationEngine initialized: flash={FLASH_IMAGE_MODEL}, pro={PRO_IMAGE_MODEL}",
            {"flash_model": FLASH_IMAGE_MODEL, "pro_model": PRO_IMAGE_MODEL},
        )

    def should_generate_image(self, weighted_avg: float, conn: sqlite3.Connection) -> bool:
        """Check whether the ad's quality score meets the dynamic threshold.

        Threshold = max(7.0, running_weighted_avg - 0.5) from the most
        recent quality snapshot, falling back to 7.0 with no data.
        """
        threshold = get_image_gen_threshold(conn)

        log_decision(
            "image_generator",
            "threshold_check",
            f"Image gen threshold check: score={weighted_avg:.2f}, threshold={threshold:.2f}, "
            f"eligible={weighted_avg >= threshold}",
            {
                "weighted_avg": weighted_avg,
                "threshold": threshold,
                "eligible": weighted_avg >= threshold,
            },
        )

        return weighted_avg >= threshold

    def generate_image(
        self,
        visual_brief: VisualBrief,
        ad_id: str,
        variant: str = "default",
    ) -> ImageResult:
        """Generate an image using the escalation ladder.

        Tries flash-image first. On failure, escalates to pro-image.
        Raises ImageGenerationError if both tiers fail.
        """
        log_decision(
            "image_generator",
            "generation_start",
            f"Starting image generation for ad={ad_id}, variant={variant}",
            {
                "ad_id": ad_id,
                "variant": variant,
                "aspect_ratio": visual_brief.aspect_ratio,
                "prompt_len": len(visual_brief.prompt),
            },
        )

        # Step 1: Try flash-image (free tier)
        try:
            result = self._generate_with_model(FLASH_IMAGE_MODEL, visual_brief, ad_id, variant)
            log_decision(
                "image_generator",
                "flash_success",
                f"Flash image generation succeeded for ad={ad_id}",
                {"ad_id": ad_id, "model": FLASH_IMAGE_MODEL, "cost": 0.0},
            )
            return result
        except Exception as flash_exc:
            log_decision(
                "image_generator",
                "escalation",
                f"Flash image failed ({type(flash_exc).__name__}), escalating to pro-image",
                {
                    "ad_id": ad_id,
                    "flash_error": str(flash_exc),
                    "escalating_to": PRO_IMAGE_MODEL,
                },
            )

        # Step 2: Escalate to pro-image
        try:
            result = self._generate_with_model(PRO_IMAGE_MODEL, visual_brief, ad_id, variant)
            log_decision(
                "image_generator",
                "pro_success",
                f"Pro image generation succeeded for ad={ad_id}",
                {
                    "ad_id": ad_id,
                    "model": PRO_IMAGE_MODEL,
                    "cost": IMAGE_COST_USD[PRO_IMAGE_MODEL],
                },
            )
            return result
        except Exception as pro_exc:
            log_decision(
                "image_generator",
                "generation_failed",
                f"Both image tiers failed for ad={ad_id}",
                {
                    "ad_id": ad_id,
                    "pro_error": str(pro_exc),
                },
            )
            raise ImageGenerationError(
                f"Image generation failed for ad={ad_id} after flash and pro attempts"
            ) from pro_exc

    @gemini_retry
    def _generate_with_model(
        self,
        model: str,
        visual_brief: VisualBrief,
        ad_id: str,
        variant: str,
    ) -> ImageResult:
        """Call the Gemini image generation API with tenacity retry.

        Handles transient errors (429, 500) within a single model tier.
        3 attempts with exponential backoff (2s, 4s, 8s).
        """
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=visual_brief.prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=visual_brief.aspect_ratio or DEFAULT_ASPECT_RATIO,
                        image_size=DEFAULT_IMAGE_SIZE,
                    ),
                ),
            )
        except APIError as exc:
            retriable = is_retriable(exc)
            log_decision(
                "image_generator",
                "api_retry" if retriable else "api_error",
                f"Image API call failed for model={model} ({exc.code} {exc.status}), "
                f"{'will retry' if retriable else 'non-retriable'}: {exc}",
                {
                    "model": model,
                    "ad_id": ad_id,
                    "error": str(exc),
                    "code": exc.code,
                    "retriable": retriable,
                },
            )
            raise
        except Exception as exc:
            log_decision(
                "image_generator",
                "api_error",
                f"Image API call failed for model={model} "
                f"(non-API error, will not retry): {type(exc).__name__}: {exc}",
                {"model": model, "ad_id": ad_id, "error": str(exc), "retriable": False},
            )
            raise

        # Extract image bytes from response parts
        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data
                break

        if image_bytes is None:
            raise ImageGenerationError(f"No image data in response from {model} for ad={ad_id}")

        file_path = self._save_image(image_bytes, ad_id, variant)
        cost = IMAGE_COST_USD.get(model, 0.0)

        log_decision(
            "image_generator",
            "image_saved",
            f"Image saved: path={file_path}, model={model}, cost=${cost:.4f}",
            {"file_path": file_path, "model": model, "cost": cost, "ad_id": ad_id},
        )

        return ImageResult(
            image_bytes=image_bytes,
            file_path=file_path,
            model_id=model,
            cost_usd=cost,
            generation_config={
                "aspect_ratio": visual_brief.aspect_ratio or DEFAULT_ASPECT_RATIO,
                "image_size": DEFAULT_IMAGE_SIZE,
                "negative_prompt": visual_brief.negative_prompt,
            },
        )

    def _save_image(self, image_bytes: bytes, ad_id: str, variant: str) -> str:
        """Save image bytes to data/images/{ad_id}_{variant}.png.

        Returns the relative path string for storage in the database.
        """
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{ad_id}_{variant}.png"
        file_path = IMAGES_DIR / filename
        file_path.write_bytes(image_bytes)
        return str(file_path)
