"""A/B visual variant generation for AdCraft.

Generates 2-3 visual variants per qualifying ad with different creative
approaches (lifestyle, product-focused, emotional/aspirational). Each
variant runs through the image generation escalation ladder independently.
Best-scoring variant is selected by visual eval weighted average.
"""

from __future__ import annotations

import io
import uuid

from PIL import Image as PILImage

from src.decisions.logger import log_decision
from src.evaluate.engine import EvaluationEngine
from src.evaluate.visual_rubrics import VISUAL_DIMENSION_WEIGHTS
from src.generate.image_engine import ImageGenerationEngine
from src.generate.visual_prompt import VisualPromptGenerator
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.creative import ImageResult
from src.models.evaluation import DimensionScore
from src.theme import THEME

VARIANT_TYPES: dict[str, str] = {
    "lifestyle": (
        "Creative direction: warm, authentic scene of a student or parent "
        "in a natural setting — kitchen table, library, dorm room. "
        "Emphasize human connection and real moments."
    ),
    "product": (
        "Creative direction: clean, focused composition featuring the "
        f"{THEME.brand_name} platform — laptop screen, app interface, branded "
        "materials. Minimal background, professional lighting."
    ),
    "emotional": (
        "Creative direction: abstract/aspirational imagery capturing the "
        "ad's emotional hook — celebration, confidence, achievement, "
        "breakthrough. Can use metaphorical visuals (opening doors, "
        "climbing, light breaking through)."
    ),
}


def _compute_visual_weighted_average(scores: list[DimensionScore]) -> float:
    """Compute weighted average from visual dimension scores."""
    return round(
        sum(s.score * VISUAL_DIMENSION_WEIGHTS[s.dimension] for s in scores),
        4,
    )


class VariantGenerator:
    """Orchestrates 2-3 visual variants per ad with different creative approaches.

    Each variant gets its own modified VisualBrief, runs through the
    ImageGenerationEngine escalation ladder independently, and is scored
    by the EvaluationEngine's visual eval. Best variant by weighted average
    is returned first.
    """

    def __init__(
        self,
        image_engine: ImageGenerationEngine,
        eval_engine: EvaluationEngine,
        prompt_generator: VisualPromptGenerator,
    ) -> None:
        self._image_engine = image_engine
        self._eval_engine = eval_engine
        self._prompt_generator = prompt_generator

        log_decision(
            "variant_generator",
            "engine_init",
            "VariantGenerator initialized with injected engines and prompt generator",
            {"variant_types": list(VARIANT_TYPES.keys())},
        )

    def generate_variants(
        self,
        ad_copy: AdCopy,
        brief: AdBrief,
        num_variants: int = 3,
    ) -> list[ImageResult]:
        """Generate visual variants and return them sorted best-first.

        Args:
            ad_copy: Approved ad copy to create imagery for.
            brief: Original ad brief with audience/tone context.
            num_variants: Number of variants to generate (capped at len(VARIANT_TYPES)).

        Returns:
            List of ImageResults sorted by visual eval score descending.
            Empty list if all variants fail (caller handles text-only fallback).
        """
        variant_group_id = str(uuid.uuid4())
        effective_num = min(num_variants, len(VARIANT_TYPES))
        variant_type_names = list(VARIANT_TYPES.keys())[:effective_num]

        log_decision(
            "variant_generator",
            "variant_generation_start",
            f"Starting variant generation for ad={ad_copy.id}, "
            f"group={variant_group_id}, variants={effective_num}",
            {
                "ad_id": ad_copy.id,
                "variant_group_id": variant_group_id,
                "num_variants": effective_num,
                "variant_types": variant_type_names,
            },
        )

        results: list[tuple[ImageResult, float]] = []
        failed_count = 0

        for variant_type in variant_type_names:
            modifier = VARIANT_TYPES[variant_type]

            # Generate base visual brief from ad copy + brief
            base_brief = self._prompt_generator.generate(ad_copy, brief)

            # Prepend creative direction modifier to the prompt
            modified_brief = base_brief.model_copy(
                update={"prompt": f"{modifier}\n\n{base_brief.prompt}"}
            )

            log_decision(
                "variant_generator",
                "variant_prompt_modified",
                f"Modified prompt for variant_type={variant_type}: {modified_brief.prompt[:200]}",
                {
                    "variant_type": variant_type,
                    "prompt_snippet": modified_brief.prompt[:200],
                },
            )

            # Generate image through escalation ladder
            try:
                image_result = self._image_engine.generate_image(
                    modified_brief,
                    ad_id=ad_copy.id,
                    variant=variant_type,
                )
            except Exception as exc:
                failed_count += 1
                log_decision(
                    "variant_generator",
                    "variant_generation_failed",
                    f"Variant {variant_type} failed: {type(exc).__name__}: {exc}",
                    {
                        "variant_type": variant_type,
                        "error": str(exc),
                        "ad_id": ad_copy.id,
                    },
                )
                continue

            # Tag the result with variant metadata
            image_result.variant_group_id = variant_group_id
            image_result.variant_type = variant_type

            # Evaluate the visual
            pil_image = PILImage.open(io.BytesIO(image_result.image_bytes))
            scores = self._eval_engine.evaluate_visual(pil_image, ad_copy)
            weighted_avg = _compute_visual_weighted_average(scores)

            log_decision(
                "variant_generator",
                "variant_eval_complete",
                f"Variant {variant_type} scored {weighted_avg:.4f}",
                {
                    "variant_type": variant_type,
                    "weighted_average": weighted_avg,
                    "scores": {s.dimension: s.score for s in scores},
                },
            )

            results.append((image_result, weighted_avg))

        # Sort by score descending
        results.sort(key=lambda pair: pair[1], reverse=True)

        if results:
            winner_type = results[0][0].variant_type
            winner_score = results[0][1]
            runner_up_scores = [score for _, score in results[1:]]

            log_decision(
                "variant_generator",
                "variant_selected",
                f"Selected variant {winner_type} with score {winner_score:.4f} "
                f"from {len(results)} candidates",
                {
                    "winning_variant_type": winner_type,
                    "winning_score": winner_score,
                    "runner_up_scores": runner_up_scores,
                    "total_candidates": len(results),
                },
            )

        best_score = results[0][1] if results else 0.0

        log_decision(
            "variant_generator",
            "variant_generation_complete",
            f"Variant generation complete: group={variant_group_id}, "
            f"succeeded={len(results)}, failed={failed_count}, "
            f"best_score={best_score:.4f}",
            {
                "variant_group_id": variant_group_id,
                "total_succeeded": len(results),
                "total_failed": failed_count,
                "best_score": best_score,
            },
        )

        return [image_result for image_result, _ in results]
