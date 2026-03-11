"""Batch pipeline for AdCraft — orchestrates generation/evaluation/iteration.

Processes a brief matrix sequentially with rate limiting, tracking pass/fail
counts and persisting quality snapshots after each batch. After text iteration
passes, ads flow through a 3-cycle visual iteration loop: generate variants,
composed-eval each, and regenerate the prompt from failure rationale on miss.
Wrapped by VisualCircuitBreaker and skippable via --no-images.

Entry point: python -m src.pipeline.main
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

from PIL import Image as PILImage

from src.briefs.seed_briefs import get_seed_briefs
from src.db.init_db import init_db
from src.db.queries import insert_quality_snapshot, update_ad_image
from src.decisions.logger import log_decision
from src.evaluate.composed import ComposedEvaluator
from src.evaluate.engine import EvaluationEngine
from src.generate.engine import GenerationEngine
from src.generate.image_engine import ImageGenerationEngine
from src.generate.variants import VariantGenerator
from src.generate.visual_prompt import VisualPromptGenerator
from src.iterate.controller import IterationController
from src.iterate.healing import SelfHealer
from src.iterate.visual_healing import VisualCircuitBreaker
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.creative import ImageResult

MAX_VISUAL_CYCLES: int = 3


@dataclass
class BatchResult:
    """Summary of a batch pipeline run."""

    total_briefs: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    total_cycles: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    errors: int = 0
    scores: list[float] = field(default_factory=list)
    images_generated: int = 0
    visual_cost_usd: float = 0.0
    visual_failures: int = 0
    circuit_breaker_tripped: bool = False


class RateLimiter:
    """Token-bucket rate limiter with exponential backoff and jitter."""

    def __init__(self, rpm_limit: int = 12, rpd_limit: int = 800) -> None:
        self._rpm_limit = rpm_limit
        self._rpd_limit = rpd_limit
        self._call_timestamps: deque[float] = deque()
        self._daily_count = 0
        self._backoff_seconds = 1.0

    def acquire(self) -> None:
        """Block until a request slot is available within rate limits."""
        now = time.time()

        # Prune timestamps older than 60 seconds
        while self._call_timestamps and self._call_timestamps[0] < now - 60:
            self._call_timestamps.popleft()

        # Check RPM
        if len(self._call_timestamps) >= self._rpm_limit:
            wait_until = self._call_timestamps[0] + 60
            sleep_time = max(0, wait_until - now)
            # Add jitter: 10-50% of sleep time
            jitter = sleep_time * random.uniform(0.1, 0.5)
            total_sleep = sleep_time + jitter

            log_decision(
                "pipeline",
                "rate_limit_throttle",
                f"RPM limit ({self._rpm_limit}) reached. "
                f"Sleeping {total_sleep:.1f}s (backoff={self._backoff_seconds:.1f}s)",
                {
                    "rpm_limit": self._rpm_limit,
                    "current_rpm": len(self._call_timestamps),
                    "sleep_seconds": total_sleep,
                    "backoff": self._backoff_seconds,
                },
            )

            time.sleep(total_sleep)
            self._backoff_seconds = min(self._backoff_seconds * 2, 60.0)

            # Re-prune after sleep
            now = time.time()
            while self._call_timestamps and self._call_timestamps[0] < now - 60:
                self._call_timestamps.popleft()
        else:
            # Reset backoff on successful acquisition without waiting
            self._backoff_seconds = 1.0

        # Check RPD
        if self._daily_count >= self._rpd_limit:
            log_decision(
                "pipeline",
                "daily_limit_reached",
                f"Daily request limit ({self._rpd_limit}) reached. Cannot proceed.",
                {"rpd_limit": self._rpd_limit, "daily_count": self._daily_count},
            )
            raise RuntimeError(f"Daily request limit ({self._rpd_limit}) reached")

        self._call_timestamps.append(time.time())
        self._daily_count += 1


class BatchPipeline:
    """Orchestrates batch ad generation with iteration and rate limiting."""

    def __init__(
        self,
        db_path: str = "data/ads.db",
        rpm_limit: int = 12,
        rpd_limit: int = 800,
        *,
        generator: GenerationEngine | None = None,
        evaluator: EvaluationEngine | None = None,
        image_engine: ImageGenerationEngine | None = None,
        composed_evaluator: ComposedEvaluator | None = None,
        prompt_generator: VisualPromptGenerator | None = None,
        variant_generator: VariantGenerator | None = None,
        circuit_breaker: VisualCircuitBreaker | None = None,
    ) -> None:
        self._conn = init_db(db_path)
        self._generator = generator or GenerationEngine()
        self._evaluator = evaluator or EvaluationEngine()
        self._healer = SelfHealer()
        self._controller = IterationController(
            self._generator, self._evaluator, self._healer, self._conn
        )
        self._rate_limiter = RateLimiter(rpm_limit=rpm_limit, rpd_limit=rpd_limit)

        # Visual pipeline components
        self._image_engine = image_engine or ImageGenerationEngine()
        self._composed_evaluator = composed_evaluator or ComposedEvaluator()
        self._prompt_generator = prompt_generator or VisualPromptGenerator()
        self._variant_generator = variant_generator or VariantGenerator(
            self._image_engine, self._evaluator, self._prompt_generator
        )
        self._circuit_breaker = circuit_breaker or VisualCircuitBreaker(batch_failure_threshold=0.5)

        log_decision(
            "pipeline",
            "pipeline_init",
            f"BatchPipeline initialized: db_path={db_path}, rpm={rpm_limit}, rpd={rpd_limit}, "
            "visual_pipeline=enabled",
            {
                "db_path": db_path,
                "rpm_limit": rpm_limit,
                "rpd_limit": rpd_limit,
                "visual_pipeline": True,
            },
        )

    def generate_brief_matrix(self) -> list[AdBrief]:
        """Generate the brief matrix from seed briefs."""
        briefs = get_seed_briefs()
        log_decision(
            "pipeline",
            "brief_matrix_generated",
            f"Generated brief matrix with {len(briefs)} briefs",
            {"count": len(briefs)},
        )
        return briefs

    def _run_visual_pipeline(self, ad: AdCopy, brief: AdBrief) -> tuple[ImageResult | None, float]:
        """Run the visual iteration loop for a single ad.

        Up to MAX_VISUAL_CYCLES attempts: generate/regenerate a visual prompt,
        produce 2 variants, composed-eval each (best-first, early exit on pass).
        On failure the composed eval rationale feeds back into the next cycle's
        prompt regeneration. Any exception is caught and logged — visual failure
        never crashes the batch.

        Returns:
            Tuple of (best passing ImageResult or None, accumulated visual cost).
        """
        accumulated_cost = 0.0
        cycle_count = 0
        last_rationale: str | None = None

        try:
            while cycle_count < MAX_VISUAL_CYCLES:
                log_decision(
                    "pipeline",
                    "visual_cycle_start",
                    f"Ad '{ad.id}' visual cycle {cycle_count}/{MAX_VISUAL_CYCLES}",
                    {
                        "ad_id": ad.id,
                        "cycle": cycle_count,
                        "max_cycles": MAX_VISUAL_CYCLES,
                        "has_prior_rationale": last_rationale is not None,
                    },
                )

                # Rate limit before prompt generation
                self._rate_limiter.acquire()

                # Cycle 0: fresh prompt. Cycles 1+: regenerate from failure rationale.
                if cycle_count == 0:
                    visual_brief = self._prompt_generator.generate(ad, brief)
                else:
                    log_decision(
                        "pipeline",
                        "visual_regeneration_trigger",
                        f"Ad '{ad.id}' regenerating prompt from rationale: "
                        f"'{last_rationale[:200] if last_rationale else ''}'",
                        {
                            "ad_id": ad.id,
                            "cycle": cycle_count,
                            "rationale_snippet": (last_rationale[:200] if last_rationale else ""),
                        },
                    )
                    visual_brief = self._prompt_generator.regenerate(
                        ad,
                        brief,
                        last_rationale,  # type: ignore[arg-type]
                    )

                # Generate 2 variants using the (re)generated brief
                variants = self._variant_generator.generate_variants(
                    ad, brief, num_variants=2, visual_brief=visual_brief
                )

                # Accumulate cost for all generated variants
                for v in variants:
                    accumulated_cost += v.cost_usd

                if not variants:
                    log_decision(
                        "pipeline",
                        "visual_no_variants",
                        f"Ad '{ad.id}' cycle {cycle_count}: no variants generated",
                        {"ad_id": ad.id, "cycle": cycle_count},
                    )
                    cycle_count += 1
                    continue

                # Composed eval each variant (best-first, early exit on pass)
                for variant in variants:
                    pil_image = PILImage.open(
                        io.BytesIO(variant.image_bytes)
                        if variant.image_bytes
                        else variant.file_path
                    )

                    self._rate_limiter.acquire()

                    composed_result = self._composed_evaluator.evaluate_composed(pil_image, ad)

                    log_decision(
                        "pipeline",
                        "visual_composed_eval_result",
                        f"Ad '{ad.id}' cycle {cycle_count} variant "
                        f"'{variant.variant_type}': "
                        f"score={composed_result['composed_score']:.1f}, "
                        f"publishable={composed_result['publishable']}",
                        {
                            "ad_id": ad.id,
                            "cycle": cycle_count,
                            "variant_type": variant.variant_type,
                            "composed_score": composed_result["composed_score"],
                            "publishable": composed_result["publishable"],
                        },
                    )

                    if composed_result["publishable"]:
                        # Persist image to DB
                        update_ad_image(
                            self._conn,
                            ad.id,
                            image_path=variant.file_path,
                            visual_prompt=variant.generation_config.get("negative_prompt", ""),
                            image_model=variant.model_id,
                            image_cost_usd=variant.cost_usd,
                            variant_group_id=variant.variant_group_id,
                            variant_type=variant.variant_type,
                        )

                        log_decision(
                            "pipeline",
                            "visual_published",
                            f"Ad '{ad.id}' image published on cycle {cycle_count}: "
                            f"model={variant.model_id}, cost=${variant.cost_usd:.4f}, "
                            f"composed_score={composed_result['composed_score']:.1f}",
                            {
                                "ad_id": ad.id,
                                "cycle": cycle_count,
                                "model_id": variant.model_id,
                                "cost_usd": variant.cost_usd,
                                "composed_score": composed_result["composed_score"],
                                "variant_type": variant.variant_type,
                                "variant_group_id": variant.variant_group_id,
                                "accumulated_cost": accumulated_cost,
                            },
                        )

                        return variant, accumulated_cost

                    # Capture rationale from this variant's failure
                    last_rationale = composed_result["rationale"]

                # All variants failed this cycle
                log_decision(
                    "pipeline",
                    "visual_cycle_failed",
                    f"Ad '{ad.id}' cycle {cycle_count}: all variants failed composed eval",
                    {
                        "ad_id": ad.id,
                        "cycle": cycle_count,
                        "last_rationale_snippet": (last_rationale[:200] if last_rationale else ""),
                    },
                )
                cycle_count += 1

            # Budget exhausted — all cycles failed
            log_decision(
                "pipeline",
                "visual_budget_exhausted",
                f"Ad '{ad.id}' exhausted {MAX_VISUAL_CYCLES} visual cycles, "
                f"falling back to text-only (accumulated_cost=${accumulated_cost:.4f})",
                {
                    "ad_id": ad.id,
                    "max_cycles": MAX_VISUAL_CYCLES,
                    "accumulated_cost": accumulated_cost,
                },
            )
            return None, accumulated_cost

        except Exception as exc:
            log_decision(
                "pipeline",
                "visual_pipeline_error",
                f"Visual pipeline error for ad '{ad.id}': {type(exc).__name__}: {exc}",
                {
                    "ad_id": ad.id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "accumulated_cost": accumulated_cost,
                },
            )
            return None, accumulated_cost

    def run(self, briefs: list[AdBrief] | None = None, *, no_images: bool = False) -> BatchResult:
        """Run the batch pipeline over the given briefs.

        Each brief goes through the full iterate loop. One failure
        does not crash the batch. After text iteration passes, qualifying
        ads flow through the visual pipeline unless no_images is True.
        """
        if briefs is None:
            briefs = self.generate_brief_matrix()

        result = BatchResult(total_briefs=len(briefs))
        start_time = time.time()

        # Reset circuit breaker state for this batch
        self._circuit_breaker.reset_batch()

        log_decision(
            "pipeline",
            "batch_start",
            f"Starting batch of {len(briefs)} briefs (no_images={no_images})",
            {"total_briefs": len(briefs), "no_images": no_images},
        )

        for i, brief in enumerate(briefs, 1):
            label = f"{brief.audience_segment} x {brief.product_offer[:30]} x {brief.tone}"
            print(f"[{i}/{len(briefs)}] Processing: {label}")

            try:
                self._rate_limiter.acquire()
                ad, records = self._controller.iterate(brief)

                result.total_cycles += len(records) + 1  # +1 for initial gen

                if ad is not None:
                    result.passed += 1
                    # Get the final evaluation score
                    final_eval = self._evaluator.evaluate_iteration(ad)
                    result.scores.append(final_eval.weighted_average)
                    result.total_tokens += ad.token_count + final_eval.token_count
                    for rec in records:
                        result.total_tokens += int(rec.token_cost)

                    log_decision(
                        "pipeline",
                        "brief_passed",
                        f"Brief {i}/{len(briefs)} passed: {label} "
                        f"(score={final_eval.weighted_average:.2f}, "
                        f"cycles={len(records) + 1})",
                        {
                            "brief_index": i,
                            "score": final_eval.weighted_average,
                            "cycles": len(records) + 1,
                        },
                    )

                    # Visual pipeline — skip if --no-images or circuit breaker tripped
                    if not no_images:
                        if self._circuit_breaker.check_batch_health() == "halt":
                            result.circuit_breaker_tripped = True
                            log_decision(
                                "pipeline",
                                "visual_circuit_breaker_halt",
                                f"Circuit breaker halted image gen at brief {i}",
                                {"brief_index": i},
                            )
                        else:
                            image_result, visual_cost = self._run_visual_pipeline(ad, brief)
                            result.visual_cost_usd += visual_cost

                            if image_result is not None:
                                result.images_generated += 1
                                self._circuit_breaker.record_variant_attempt(
                                    ad.id,
                                    image_result.variant_type or "unknown",
                                    passed=True,
                                )
                            else:
                                result.visual_failures += 1
                                self._circuit_breaker.record_variant_attempt(
                                    ad.id, "unknown", passed=False
                                )

                            self._circuit_breaker.check_ad_status(ad.id)

                else:
                    result.failed += 1
                    log_decision(
                        "pipeline",
                        "brief_failed",
                        f"Brief {i}/{len(briefs)} failed after max cycles: {label}",
                        {"brief_index": i, "cycles": len(records) + 1},
                    )

            except Exception as exc:
                result.errors += 1
                result.failed += 1
                log_decision(
                    "pipeline",
                    "brief_error",
                    f"Brief {i}/{len(briefs)} error: {type(exc).__name__}: {exc}",
                    {
                        "brief_index": i,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(f"  ERROR: {exc}", file=sys.stderr)

        result.duration_seconds = time.time() - start_time
        result.avg_score = sum(result.scores) / len(result.scores) if result.scores else 0.0
        result.total_cost_usd += result.visual_cost_usd

        # Persist quality snapshot
        self._persist_quality_snapshot(result)

        log_decision(
            "pipeline",
            "batch_complete",
            f"Batch complete: {result.passed}/{result.total_briefs} passed, "
            f"avg_score={result.avg_score:.2f}, "
            f"duration={result.duration_seconds:.1f}s, "
            f"images={result.images_generated}, visual_cost=${result.visual_cost_usd:.2f}",
            {
                "passed": result.passed,
                "failed": result.failed,
                "errors": result.errors,
                "total": result.total_briefs,
                "avg_score": result.avg_score,
                "total_cycles": result.total_cycles,
                "duration_seconds": result.duration_seconds,
                "images_generated": result.images_generated,
                "visual_cost_usd": result.visual_cost_usd,
                "visual_failures": result.visual_failures,
                "circuit_breaker_tripped": result.circuit_breaker_tripped,
            },
        )

        return result

    def _persist_quality_snapshot(self, result: BatchResult) -> None:
        """Write a quality snapshot row to the database."""
        # Compute dimension averages from the batch (not available at this level
        # without re-querying, so we store what we have)
        insert_quality_snapshot(
            self._conn,
            cycle_number=result.total_briefs,
            avg_weighted_score=result.avg_score,
            ads_above_threshold=result.passed,
            total_ads=result.total_briefs,
            token_spend_usd=result.total_cost_usd,
            quality_per_dollar=(
                result.avg_score / result.total_cost_usd if result.total_cost_usd > 0 else None
            ),
        )


def main() -> None:
    """CLI entry point for the batch pipeline."""
    parser = argparse.ArgumentParser(description="AdCraft batch pipeline")
    parser.add_argument(
        "--briefs",
        type=int,
        default=None,
        help="Limit number of briefs to process",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=12,
        help="Rate limit: requests per minute (default 12)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/ads.db",
        help="Database path (default data/ads.db)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        default=False,
        help="Skip image generation (text-only pipeline)",
    )
    args = parser.parse_args()

    pipeline = BatchPipeline(db_path=args.db, rpm_limit=args.rpm)
    briefs = pipeline.generate_brief_matrix()

    if args.briefs is not None:
        briefs = briefs[: args.briefs]
        print(f"Limited to {len(briefs)} briefs")

    result = pipeline.run(briefs, no_images=args.no_images)

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"  Total briefs:  {result.total_briefs}")
    print(f"  Passed:        {result.passed}")
    print(f"  Failed:        {result.failed}")
    print(f"  Errors:        {result.errors}")
    print(f"  Avg score:     {result.avg_score:.2f}")
    print(f"  Total cycles:  {result.total_cycles}")
    print(f"  Images:        {result.images_generated}")
    print(f"  Visual cost:   ${result.visual_cost_usd:.2f}")
    print(f"  Visual fails:  {result.visual_failures}")
    print(f"  CB tripped:    {result.circuit_breaker_tripped}")
    print(f"  Duration:      {result.duration_seconds:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
