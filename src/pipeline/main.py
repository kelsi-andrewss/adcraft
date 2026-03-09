"""Batch pipeline for AdCraft — orchestrates generation/evaluation/iteration.

Processes a brief matrix sequentially with rate limiting, tracking pass/fail
counts and persisting quality snapshots after each batch.

Entry point: python -m src.pipeline.main
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field

from src.briefs.seed_briefs import get_seed_briefs
from src.db.init_db import init_db
from src.db.queries import insert_quality_snapshot
from src.decisions.logger import log_decision
from src.evaluate.engine import EvaluationEngine
from src.generate.engine import GenerationEngine
from src.iterate.controller import IterationController
from src.iterate.healing import SelfHealer
from src.models.brief import AdBrief


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
    ) -> None:
        self._conn = init_db(db_path)
        self._generator = generator or GenerationEngine()
        self._evaluator = evaluator or EvaluationEngine()
        self._healer = SelfHealer()
        self._controller = IterationController(
            self._generator, self._evaluator, self._healer, self._conn
        )
        self._rate_limiter = RateLimiter(rpm_limit=rpm_limit, rpd_limit=rpd_limit)

        log_decision(
            "pipeline",
            "pipeline_init",
            f"BatchPipeline initialized: db_path={db_path}, rpm={rpm_limit}, rpd={rpd_limit}",
            {"db_path": db_path, "rpm_limit": rpm_limit, "rpd_limit": rpd_limit},
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

    def run(self, briefs: list[AdBrief] | None = None) -> BatchResult:
        """Run the batch pipeline over the given briefs.

        Each brief goes through the full iterate loop. One failure
        does not crash the batch.
        """
        if briefs is None:
            briefs = self.generate_brief_matrix()

        result = BatchResult(total_briefs=len(briefs))
        start_time = time.time()

        log_decision(
            "pipeline",
            "batch_start",
            f"Starting batch of {len(briefs)} briefs",
            {"total_briefs": len(briefs)},
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

        # Persist quality snapshot
        self._persist_quality_snapshot(result)

        log_decision(
            "pipeline",
            "batch_complete",
            f"Batch complete: {result.passed}/{result.total_briefs} passed, "
            f"avg_score={result.avg_score:.2f}, "
            f"duration={result.duration_seconds:.1f}s",
            {
                "passed": result.passed,
                "failed": result.failed,
                "errors": result.errors,
                "total": result.total_briefs,
                "avg_score": result.avg_score,
                "total_cycles": result.total_cycles,
                "duration_seconds": result.duration_seconds,
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
    args = parser.parse_args()

    pipeline = BatchPipeline(db_path=args.db, rpm_limit=args.rpm)
    briefs = pipeline.generate_brief_matrix()

    if args.briefs is not None:
        briefs = briefs[: args.briefs]
        print(f"Limited to {len(briefs)} briefs")

    result = pipeline.run(briefs)

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"  Total briefs:  {result.total_briefs}")
    print(f"  Passed:        {result.passed}")
    print(f"  Failed:        {result.failed}")
    print(f"  Errors:        {result.errors}")
    print(f"  Avg score:     {result.avg_score:.2f}")
    print(f"  Total cycles:  {result.total_cycles}")
    print(f"  Duration:      {result.duration_seconds:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
