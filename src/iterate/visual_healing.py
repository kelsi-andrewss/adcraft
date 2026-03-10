"""Visual pipeline circuit breaker — three-tier failure handling for image generation.

Prevents silent waste of free tier quota against broken prompt templates
and surfaces systemic issues before they consume an entire batch.

Tier 1 (per-variant): retry cap — flash -> pro -> discard.
Tier 2 (per-ad): text-only fallback when all variants fail.
Tier 3 (per-batch): halt image generation when >=50% of ads fail.
"""

from __future__ import annotations

from src.decisions.logger import log_circuit_breaker_event, log_decision


class VisualCircuitBreaker:
    """Three-tier circuit breaker for visual ad generation failures."""

    def __init__(self, batch_failure_threshold: float = 0.5) -> None:
        self.batch_failure_threshold = batch_failure_threshold

        # Tier 1: per-variant attempt tracking — ad_id -> variant -> attempt_count
        self._variant_attempts: dict[str, dict[str, int]] = {}

        # Tier 2: per-ad candidate tracking — ad_id -> list of passing variant names
        self._ad_candidates: dict[str, list[str]] = {}

        # Tier 3: batch-level counters
        self._total_ads_attempted: int = 0
        self._ads_with_no_candidates: int = 0

    def record_variant_attempt(self, ad_id: str, variant: str, passed: bool) -> str:
        """Track a variant attempt and return the action to take.

        Returns:
            "accept" — variant passed, added to candidates.
            "retry_pro" — first failure, retry with pro model.
            "discard" — second failure, discard this variant.
        """
        if ad_id not in self._variant_attempts:
            self._variant_attempts[ad_id] = {}
        if ad_id not in self._ad_candidates:
            self._ad_candidates[ad_id] = []

        attempts = self._variant_attempts[ad_id]

        if passed:
            self._ad_candidates[ad_id].append(variant)
            log_decision(
                "visual_circuit_breaker",
                "variant_accepted",
                f"Variant '{variant}' for ad '{ad_id}' passed visual check",
                {"ad_id": ad_id, "variant": variant},
            )
            return "accept"

        # Failed — increment attempt count
        attempts[variant] = attempts.get(variant, 0) + 1
        attempt_count = attempts[variant]

        if attempt_count == 1:
            log_decision(
                "visual_circuit_breaker",
                "variant_retry_pro",
                f"Variant '{variant}' for ad '{ad_id}' failed on flash, retrying with pro model",
                {"ad_id": ad_id, "variant": variant, "attempt": attempt_count},
            )
            return "retry_pro"

        # attempt_count >= 2
        log_decision(
            "visual_circuit_breaker",
            "variant_discarded",
            f"Variant '{variant}' for ad '{ad_id}' failed on pro model, "
            f"discarding after {attempt_count} attempts",
            {"ad_id": ad_id, "variant": variant, "attempt": attempt_count},
        )
        return "discard"

    def check_ad_status(self, ad_id: str) -> str:
        """Check whether an ad has any passing visual variants.

        Returns:
            "has_candidates" — at least one variant passed.
            "text_only" — all variants exhausted, fall back to text-only.
        """
        candidates = self._ad_candidates.get(ad_id, [])

        if candidates:
            self._total_ads_attempted += 1
            log_decision(
                "visual_circuit_breaker",
                "ad_has_candidates",
                f"Ad '{ad_id}' has {len(candidates)} passing variant(s)",
                {"ad_id": ad_id, "candidates": candidates},
            )
            return "has_candidates"

        # No candidates — text-only fallback
        self._total_ads_attempted += 1
        self._ads_with_no_candidates += 1
        log_decision(
            "visual_circuit_breaker",
            "ad_text_only_fallback",
            f"Ad '{ad_id}' has no passing variants, falling back to text-only",
            {
                "ad_id": ad_id,
                "variants_attempted": list(self._variant_attempts.get(ad_id, {}).keys()),
            },
        )
        return "text_only"

    def check_batch_health(self) -> str:
        """Check batch-level failure rate and decide whether to continue.

        Returns:
            "continue" — failure rate below threshold (or no ads attempted).
            "halt" — failure rate at or above threshold, stop image generation.
        """
        if self._total_ads_attempted == 0:
            log_decision(
                "visual_circuit_breaker",
                "batch_health_no_ads",
                "No ads attempted yet, continuing",
                {"total_ads_attempted": 0},
            )
            return "continue"

        failure_rate = self._ads_with_no_candidates / self._total_ads_attempted

        if failure_rate < self.batch_failure_threshold:
            log_decision(
                "visual_circuit_breaker",
                "batch_health_ok",
                f"Batch failure rate {failure_rate:.1%} below threshold "
                f"{self.batch_failure_threshold:.1%}, continuing",
                {
                    "failure_rate": failure_rate,
                    "ads_with_no_candidates": self._ads_with_no_candidates,
                    "total_ads_attempted": self._total_ads_attempted,
                    "threshold": self.batch_failure_threshold,
                },
            )
            return "continue"

        # Failure rate at or above threshold — halt
        log_circuit_breaker_event(
            tier="3",
            failure_rate=failure_rate,
            ads_affected=self._ads_with_no_candidates,
            total_ads=self._total_ads_attempted,
            action="halt_image_gen",
        )
        return "halt"

    def reset_batch(self) -> None:
        """Clear all state between batches."""
        self._variant_attempts = {}
        self._ad_candidates = {}
        self._total_ads_attempted = 0
        self._ads_with_no_candidates = 0

        log_decision(
            "visual_circuit_breaker",
            "batch_reset",
            "Circuit breaker state cleared for new batch",
            {},
        )
