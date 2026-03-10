"""Tests for VisualCircuitBreaker — three-tier image gen failure handling.

All tests mock log_decision to avoid DB dependency.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.iterate.visual_healing import VisualCircuitBreaker


@pytest.fixture(autouse=True)
def _mock_log_decision():
    """Prevent all log_decision calls from hitting the database."""
    with patch("src.iterate.visual_healing.log_decision"):
        with patch("src.decisions.logger.log_decision") as mock_ld:
            mock_ld.return_value = "mock-decision-id"
            yield mock_ld


# -------------------------------------------------------------------
# Tier 1: per-variant retry capping
# -------------------------------------------------------------------


class TestTier1VariantRetry:
    def test_pass_returns_accept(self):
        cb = VisualCircuitBreaker()
        result = cb.record_variant_attempt("ad-1", "hero", passed=True)
        assert result == "accept"

    def test_first_failure_returns_retry_pro(self):
        cb = VisualCircuitBreaker()
        result = cb.record_variant_attempt("ad-1", "hero", passed=False)
        assert result == "retry_pro"

    def test_second_failure_returns_discard(self):
        cb = VisualCircuitBreaker()
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        result = cb.record_variant_attempt("ad-1", "hero", passed=False)
        assert result == "discard"

    def test_third_failure_still_discard(self):
        cb = VisualCircuitBreaker()
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        result = cb.record_variant_attempt("ad-1", "hero", passed=False)
        assert result == "discard"

    def test_accept_adds_to_candidates(self):
        cb = VisualCircuitBreaker()
        cb.record_variant_attempt("ad-1", "hero", passed=True)
        assert "hero" in cb._ad_candidates["ad-1"]


# -------------------------------------------------------------------
# Tier 2: per-ad text-only fallback
# -------------------------------------------------------------------


class TestTier2AdStatus:
    def test_has_candidates_when_variant_passes(self):
        cb = VisualCircuitBreaker()
        cb.record_variant_attempt("ad-1", "hero", passed=True)
        cb.record_variant_attempt("ad-1", "lifestyle", passed=False)
        result = cb.check_ad_status("ad-1")
        assert result == "has_candidates"

    def test_text_only_when_all_variants_fail(self):
        cb = VisualCircuitBreaker()
        # Two variants, both fail twice (flash + pro)
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.record_variant_attempt("ad-1", "lifestyle", passed=False)
        cb.record_variant_attempt("ad-1", "lifestyle", passed=False)
        result = cb.check_ad_status("ad-1")
        assert result == "text_only"

    def test_text_only_increments_failure_counter(self):
        cb = VisualCircuitBreaker()
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.check_ad_status("ad-1")
        assert cb._ads_with_no_candidates == 1
        assert cb._total_ads_attempted == 1

    def test_has_candidates_increments_total_only(self):
        cb = VisualCircuitBreaker()
        cb.record_variant_attempt("ad-1", "hero", passed=True)
        cb.check_ad_status("ad-1")
        assert cb._ads_with_no_candidates == 0
        assert cb._total_ads_attempted == 1


# -------------------------------------------------------------------
# Tier 3: per-batch health check
# -------------------------------------------------------------------


class TestTier3BatchHealth:
    def test_halt_when_above_threshold(self):
        cb = VisualCircuitBreaker(batch_failure_threshold=0.5)

        # 3 ads: 2 fail, 1 passes => 66.7% failure rate
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.check_ad_status("ad-1")

        cb.record_variant_attempt("ad-2", "hero", passed=False)
        cb.record_variant_attempt("ad-2", "hero", passed=False)
        cb.check_ad_status("ad-2")

        cb.record_variant_attempt("ad-3", "hero", passed=True)
        cb.check_ad_status("ad-3")

        result = cb.check_batch_health()
        assert result == "halt"

    def test_continue_when_below_threshold(self):
        cb = VisualCircuitBreaker(batch_failure_threshold=0.5)

        # 3 ads: 1 fails, 2 pass => 33.3% failure rate
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.check_ad_status("ad-1")

        cb.record_variant_attempt("ad-2", "hero", passed=True)
        cb.check_ad_status("ad-2")

        cb.record_variant_attempt("ad-3", "hero", passed=True)
        cb.check_ad_status("ad-3")

        result = cb.check_batch_health()
        assert result == "continue"

    def test_continue_when_zero_ads(self):
        cb = VisualCircuitBreaker()
        result = cb.check_batch_health()
        assert result == "continue"

    def test_halt_at_exact_threshold(self):
        cb = VisualCircuitBreaker(batch_failure_threshold=0.5)

        # 2 ads: 1 fails, 1 passes => exactly 50%
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.check_ad_status("ad-1")

        cb.record_variant_attempt("ad-2", "hero", passed=True)
        cb.check_ad_status("ad-2")

        result = cb.check_batch_health()
        assert result == "halt"


# -------------------------------------------------------------------
# Batch reset
# -------------------------------------------------------------------


class TestBatchReset:
    def test_reset_clears_all_state(self):
        cb = VisualCircuitBreaker()

        # Populate state
        cb.record_variant_attempt("ad-1", "hero", passed=True)
        cb.check_ad_status("ad-1")
        cb.record_variant_attempt("ad-2", "hero", passed=False)
        cb.record_variant_attempt("ad-2", "hero", passed=False)
        cb.check_ad_status("ad-2")

        cb.reset_batch()

        assert cb._variant_attempts == {}
        assert cb._ad_candidates == {}
        assert cb._total_ads_attempted == 0
        assert cb._ads_with_no_candidates == 0

    def test_reset_allows_fresh_batch(self):
        cb = VisualCircuitBreaker()

        # First batch: all fail
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.record_variant_attempt("ad-1", "hero", passed=False)
        cb.check_ad_status("ad-1")
        assert cb.check_batch_health() == "halt"

        # Reset and start fresh
        cb.reset_batch()
        cb.record_variant_attempt("ad-2", "hero", passed=True)
        cb.check_ad_status("ad-2")
        assert cb.check_batch_health() == "continue"


# -------------------------------------------------------------------
# Decision logging integration
# -------------------------------------------------------------------


class TestDecisionLogging:
    def test_halt_calls_log_circuit_breaker_event(self):
        with patch("src.iterate.visual_healing.log_circuit_breaker_event") as mock_cb_event:
            with patch("src.iterate.visual_healing.log_decision"):
                cb = VisualCircuitBreaker(batch_failure_threshold=0.5)

                cb.record_variant_attempt("ad-1", "hero", passed=False)
                cb.record_variant_attempt("ad-1", "hero", passed=False)
                cb.check_ad_status("ad-1")

                cb.check_batch_health()

                mock_cb_event.assert_called_once_with(
                    tier="3",
                    failure_rate=1.0,
                    ads_affected=1,
                    total_ads=1,
                    action="halt_image_gen",
                )

    def test_continue_does_not_call_circuit_breaker_event(self):
        with patch("src.iterate.visual_healing.log_circuit_breaker_event") as mock_cb_event:
            with patch("src.iterate.visual_healing.log_decision"):
                cb = VisualCircuitBreaker(batch_failure_threshold=0.5)

                cb.record_variant_attempt("ad-1", "hero", passed=True)
                cb.check_ad_status("ad-1")

                cb.check_batch_health()

                mock_cb_event.assert_not_called()
