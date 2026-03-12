"""Tests for calibration metrics and overlap guard.

Pure math tests -- no API mocking, no LLM calls. Validates Krippendorff's Alpha,
Spearman correlation, MAE computation, and the overlap guard that prevents
circular validation between gold set and few-shot examples.
"""

from __future__ import annotations

import json
from pathlib import Path

import krippendorff
import numpy as np
import pytest
from scipy.stats import spearmanr

from src.evaluate.calibrate import calculate_metrics, check_overlap
from src.evaluate.rubrics import DIMENSIONS

# ---------------------------------------------------------------------------
# Fixtures: synthetic score data
# ---------------------------------------------------------------------------

GOLD_ADS_PERFECT_AGREEMENT = [
    {
        "id": "synth-001",
        "human_scores": {d: 8.0 for d in DIMENSIONS},
    },
    {
        "id": "synth-002",
        "human_scores": {d: 4.0 for d in DIMENSIONS},
    },
]

EVAL_RESULTS_PERFECT_AGREEMENT = [
    {"llm_scores": {d: 8.0 for d in DIMENSIONS}},
    {"llm_scores": {d: 4.0 for d in DIMENSIONS}},
]


# ---------------------------------------------------------------------------
# Test: Krippendorff's Alpha
# ---------------------------------------------------------------------------


class TestKrippendorffAlpha:
    """Verify inter-rater reliability computation using known inputs."""

    def test_perfect_agreement_yields_alpha_1(self):
        """Two raters with identical scores produce alpha = 1.0."""
        human = [8.0, 4.0, 8.0, 4.0]
        llm = [8.0, 4.0, 8.0, 4.0]

        reliability_data = np.array([human, llm])
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement="ordinal",
        )

        assert alpha == pytest.approx(1.0, abs=0.001)

    def test_known_disagreement_yields_expected_alpha(self):
        """A hand-computed case with moderate agreement."""
        # Two raters, 6 items. Known alpha ~0.691 for ordinal.
        human = [9.0, 7.0, 5.0, 3.0, 8.0, 6.0]
        llm = [8.0, 7.0, 6.0, 4.0, 7.0, 5.0]

        reliability_data = np.array([human, llm])
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement="ordinal",
        )

        # Alpha should be positive (substantial agreement) but below 1.0
        assert 0.4 < alpha < 1.0

    def test_calculate_metrics_returns_alpha_for_perfect_agreement(self):
        """calculate_metrics returns alpha=1.0 when human and LLM scores match exactly."""
        alpha, _rho, _mae = calculate_metrics(
            GOLD_ADS_PERFECT_AGREEMENT,
            EVAL_RESULTS_PERFECT_AGREEMENT,
        )
        assert alpha == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Test: Spearman rank correlation
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    """Verify rank correlation computation."""

    def test_perfect_positive_correlation(self):
        """Monotonically increasing scores yield rho = 1.0."""
        human = [1.0, 2.0, 3.0, 4.0, 5.0]
        llm = [1.0, 2.0, 3.0, 4.0, 5.0]

        rho, _p = spearmanr(human, llm)
        assert rho == pytest.approx(1.0, abs=0.001)

    def test_perfect_negative_correlation(self):
        """Reversed rankings yield rho = -1.0."""
        human = [1.0, 2.0, 3.0, 4.0, 5.0]
        llm = [5.0, 4.0, 3.0, 2.0, 1.0]

        rho, _p = spearmanr(human, llm)
        assert rho == pytest.approx(-1.0, abs=0.001)

    def test_known_partial_correlation(self):
        """A case with known moderate positive correlation."""
        human = [9.0, 7.0, 5.0, 3.0, 8.0, 6.0]
        llm = [8.0, 7.0, 6.0, 4.0, 7.0, 5.0]

        rho, _p = spearmanr(human, llm)
        # Ranks are similar but not identical -- expect strong positive rho
        assert 0.7 < rho < 1.0

    def test_calculate_metrics_returns_rho_for_perfect_agreement(self):
        """calculate_metrics returns rho=1.0 when human and LLM scores match exactly."""
        _alpha, rho, _mae = calculate_metrics(
            GOLD_ADS_PERFECT_AGREEMENT,
            EVAL_RESULTS_PERFECT_AGREEMENT,
        )
        assert rho == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Test: Mean Absolute Error
# ---------------------------------------------------------------------------


class TestMeanAbsoluteError:
    """Verify per-dimension MAE computation."""

    def test_zero_error_for_identical_scores(self):
        """When human and LLM agree exactly, MAE is 0.0 for every dimension."""
        _alpha, _rho, mae = calculate_metrics(
            GOLD_ADS_PERFECT_AGREEMENT,
            EVAL_RESULTS_PERFECT_AGREEMENT,
        )
        for dim in DIMENSIONS:
            assert mae[dim] == pytest.approx(0.0, abs=0.001)

    def test_known_mae_values(self):
        """Hand-computed MAE for a known score pair."""
        gold_ads = [
            {
                "id": "mae-001",
                "human_scores": {
                    "clarity": 8.0,
                    "learner_benefit": 6.0,
                    "cta_effectiveness": 7.0,
                    "brand_voice": 9.0,
                    "student_empathy": 5.0,
                    "pedagogical_integrity": 7.0,
                },
            },
            {
                "id": "mae-002",
                "human_scores": {
                    "clarity": 6.0,
                    "learner_benefit": 4.0,
                    "cta_effectiveness": 5.0,
                    "brand_voice": 7.0,
                    "student_empathy": 3.0,
                    "pedagogical_integrity": 5.0,
                },
            },
        ]
        eval_results = [
            {
                "llm_scores": {
                    "clarity": 7.0,
                    "learner_benefit": 7.0,
                    "cta_effectiveness": 6.0,
                    "brand_voice": 8.0,
                    "student_empathy": 6.0,
                    "pedagogical_integrity": 6.0,
                },
            },
            {
                "llm_scores": {
                    "clarity": 5.0,
                    "learner_benefit": 5.0,
                    "cta_effectiveness": 6.0,
                    "brand_voice": 6.0,
                    "student_empathy": 4.0,
                    "pedagogical_integrity": 4.0,
                },
            },
        ]

        _alpha, _rho, mae = calculate_metrics(gold_ads, eval_results)

        # clarity: (|8-7| + |6-5|) / 2 = 1.0
        assert mae["clarity"] == pytest.approx(1.0, abs=0.001)
        # learner_benefit: (|6-7| + |4-5|) / 2 = 1.0
        assert mae["learner_benefit"] == pytest.approx(1.0, abs=0.001)
        # cta_effectiveness: (|7-6| + |5-6|) / 2 = 1.0
        assert mae["cta_effectiveness"] == pytest.approx(1.0, abs=0.001)
        # brand_voice: (|9-8| + |7-6|) / 2 = 1.0
        assert mae["brand_voice"] == pytest.approx(1.0, abs=0.001)
        # student_empathy: (|5-6| + |3-4|) / 2 = 1.0
        assert mae["student_empathy"] == pytest.approx(1.0, abs=0.001)
        # pedagogical_integrity: (|7-6| + |5-4|) / 2 = 1.0
        assert mae["pedagogical_integrity"] == pytest.approx(1.0, abs=0.001)

    def test_mae_with_asymmetric_errors(self):
        """MAE correctly handles different error magnitudes per dimension."""
        gold_ads = [
            {
                "id": "asym-001",
                "human_scores": {
                    "clarity": 9.0,
                    "learner_benefit": 3.0,
                    "cta_effectiveness": 7.0,
                    "brand_voice": 5.0,
                    "student_empathy": 8.0,
                    "pedagogical_integrity": 6.0,
                },
            },
        ]
        eval_results = [
            {
                "llm_scores": {
                    "clarity": 7.0,  # error 2
                    "learner_benefit": 5.0,  # error 2
                    "cta_effectiveness": 7.0,  # error 0
                    "brand_voice": 8.0,  # error 3
                    "student_empathy": 6.0,  # error 2
                    "pedagogical_integrity": 6.0,  # error 0
                },
            },
        ]

        _alpha, _rho, mae = calculate_metrics(gold_ads, eval_results)

        assert mae["clarity"] == pytest.approx(2.0, abs=0.001)
        assert mae["learner_benefit"] == pytest.approx(2.0, abs=0.001)
        assert mae["cta_effectiveness"] == pytest.approx(0.0, abs=0.001)
        assert mae["brand_voice"] == pytest.approx(3.0, abs=0.001)
        assert mae["student_empathy"] == pytest.approx(2.0, abs=0.001)
        assert mae["pedagogical_integrity"] == pytest.approx(0.0, abs=0.001)

    def test_mae_values_are_rounded_to_3_decimals(self):
        """calculate_metrics rounds MAE to 3 decimal places."""
        gold_ads = [
            {
                "id": "round-001",
                "human_scores": {d: 7.0 for d in DIMENSIONS},
            },
            {
                "id": "round-002",
                "human_scores": {d: 5.0 for d in DIMENSIONS},
            },
            {
                "id": "round-003",
                "human_scores": {d: 8.0 for d in DIMENSIONS},
            },
        ]
        eval_results = [
            {"llm_scores": {d: 6.0 for d in DIMENSIONS}},  # error 1
            {"llm_scores": {d: 6.0 for d in DIMENSIONS}},  # error 1
            {"llm_scores": {d: 6.0 for d in DIMENSIONS}},  # error 2
        ]

        _alpha, _rho, mae = calculate_metrics(gold_ads, eval_results)

        # MAE = (1 + 1 + 2) / 3 = 1.333...
        for dim in DIMENSIONS:
            assert mae[dim] == pytest.approx(1.333, abs=0.001)
            # Verify exact rounding: str representation should have at most 3 decimals
            decimal_part = str(mae[dim]).split(".")[-1]
            assert len(decimal_part) <= 3


# ---------------------------------------------------------------------------
# Test: Overlap guard
# ---------------------------------------------------------------------------


class TestOverlapGuard:
    """Verify that the overlap guard prevents circular validation."""

    def test_no_overlap_between_json_files(self):
        """Gold set and few-shot example JSON files share no ad IDs."""
        gold_path = Path("data/reference_ads/calibration_gold_set.json")
        few_shot_path = Path("data/reference_ads/few_shot_examples.json")

        with open(gold_path) as f:
            gold_data = json.load(f)
        with open(few_shot_path) as f:
            few_shot_data = json.load(f)

        gold_ids = {ad["id"] for ad in gold_data["gold_ads"]}
        few_shot_ids = {ad["id"] for ad in few_shot_data["few_shot_ads"]}

        overlap = gold_ids & few_shot_ids
        assert overlap == set(), (
            f"Overlap detected between gold set and few-shot examples: {overlap}"
        )

    def test_no_overlap_between_gold_texts_and_few_shot_examples(self):
        """Gold ad primary_text must not appear in FEW_SHOT_EXAMPLES strings."""
        gold_path = Path("data/reference_ads/calibration_gold_set.json")
        with open(gold_path) as f:
            gold_data = json.load(f)

        # This should not raise -- the real data is clean
        check_overlap(gold_data["gold_ads"])

    def test_check_overlap_raises_on_contaminated_data(self):
        """check_overlap raises ValueError when a gold ad's text appears in few-shot examples."""
        from src.evaluate.rubrics import FEW_SHOT_EXAMPLES

        # Pick the first few-shot example's text and embed it in a fake gold ad
        first_dim = DIMENSIONS[0]
        # Extract a substring that actually appears in the few-shot example string
        example_str = FEW_SHOT_EXAMPLES[first_dim]
        # The clarity example contains this ref-great-001 text
        contaminated_text = (
            "Your child's SAT score shouldn't be limited by access to great teaching."
        )

        # Verify our contaminated text actually appears in the few-shot examples
        assert contaminated_text in example_str, (
            "Test setup error: contaminated text not found in FEW_SHOT_EXAMPLES"
        )

        fake_gold_ads = [
            {
                "id": "contaminated-001",
                "primary_text": contaminated_text,
            },
        ]

        with pytest.raises(ValueError, match="Overlap detected"):
            check_overlap(fake_gold_ads)

    def test_check_overlap_passes_for_clean_data(self):
        """check_overlap does not raise for ads with unique text."""
        clean_gold_ads = [
            {
                "id": "clean-001",
                "primary_text": "This text does not appear in any few-shot example.",
            },
            {
                "id": "clean-002",
                "primary_text": "Another completely unique ad text for testing purposes.",
            },
        ]

        # Should not raise
        check_overlap(clean_gold_ads)
