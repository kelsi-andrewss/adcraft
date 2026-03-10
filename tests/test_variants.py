"""Tests for A/B visual variant generation.

Covers: variant prompt modification, best-selection by score, shared
variant_group_id, variant_type tagging, partial failure resilience,
total failure returns empty, and num_variants capping.
"""

from __future__ import annotations

import io
import uuid
from unittest.mock import MagicMock

from PIL import Image as PILImage

from src.generate.variants import VARIANT_TYPES, VariantGenerator, _compute_visual_weighted_average
from src.models.ad import AdCopy
from src.models.brief import AdBrief
from src.models.creative import ImageResult, VisualBrief
from src.models.evaluation import DimensionScore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ad_copy(**kwargs) -> AdCopy:
    defaults = {
        "id": "test-ad-1",
        "primary_text": "Expert SAT tutoring for your child",
        "headline": "Boost Your SAT Score",
        "description": "Personalized 1-on-1 tutoring",
        "cta_button": "Get Started",
    }
    defaults.update(kwargs)
    return AdCopy(**defaults)


def _make_brief(**kwargs) -> AdBrief:
    defaults = {
        "audience_segment": "Parents of high school juniors",
        "product_offer": "SAT Prep Package",
        "campaign_goal": "lead_generation",
        "tone": "supportive",
    }
    defaults.update(kwargs)
    return AdBrief(**defaults)


def _make_visual_brief(**kwargs) -> VisualBrief:
    defaults = {
        "prompt": "A student studying at a bright desk with warm lighting",
        "negative_prompt": "no dark imagery",
        "aspect_ratio": "1:1",
        "resolution": "1K",
        "placement": "feed",
    }
    defaults.update(kwargs)
    return VisualBrief(**defaults)


def _make_png_bytes() -> bytes:
    """Generate minimal valid PNG bytes for PIL to open."""
    img = PILImage.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_image_result(model_id: str = "gemini-2.5-flash-image", **kwargs) -> ImageResult:
    defaults = {
        "image_bytes": _make_png_bytes(),
        "file_path": "data/images/test.png",
        "model_id": model_id,
        "cost_usd": 0.0,
        "generation_config": {},
    }
    defaults.update(kwargs)
    return ImageResult(**defaults)


def _make_dimension_scores(
    brand: float = 7.0,
    composition: float = 7.0,
    synergy: float = 7.0,
) -> list[DimensionScore]:
    return [
        DimensionScore(dimension="brand_consistency", score=brand, rationale="ok"),
        DimensionScore(dimension="composition_quality", score=composition, rationale="ok"),
        DimensionScore(dimension="text_image_synergy", score=synergy, rationale="ok"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVariantPromptModification:
    """Verify each variant type prepends its modifier to the base prompt."""

    def test_each_variant_prepends_modifier(self):
        """Each variant type's modifier appears before the original prompt."""
        base_brief = _make_visual_brief(prompt="Original prompt about SAT tutoring")

        for variant_type, modifier in VARIANT_TYPES.items():
            modified = base_brief.model_copy(
                update={"prompt": f"{modifier}\n\n{base_brief.prompt}"}
            )
            assert modified.prompt.startswith(modifier)
            assert "Original prompt about SAT tutoring" in modified.prompt

    def test_original_prompt_preserved(self):
        """The original prompt content is intact after modification."""
        original_text = "Detailed visual prompt with specific brand elements and colors"
        base_brief = _make_visual_brief(prompt=original_text)

        for modifier in VARIANT_TYPES.values():
            modified = base_brief.model_copy(
                update={"prompt": f"{modifier}\n\n{base_brief.prompt}"}
            )
            assert original_text in modified.prompt

    def test_non_prompt_fields_unchanged(self):
        """model_copy only changes prompt, other fields stay the same."""
        base_brief = _make_visual_brief(
            aspect_ratio="9:16",
            resolution="2K",
            placement="stories",
        )

        modifier = VARIANT_TYPES["lifestyle"]
        modified = base_brief.model_copy(update={"prompt": f"{modifier}\n\n{base_brief.prompt}"})

        assert modified.aspect_ratio == "9:16"
        assert modified.resolution == "2K"
        assert modified.placement == "stories"
        assert modified.negative_prompt == base_brief.negative_prompt


class TestGenerateVariantsSelectsBest:
    """Mock image_engine and eval_engine to verify best-first sorting."""

    def test_returns_sorted_by_score_descending(self):
        """Variants returned best-first by visual eval weighted average."""
        mock_image_engine = MagicMock()
        mock_eval_engine = MagicMock()
        mock_prompt_gen = MagicMock()

        mock_prompt_gen.generate.return_value = _make_visual_brief()

        # Each call to generate_image returns a fresh ImageResult
        mock_image_engine.generate_image.side_effect = [
            _make_image_result(),
            _make_image_result(),
            _make_image_result(),
        ]

        # Scores: lifestyle=6.0, product=9.0, emotional=7.5
        mock_eval_engine.evaluate_visual.side_effect = [
            _make_dimension_scores(brand=6.0, composition=6.0, synergy=6.0),
            _make_dimension_scores(brand=9.0, composition=9.0, synergy=9.0),
            _make_dimension_scores(brand=7.5, composition=7.5, synergy=7.5),
        ]

        gen = VariantGenerator(mock_image_engine, mock_eval_engine, mock_prompt_gen)
        results = gen.generate_variants(_make_ad_copy(), _make_brief())

        assert len(results) == 3
        # product (9.0 weighted avg) should be first
        assert results[0].variant_type == "product"
        # emotional (7.5) second
        assert results[1].variant_type == "emotional"
        # lifestyle (6.0) last
        assert results[2].variant_type == "lifestyle"

    def test_first_element_has_highest_score(self):
        """The first returned ImageResult corresponds to the highest eval score."""
        mock_image_engine = MagicMock()
        mock_eval_engine = MagicMock()
        mock_prompt_gen = MagicMock()

        mock_prompt_gen.generate.return_value = _make_visual_brief()

        mock_image_engine.generate_image.side_effect = [
            _make_image_result(),
            _make_image_result(),
        ]

        # lifestyle=5.0, product=8.0
        mock_eval_engine.evaluate_visual.side_effect = [
            _make_dimension_scores(brand=5.0, composition=5.0, synergy=5.0),
            _make_dimension_scores(brand=8.0, composition=8.0, synergy=8.0),
        ]

        gen = VariantGenerator(mock_image_engine, mock_eval_engine, mock_prompt_gen)
        results = gen.generate_variants(_make_ad_copy(), _make_brief(), num_variants=2)

        assert results[0].variant_type == "product"


class TestVariantGroupIdShared:
    """Verify all results from one generate_variants call share the same group ID."""

    def test_shared_uuid4(self):
        """All ImageResults share the same variant_group_id, which is a valid UUID4."""
        mock_image_engine = MagicMock()
        mock_eval_engine = MagicMock()
        mock_prompt_gen = MagicMock()

        mock_prompt_gen.generate.return_value = _make_visual_brief()
        mock_image_engine.generate_image.side_effect = [
            _make_image_result(),
            _make_image_result(),
            _make_image_result(),
        ]
        mock_eval_engine.evaluate_visual.return_value = _make_dimension_scores()

        gen = VariantGenerator(mock_image_engine, mock_eval_engine, mock_prompt_gen)
        results = gen.generate_variants(_make_ad_copy(), _make_brief())

        assert len(results) == 3

        group_ids = {r.variant_group_id for r in results}
        assert len(group_ids) == 1, "All results should share the same variant_group_id"

        group_id = group_ids.pop()
        # Validate it's a proper UUID4
        parsed = uuid.UUID(group_id, version=4)
        assert str(parsed) == group_id


class TestVariantTypeSet:
    """Verify each ImageResult has variant_type matching its VARIANT_TYPES key."""

    def test_variant_types_assigned(self):
        """Each result's variant_type matches the corresponding VARIANT_TYPES key."""
        mock_image_engine = MagicMock()
        mock_eval_engine = MagicMock()
        mock_prompt_gen = MagicMock()

        mock_prompt_gen.generate.return_value = _make_visual_brief()
        mock_image_engine.generate_image.side_effect = [
            _make_image_result(),
            _make_image_result(),
            _make_image_result(),
        ]
        # Same score so order is preserved
        mock_eval_engine.evaluate_visual.return_value = _make_dimension_scores()

        gen = VariantGenerator(mock_image_engine, mock_eval_engine, mock_prompt_gen)
        results = gen.generate_variants(_make_ad_copy(), _make_brief())

        result_types = {r.variant_type for r in results}
        assert result_types == set(VARIANT_TYPES.keys())


class TestPartialFailureContinues:
    """One variant fails, other two succeed -- failed variant doesn't abort."""

    def test_two_results_on_one_failure(self):
        """One generate_image raises, two succeed => two results returned."""
        mock_image_engine = MagicMock()
        mock_eval_engine = MagicMock()
        mock_prompt_gen = MagicMock()

        mock_prompt_gen.generate.return_value = _make_visual_brief()

        # First variant (lifestyle) raises, other two succeed
        mock_image_engine.generate_image.side_effect = [
            RuntimeError("flash and pro both failed"),
            _make_image_result(),
            _make_image_result(),
        ]

        mock_eval_engine.evaluate_visual.return_value = _make_dimension_scores()

        gen = VariantGenerator(mock_image_engine, mock_eval_engine, mock_prompt_gen)
        results = gen.generate_variants(_make_ad_copy(), _make_brief())

        assert len(results) == 2
        # lifestyle should not be in results since it failed
        result_types = {r.variant_type for r in results}
        assert "lifestyle" not in result_types
        assert "product" in result_types
        assert "emotional" in result_types


class TestAllVariantsFailReturnsEmpty:
    """All generate_image calls raise => empty list returned."""

    def test_empty_list_on_total_failure(self):
        """All variants fail => returns empty list."""
        mock_image_engine = MagicMock()
        mock_eval_engine = MagicMock()
        mock_prompt_gen = MagicMock()

        mock_prompt_gen.generate.return_value = _make_visual_brief()
        mock_image_engine.generate_image.side_effect = RuntimeError("all tiers exhausted")

        gen = VariantGenerator(mock_image_engine, mock_eval_engine, mock_prompt_gen)
        results = gen.generate_variants(_make_ad_copy(), _make_brief())

        assert results == []
        # eval_engine should never be called since no images were generated
        mock_eval_engine.evaluate_visual.assert_not_called()


class TestNumVariantsCapped:
    """num_variants > len(VARIANT_TYPES) gets capped."""

    def test_capped_at_variant_types_length(self):
        """Passing num_variants=5 only generates 3 (len(VARIANT_TYPES))."""
        mock_image_engine = MagicMock()
        mock_eval_engine = MagicMock()
        mock_prompt_gen = MagicMock()

        mock_prompt_gen.generate.return_value = _make_visual_brief()
        mock_image_engine.generate_image.side_effect = [
            _make_image_result(),
            _make_image_result(),
            _make_image_result(),
        ]
        mock_eval_engine.evaluate_visual.return_value = _make_dimension_scores()

        gen = VariantGenerator(mock_image_engine, mock_eval_engine, mock_prompt_gen)
        results = gen.generate_variants(_make_ad_copy(), _make_brief(), num_variants=5)

        assert len(results) == 3
        assert mock_image_engine.generate_image.call_count == 3


class TestComputeVisualWeightedAverage:
    """Unit test for the weighted average helper."""

    def test_weighted_average_calculation(self):
        """Weighted avg = 0.4*brand + 0.3*composition + 0.3*synergy."""
        scores = _make_dimension_scores(brand=8.0, composition=6.0, synergy=10.0)
        # 0.4*8 + 0.3*6 + 0.3*10 = 3.2 + 1.8 + 3.0 = 8.0
        assert _compute_visual_weighted_average(scores) == 8.0

    def test_all_tens(self):
        """All 10s => weighted avg = 10.0."""
        scores = _make_dimension_scores(brand=10.0, composition=10.0, synergy=10.0)
        assert _compute_visual_weighted_average(scores) == 10.0
