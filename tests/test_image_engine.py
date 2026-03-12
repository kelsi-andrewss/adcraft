"""Tests for the image generation engine.

Covers: dynamic threshold calculation, escalation logic, image save,
and query helpers. All Gemini API calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.db.init_db import init_db
from src.db.queries import (
    get_ads_with_images,
    get_image_gen_threshold,
    insert_ad,
    insert_quality_snapshot,
    update_ad_image,
)
from src.generate.image_engine import (
    FLASH_IMAGE_MODEL,
    IMAGE_COST_USD,
    IMAGES_DIR,
    PRO_IMAGE_MODEL,
    ImageGenerationEngine,
    ImageGenerationError,
)
from src.models.creative import ImageResult, VisualBrief

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_visual_brief(**kwargs) -> VisualBrief:
    defaults = {
        "prompt": "A vibrant SAT tutoring ad with a student celebrating",
        "negative_prompt": "no dark imagery, no fear-based visuals",
        "aspect_ratio": "1:1",
        "resolution": "1K",
        "placement": "feed",
    }
    defaults.update(kwargs)
    return VisualBrief(**defaults)


def _make_mock_response(image_bytes: bytes = b"fake-png-data") -> MagicMock:
    """Build a mock Gemini response with inline image data."""
    inline_data = MagicMock()
    inline_data.data = image_bytes

    image_part = MagicMock()
    image_part.inline_data = inline_data

    content = MagicMock()
    content.parts = [image_part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_empty_response() -> MagicMock:
    """Build a mock Gemini response with no image data."""
    text_part = MagicMock()
    text_part.inline_data = None

    content = MagicMock()
    content.parts = [text_part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# Dynamic threshold (get_image_gen_threshold)
# ---------------------------------------------------------------------------


class TestDynamicThreshold:
    """Tests for get_image_gen_threshold query helper."""

    def test_no_snapshots_returns_default(self, db_conn):
        """No quality snapshots => falls back to 7.0."""
        assert get_image_gen_threshold(db_conn) == 7.0

    def test_high_avg_returns_avg_minus_half(self, db_conn):
        """Snapshot with avg 8.5 => max(7.0, 8.0) = 8.0."""
        insert_quality_snapshot(db_conn, cycle_number=1, avg_weighted_score=8.5)
        assert get_image_gen_threshold(db_conn) == 8.0

    def test_low_avg_clamps_to_floor(self, db_conn):
        """Snapshot with avg 7.2 => max(7.0, 6.7) = 7.0."""
        insert_quality_snapshot(db_conn, cycle_number=1, avg_weighted_score=7.2)
        assert get_image_gen_threshold(db_conn) == 7.0

    def test_boundary_avg_returns_floor(self, db_conn):
        """Snapshot with avg 7.5 => max(7.0, 7.0) = 7.0."""
        insert_quality_snapshot(db_conn, cycle_number=1, avg_weighted_score=7.5)
        assert get_image_gen_threshold(db_conn) == 7.0

    def test_uses_most_recent_snapshot(self, db_conn):
        """Multiple snapshots: uses the one with highest cycle_number."""
        insert_quality_snapshot(db_conn, cycle_number=1, avg_weighted_score=6.0)
        insert_quality_snapshot(db_conn, cycle_number=2, avg_weighted_score=9.0)
        # Most recent (cycle 2): max(7.0, 9.0 - 0.5) = 8.5
        assert get_image_gen_threshold(db_conn) == 8.5

    def test_null_avg_returns_default(self, db_conn):
        """Snapshot with NULL avg_weighted_score => falls back to 7.0."""
        insert_quality_snapshot(db_conn, cycle_number=1, avg_weighted_score=None)
        assert get_image_gen_threshold(db_conn) == 7.0


# ---------------------------------------------------------------------------
# generate_image (escalation)
# ---------------------------------------------------------------------------


class TestGenerateImage:
    """Tests for the escalation ladder."""

    def test_flash_success(self, tmp_path):
        """Flash succeeds on first try => returns flash result."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_mock_response(b"flash-image-data")

        with patch.object(IMAGES_DIR.__class__, "mkdir"):
            engine = ImageGenerationEngine(client=mock_client)
            engine._save_image = lambda data, aid, var: str(tmp_path / f"{aid}_{var}.png")
            result = engine.generate_image(_make_visual_brief(), ad_id="test-ad-1", variant="v1")

        assert isinstance(result, ImageResult)
        assert result.model_id == FLASH_IMAGE_MODEL
        assert result.cost_usd == IMAGE_COST_USD[FLASH_IMAGE_MODEL]
        # Flash called once
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == FLASH_IMAGE_MODEL

    def test_flash_fails_escalates_to_pro(self, tmp_path):
        """Flash fails => escalates to pro => returns pro result."""
        mock_client = MagicMock()

        flash_response = RuntimeError("flash model unavailable")
        pro_response = _make_mock_response(b"pro-image-data")

        mock_client.models.generate_content.side_effect = [
            flash_response,
            pro_response,
        ]

        with patch.object(IMAGES_DIR.__class__, "mkdir"):
            engine = ImageGenerationEngine(client=mock_client)
            # Disable tenacity retry for test speed
            engine._generate_with_model = engine._generate_with_model.__wrapped__.__get__(engine)
            engine._save_image = lambda data, aid, var: str(tmp_path / f"{aid}_{var}.png")
            result = engine.generate_image(_make_visual_brief(), ad_id="test-ad-2", variant="v1")

        assert result.model_id == PRO_IMAGE_MODEL
        assert result.cost_usd == IMAGE_COST_USD[PRO_IMAGE_MODEL]
        assert mock_client.models.generate_content.call_count == 2

    def test_both_tiers_fail_raises(self, tmp_path):
        """Both flash and pro fail => raises ImageGenerationError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("all down")

        with patch.object(IMAGES_DIR.__class__, "mkdir"):
            engine = ImageGenerationEngine(client=mock_client)
            engine._generate_with_model = engine._generate_with_model.__wrapped__.__get__(engine)

            with pytest.raises(ImageGenerationError, match="failed for ad=test-ad-3"):
                engine.generate_image(_make_visual_brief(), ad_id="test-ad-3", variant="v1")

    def test_no_image_data_in_response_raises(self, tmp_path):
        """Response with no inline_data => raises ImageGenerationError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_empty_response()

        with patch.object(IMAGES_DIR.__class__, "mkdir"):
            engine = ImageGenerationEngine(client=mock_client)
            engine._generate_with_model = engine._generate_with_model.__wrapped__.__get__(engine)

            with pytest.raises(ImageGenerationError, match="No image data"):
                engine._generate_with_model(
                    FLASH_IMAGE_MODEL,
                    _make_visual_brief(),
                    "test-ad-4",
                    "v1",
                )


# ---------------------------------------------------------------------------
# _save_image
# ---------------------------------------------------------------------------


class TestSaveImage:
    """Tests for filesystem image saving."""

    def test_saves_to_correct_path(self, tmp_path):
        """Image bytes written to data/images/{ad_id}_{variant}.png."""
        mock_client = MagicMock()
        with patch.object(IMAGES_DIR.__class__, "mkdir"):
            engine = ImageGenerationEngine(client=mock_client)

        # Override IMAGES_DIR for isolation
        with patch("src.generate.image_engine.IMAGES_DIR", tmp_path):
            path = engine._save_image(b"test-image-bytes", "ad-123", "banner")

        assert path == str(tmp_path / "ad-123_banner.png")
        assert (tmp_path / "ad-123_banner.png").read_bytes() == b"test-image-bytes"

    def test_returns_relative_path_string(self, tmp_path):
        """Returned path is a string, not a Path object."""
        mock_client = MagicMock()
        with patch.object(IMAGES_DIR.__class__, "mkdir"):
            engine = ImageGenerationEngine(client=mock_client)

        with patch("src.generate.image_engine.IMAGES_DIR", tmp_path):
            path = engine._save_image(b"bytes", "ad-456", "default")

        assert isinstance(path, str)


# ---------------------------------------------------------------------------
# Query helpers: update_ad_image, get_ads_with_images
# ---------------------------------------------------------------------------


class TestImageQueryHelpers:
    """Tests for image-related query functions."""

    def test_update_ad_image(self, db_conn):
        """update_ad_image persists image metadata to the ads row."""
        ad_id = insert_ad(
            db_conn,
            primary_text="Test ad",
            headline="Test",
            description="Desc",
            cta_button="Learn More",
        )

        update_ad_image(
            db_conn,
            ad_id,
            image_path="data/images/test.png",
            visual_prompt="A bright tutoring scene",
            image_model=FLASH_IMAGE_MODEL,
            image_cost_usd=0.0,
            variant_group_id="group-1",
            variant_type="feed",
        )

        db_conn.row_factory = __import__("sqlite3").Row
        row = db_conn.execute("SELECT * FROM ads WHERE id = ?", (ad_id,)).fetchone()
        row_dict = dict(row)

        assert row_dict["image_path"] == "data/images/test.png"
        assert row_dict["visual_prompt"] == "A bright tutoring scene"
        assert row_dict["image_model"] == FLASH_IMAGE_MODEL
        assert row_dict["image_cost_usd"] == 0.0
        assert row_dict["variant_group_id"] == "group-1"
        assert row_dict["variant_type"] == "feed"

    def test_get_ads_with_images_returns_only_image_ads(self, db_conn):
        """get_ads_with_images returns only ads with image_path set."""
        ad_with = insert_ad(
            db_conn,
            primary_text="Has image",
            headline="Yes",
            description="Desc",
            cta_button="Learn More",
        )
        insert_ad(
            db_conn,
            primary_text="No image",
            headline="No",
            description="Desc",
            cta_button="Learn More",
        )

        update_ad_image(
            db_conn,
            ad_with,
            image_path="data/images/yes.png",
            visual_prompt="prompt",
            image_model=FLASH_IMAGE_MODEL,
            image_cost_usd=0.0,
        )

        results = get_ads_with_images(db_conn)
        assert len(results) == 1
        assert results[0]["id"] == ad_with

    def test_get_ads_with_images_respects_limit(self, db_conn):
        """Limit parameter caps the number of returned rows."""
        for i in range(5):
            ad_id = insert_ad(
                db_conn,
                primary_text=f"Ad {i}",
                headline=f"H{i}",
                description="Desc",
                cta_button="Learn More",
            )
            update_ad_image(
                db_conn,
                ad_id,
                image_path=f"data/images/{i}.png",
                visual_prompt="prompt",
                image_model=FLASH_IMAGE_MODEL,
                image_cost_usd=0.0,
            )

        results = get_ads_with_images(db_conn, limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Schema migration idempotency
# ---------------------------------------------------------------------------


class TestMigrationIdempotency:
    """Verify calling init_db twice doesn't error."""

    def test_double_init_no_error(self, tmp_path):
        """init_db() called twice on same DB file succeeds."""
        db_path = str(tmp_path / "test.db")
        conn1 = init_db(db_path)
        conn1.close()

        # Second call should be idempotent
        conn2 = init_db(db_path)
        conn2.row_factory = __import__("sqlite3").Row

        # Verify image columns exist
        rows = conn2.execute("PRAGMA table_info(ads)").fetchall()
        col_names = {row["name"] for row in rows}
        for expected in [
            "image_path",
            "visual_prompt",
            "image_model",
            "image_cost_usd",
            "variant_group_id",
            "variant_type",
        ]:
            assert expected in col_names, f"Missing column: {expected}"
        conn2.close()

    def test_in_memory_has_image_columns(self, db_conn):
        """In-memory DB from conftest has image columns after migration."""
        db_conn.row_factory = __import__("sqlite3").Row
        rows = db_conn.execute("PRAGMA table_info(ads)").fetchall()
        col_names = {row["name"] for row in rows}
        assert "image_path" in col_names
        assert "image_model" in col_names
