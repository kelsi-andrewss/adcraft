"""Tests for competitive intelligence: analyzer, pattern extraction, DB seeding."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.db.init_db import init_db
from src.intel.analyzer import (
    extract_patterns,
    load_curated_ads,
    seed_competitor_ads,
)


@pytest.fixture()
def db_conn():
    conn = init_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture()
def sample_ads() -> list[dict]:
    return [
        {
            "brand": "BrandA",
            "primary_text": "Ad text one",
            "headline": "Headline A",
            "cta_button": "Learn More",
            "hook_type": "social_proof",
            "emotional_angle": "achievement",
        },
        {
            "brand": "BrandA",
            "primary_text": "Ad text two",
            "headline": "Headline A2",
            "cta_button": "Sign Up",
            "hook_type": "social_proof",
            "emotional_angle": "parental_anxiety",
        },
        {
            "brand": "BrandB",
            "primary_text": "Ad text three",
            "headline": "Headline B",
            "cta_button": "Learn More",
            "hook_type": "urgency",
            "emotional_angle": "achievement",
        },
        {
            "brand": "BrandC",
            "primary_text": "Ad text four",
            "headline": "Headline C",
            "cta_button": "Get Offer",
            "hook_type": "question_hook",
            "emotional_angle": "affordability",
        },
    ]


class TestLoadCuratedAds:
    def test_load_valid_json(self, sample_ads: list[dict]) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_ads, f)
            f.flush()
            ads = load_curated_ads(f.name)

        assert len(ads) == 4
        assert ads[0]["brand"] == "BrandA"

    def test_load_missing_field_raises(self) -> None:
        bad_ads = [{"brand": "X", "primary_text": "text"}]  # missing fields
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(bad_ads, f)
            f.flush()
            with pytest.raises(ValueError, match="missing fields"):
                load_curated_ads(f.name)


class TestSeedCompetitorAds:
    def test_seed_inserts_all(self, db_conn, sample_ads: list[dict]) -> None:
        count = seed_competitor_ads(db_conn, sample_ads)
        assert count == 4

        rows = db_conn.execute("SELECT COUNT(*) FROM competitor_ads").fetchone()
        assert rows[0] == 4

    def test_seed_is_idempotent(self, db_conn, sample_ads: list[dict]) -> None:
        first = seed_competitor_ads(db_conn, sample_ads)
        second = seed_competitor_ads(db_conn, sample_ads)

        assert first == 4
        assert second == 0  # all duplicates

        rows = db_conn.execute("SELECT COUNT(*) FROM competitor_ads").fetchone()
        assert rows[0] == 4


class TestExtractPatterns:
    def test_top_hooks(self, sample_ads: list[dict]) -> None:
        patterns = extract_patterns(sample_ads)
        # social_proof appears 2x, others 1x
        assert patterns.top_hooks[0] == ("social_proof", 2)
        assert len(patterns.top_hooks) == 3

    def test_top_angles(self, sample_ads: list[dict]) -> None:
        patterns = extract_patterns(sample_ads)
        assert patterns.top_angles[0] == ("achievement", 2)
        assert len(patterns.top_angles) == 3

    def test_cta_buttons(self, sample_ads: list[dict]) -> None:
        patterns = extract_patterns(sample_ads)
        assert set(patterns.cta_buttons) == {"Learn More", "Sign Up", "Get Offer"}

    def test_sample_headlines_one_per_brand(self, sample_ads: list[dict]) -> None:
        patterns = extract_patterns(sample_ads)
        # 3 unique brands -> 3 headlines
        assert len(patterns.sample_headlines) == 3

    def test_empty_ads_returns_empty_patterns(self) -> None:
        patterns = extract_patterns([])
        assert patterns.top_hooks == []
        assert patterns.top_angles == []
        assert patterns.cta_buttons == []
        assert patterns.sample_headlines == []


class TestCuratedJsonIntegrity:
    """Validate the actual curated.json file."""

    def test_curated_json_structure(self) -> None:
        curated_path = Path(__file__).parent.parent / "data" / "competitor_ads" / "curated.json"
        if not curated_path.exists():
            pytest.skip("curated.json not found")

        ads = load_curated_ads(curated_path)
        assert 20 <= len(ads) <= 30

        brands = {a["brand"] for a in ads}
        assert len(brands) == 4

    def test_curated_json_pattern_extraction(self) -> None:
        curated_path = Path(__file__).parent.parent / "data" / "competitor_ads" / "curated.json"
        if not curated_path.exists():
            pytest.skip("curated.json not found")

        ads = load_curated_ads(curated_path)
        patterns = extract_patterns(ads)

        assert len(patterns.top_hooks) == 3
        assert len(patterns.top_angles) == 3
        assert len(patterns.cta_buttons) >= 3
        assert len(patterns.sample_headlines) == 4
