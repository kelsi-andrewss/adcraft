"""Competitive intelligence analyzer.

Loads curated competitor ads, seeds the database, and extracts recurring
hook/CTA/emotional patterns for injection into generation prompts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.db.queries import insert_competitor_ad
from src.decisions.logger import log_decision

REQUIRED_FIELDS = {
    "brand", "primary_text", "headline", "cta_button", "hook_type", "emotional_angle"
}


@dataclass
class CompetitorPatterns:
    """Extracted patterns from competitor ad analysis."""

    top_hooks: list[tuple[str, int]] = field(default_factory=list)
    top_angles: list[tuple[str, int]] = field(default_factory=list)
    cta_buttons: list[str] = field(default_factory=list)
    sample_headlines: list[str] = field(default_factory=list)


def load_curated_ads(json_path: str | Path) -> list[dict]:
    """Read curated.json and validate that every entry has the required fields."""
    path = Path(json_path)
    with path.open() as f:
        ads = json.load(f)

    for i, ad in enumerate(ads):
        missing = REQUIRED_FIELDS - set(ad.keys())
        if missing:
            raise ValueError(f"Ad at index {i} missing fields: {missing}")

    return ads


def seed_competitor_ads(db_conn: sqlite3.Connection, ads: list[dict]) -> int:
    """Upsert curated ads into the competitor_ads table.

    Idempotent: skips duplicates by hashing primary_text.
    Returns the number of newly inserted ads.
    """
    seen_hashes: set[str] = set()

    # Collect existing primary_text hashes to avoid duplicates
    db_conn.row_factory = sqlite3.Row
    existing = db_conn.execute("SELECT primary_text FROM competitor_ads").fetchall()
    for row in existing:
        if row["primary_text"]:
            seen_hashes.add(hashlib.sha256(row["primary_text"].encode()).hexdigest())

    inserted = 0
    for ad in ads:
        text_hash = hashlib.sha256(ad["primary_text"].encode()).hexdigest()
        if text_hash in seen_hashes:
            continue

        insert_competitor_ad(
            db_conn,
            brand=ad["brand"],
            primary_text=ad["primary_text"],
            headline=ad["headline"],
            cta_button=ad["cta_button"],
            hook_type=ad["hook_type"],
            emotional_angle=ad["emotional_angle"],
        )
        seen_hashes.add(text_hash)
        inserted += 1

    log_decision(
        "intel",
        "seeded_competitor_ads",
        f"Seeded {inserted} new competitor ads ({len(ads)} total in dataset, "
        f"{len(ads) - inserted} duplicates skipped)",
        {"inserted": inserted, "total": len(ads)},
        conn=db_conn,
    )

    return inserted


def extract_patterns(ads: list[dict]) -> CompetitorPatterns:
    """Count hook_type and emotional_angle frequencies, identify top patterns.

    Returns a CompetitorPatterns dataclass with the top 3 hooks, top 3 angles,
    all unique CTA buttons, and one strong headline per brand.
    """
    hook_counts: Counter[str] = Counter()
    angle_counts: Counter[str] = Counter()
    cta_set: set[str] = set()
    brand_headlines: dict[str, str] = {}

    for ad in ads:
        hook_counts[ad["hook_type"]] += 1
        angle_counts[ad["emotional_angle"]] += 1
        cta_set.add(ad["cta_button"])

        # Keep first headline per brand as the sample
        brand = ad["brand"]
        if brand not in brand_headlines:
            brand_headlines[brand] = ad["headline"]

    patterns = CompetitorPatterns(
        top_hooks=hook_counts.most_common(3),
        top_angles=angle_counts.most_common(3),
        cta_buttons=sorted(cta_set),
        sample_headlines=list(brand_headlines.values()),
    )

    log_decision(
        "intel",
        "extracted_patterns",
        f"Extracted patterns from {len(ads)} competitor ads: "
        f"top hooks={[h for h, _ in patterns.top_hooks]}, "
        f"top angles={[a for a, _ in patterns.top_angles]}, "
        f"{len(patterns.cta_buttons)} unique CTAs, "
        f"{len(patterns.sample_headlines)} brand headlines",
        {
            "ad_count": len(ads),
            "hook_counts": dict(hook_counts),
            "angle_counts": dict(angle_counts),
            "cta_buttons": patterns.cta_buttons,
        },
    )

    return patterns
