"""Shadow test: generate sample ads with pedagogical alignment.

Loads environment, generates 2-3 ads from sample briefs, and prints
the output to verify THEME.brand_name and Pedagogical North Star
are present in the generation prompt.

Usage:
    uv run python scripts/shadow_test.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from src.generate.engine import GenerationEngine  # noqa: E402
from src.models.brief import AdBrief  # noqa: E402
from src.theme import THEME  # noqa: E402

SAMPLE_BRIEFS = [
    AdBrief(
        audience_segment="parents of high school juniors",
        product_offer="1-on-1 SAT tutoring with expert instructors",
        campaign_goal="lead generation",
        tone="supportive, results-oriented",
    ),
    AdBrief(
        audience_segment="high school students preparing for the SAT",
        product_offer="personalized SAT prep with diagnostic assessments",
        campaign_goal="app installs",
        tone="energetic, motivating",
    ),
    AdBrief(
        audience_segment="parents seeking math tutoring for middle schoolers",
        product_offer="expert math tutoring for grades 6-8",
        campaign_goal="lead generation",
        tone="warm, knowledgeable",
    ),
]


def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set -- skipping live generation")
        print(f"Brand name: {THEME.brand_name}")
        print("Shadow test would generate ads with Pedagogical North Star principles.")
        return

    engine = GenerationEngine()

    for i, brief in enumerate(SAMPLE_BRIEFS, 1):
        print(f"\n{'=' * 60}")
        print(f"Brief {i}: {brief.audience_segment} / {brief.product_offer}")
        print(f"{'=' * 60}")

        ad = engine.generate(brief)
        print(f"Headline:     {ad.headline}")
        print(f"Primary text: {ad.primary_text}")
        print(f"Description:  {ad.description}")
        print(f"CTA:          {ad.cta_button}")
        print(f"Tokens:       {ad.token_count}")

    print(f"\nShadow test complete. Brand: {THEME.brand_name}")


if __name__ == "__main__":
    main()
