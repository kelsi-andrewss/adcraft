"""Brand theme system for AdCraft.

Centralizes all brand-specific values (name, colors, visual tone, constraints)
into a single Pydantic model. All prompt templates, evaluation rubrics, and
dashboard CSS derive from THEME instead of hardcoding brand strings.

Swap THEME to change the entire brand identity across the visual pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel


class BrandTheme(BaseModel):
    """Brand identity configuration for the visual pipeline."""

    brand_name: str
    primary_color: str  # hex
    secondary_color: str  # hex
    background_color: str  # hex
    accent_color: str  # hex
    text_color: str  # hex
    visual_tone: list[str]
    people_descriptors: list[str]
    setting_descriptors: list[str]
    negative_constraints: list[str]

    def hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert a hex color string to an (R, G, B) tuple."""
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    @property
    def primary_rgb(self) -> tuple[int, int, int]:
        """Primary color as (R, G, B) tuple for CSS interpolation."""
        return self.hex_to_rgb(self.primary_color)


NERDY_THEME = BrandTheme(
    brand_name="Nerdy",
    primary_color="#17E2EA",
    secondary_color="#0F0928",
    background_color="#161C2C",
    accent_color="#3C4CDB",
    text_color="#FFFFFF",
    visual_tone=["modern", "tech-forward", "approachable", "innovative"],
    people_descriptors=["diverse learners", "confident students", "engaged expressions"],
    setting_descriptors=[
        "bright learning environments",
        "modern study spaces",
        "clean tech aesthetic",
    ],
    negative_constraints=[
        "No distress, anxiety, or frustrated student depictions",
        "No dark, gloomy, or harsh lighting",
        "No cluttered or busy compositions",
        "No text overlays or typography in the image (Meta adds those separately)",
        "No stock photo cliches (thumbs up to camera, fake smiles, overly staged poses)",
        "No violent, sexual, or controversial imagery",
    ],
)

# Active theme -- swap this to change brand identity
THEME = NERDY_THEME
