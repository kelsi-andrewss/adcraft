"""Pydantic models for multimodal ad creatives.

Shared interfaces for the visual pipeline: prompt engineering, image generation,
and visual evaluation. Used by features 15-22 of the v2 image generation scope.
"""

from pydantic import BaseModel, Field


class VisualBrief(BaseModel):
    """Structured visual prompt for image generation.

    Synthesized from approved ad copy + brand style guide by the
    VisualPromptGenerator. Contains both the positive prompt and
    negative prompt to constrain generation away from off-brand imagery.
    """

    prompt: str = Field(description="Image generation prompt text")
    negative_prompt: str = Field(description="Negative prompt to prevent off-brand imagery")
    aspect_ratio: str = Field(
        default="1:1",
        description="Image aspect ratio (1:1, 9:16, 16:9)",
    )
    resolution: str = Field(
        default="1K",
        description="Image resolution (512, 1K, 2K, 4K)",
    )
    style_refs: list[str] = Field(
        default_factory=list,
        description="Paths to style reference images for brand consistency",
    )
    placement: str = Field(
        default="feed",
        description="Ad placement (feed, stories, banner)",
    )


class ImageResult(BaseModel):
    """Result from an image generation call.

    Stores the generated image bytes (optional, may be None if only
    file_path is retained) and metadata about the generation.
    """

    image_bytes: bytes | None = Field(
        default=None,
        description="Raw image bytes (None if only file path retained)",
    )
    file_path: str = Field(description="Local filesystem path to saved image")
    model_id: str = Field(description="Image generation model used")
    cost_usd: float = Field(description="Cost of this generation call in USD")
    generation_config: dict = Field(
        default_factory=dict,
        description="Generation parameters used (aspect_ratio, resolution, etc.)",
    )
    variant_group_id: str | None = Field(
        default=None,
        description="Shared UUID4 linking all variants from the same ad",
    )
    variant_type: str | None = Field(
        default=None,
        description="Creative direction type (lifestyle, product, emotional)",
    )
