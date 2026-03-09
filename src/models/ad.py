from pydantic import BaseModel, Field


class AdCopy(BaseModel):
    """Generated ad copy for a Facebook/Instagram ad."""

    id: str = Field(default="")
    primary_text: str
    headline: str
    description: str
    cta_button: str
    brief_id: str = ""
    model_id: str = ""
    generation_config: dict = Field(default_factory=dict)
    token_count: int = 0
