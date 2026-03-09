from pydantic import BaseModel


class AdBrief(BaseModel):
    """Structured ad brief defining the generation target."""

    audience_segment: str
    product_offer: str
    campaign_goal: str
    tone: str
    competitive_context: str = ""
