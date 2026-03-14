from datetime import date

from pydantic import BaseModel, Field


class PerformanceFeedback(BaseModel):
    """Platform performance metrics for a published ad."""

    id: str = Field(default="")
    ad_id: str
    platform: str
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend_usd: float = 0.0
    date_start: date
    date_end: date

    @property
    def ctr(self) -> float:
        if self.impressions == 0:
            return 0.0
        return self.clicks / self.impressions

    @property
    def conversion_rate(self) -> float:
        if self.clicks == 0:
            return 0.0
        return self.conversions / self.clicks
