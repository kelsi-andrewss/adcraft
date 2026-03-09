from typing import Literal

from pydantic import BaseModel, Field


class IterationRecord(BaseModel):
    """Record of a single iteration cycle."""

    source_ad_id: str
    target_ad_id: str
    cycle_number: int
    action_type: Literal["component_fix", "full_regen"]
    weak_dimension: str
    delta_scores: dict[str, float] = Field(default_factory=dict)
    token_cost: float = 0.0
