from datetime import datetime

from pydantic import BaseModel, Field


class DecisionEntry(BaseModel):
    """Decision log entry recording a system choice."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    component: str
    action: str
    rationale: str
    context: dict = Field(default_factory=dict)
    agent_id: str = ""
