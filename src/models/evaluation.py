from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: str
    score: float
    rationale: str
    confidence: float = 1.0


class EvaluationResult(BaseModel):
    """Complete evaluation result for an ad."""

    ad_id: str
    scores: list[DimensionScore]
    weighted_average: float
    passed_threshold: bool
    hard_gate_failures: list[str] = Field(default_factory=list)
    evaluator_model: str = ""
    token_count: int = 0
