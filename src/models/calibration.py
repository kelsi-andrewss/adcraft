from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DriftAlert(BaseModel):
    """A single drift detection alert."""

    alert_type: Literal["alpha_drift", "mae_drift"]
    message: str
    detail: dict = Field(default_factory=dict)


class CalibrationResult(BaseModel):
    """Result from a calibration run with inter-rater reliability metrics."""

    alpha_overall: float
    spearman_rho: float
    per_dimension_mae: dict[str, float]
    passed: bool
    ad_count: int
    model_version: str
    timestamp: str = ""
    details: dict = Field(default_factory=dict)
