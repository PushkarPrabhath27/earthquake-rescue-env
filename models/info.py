from __future__ import annotations

from pydantic import BaseModel, Field

from .reward_model import RewardBreakdown


class StepInfo(BaseModel):
    task: str
    victims_rescued: int = Field(..., ge=0)
    victims_total: int = Field(..., ge=0)
    drone_battery: float = Field(..., ge=0.0, le=1.0)
    rover_battery: float = Field(..., ge=0.0, le=1.0)
    drone_battery_used: float = Field(..., ge=0.0, le=1.0)
    rover_battery_used: float = Field(..., ge=0.0, le=1.0)
    terminated: bool
    truncated: bool
    success: bool
    coordination_score: float = Field(..., ge=0.0, le=1.0)
    steps_used: int = Field(..., ge=0)
    max_steps: int = Field(..., gt=0)
    reward_breakdown: RewardBreakdown | None = None
