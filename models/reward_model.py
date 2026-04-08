from __future__ import annotations

from pydantic import BaseModel


class RewardBreakdown(BaseModel):
    distance_reward: float
    modality_reward: float
    energy_reward: float
    loop_penalty: float
    collision_penalty: float
    completion_reward: float
    total: float
