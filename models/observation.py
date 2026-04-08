from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    x: int = Field(..., ge=0, le=63)
    y: int = Field(..., ge=0, le=63)
    battery: float = Field(..., ge=0.0, le=1.0)
    modality: Literal["drone", "rover"]


class VictimSignal(BaseModel):
    id: int = Field(..., ge=0)
    x: int = Field(..., ge=0, le=63)
    y: int = Field(..., ge=0, le=63)
    strength: float = Field(..., ge=0.0, le=1.0)
    rescued: bool = False
    scouted: bool = False


class Observation(BaseModel):
    terrain_map: List[List[int]]
    drone: AgentState
    rover: AgentState
    victim_signals: List[VictimSignal]
    timestep: int = Field(..., ge=0)
