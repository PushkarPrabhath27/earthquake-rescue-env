from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


DroneAction = Literal["up", "down", "left", "right", "hover", "switch_to_rover"]
RoverAction = Literal["up", "down", "left", "right", "wait", "switch_to_drone"]


class StepAction(BaseModel):
    drone: DroneAction
    rover: RoverAction


class ResetRequest(BaseModel):
    seed: int | None = None
    task_id: str | None = None
