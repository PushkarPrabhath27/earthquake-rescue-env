from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskDescriptor(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str
    objective: str
    success_metric: str
    config_path: str
    grader_path: str
    max_steps: int = Field(..., gt=0)
    reward_range: list[float]


class TasksResponse(BaseModel):
    tasks: list[TaskDescriptor]


class GradeRequest(BaseModel):
    task_id: str
    episode_result: dict[str, Any] | None = None
    info: dict[str, Any] | None = None


class GradeResponse(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
