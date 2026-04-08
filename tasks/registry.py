from __future__ import annotations

from .easy import TASK as EASY_TASK
from .hard import TASK as HARD_TASK
from .medium import TASK as MEDIUM_TASK


TASKS = [EASY_TASK, MEDIUM_TASK, HARD_TASK]
TASK_REGISTRY = {task["id"]: task for task in TASKS}

__all__ = ["TASKS", "TASK_REGISTRY"]
