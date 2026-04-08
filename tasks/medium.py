from __future__ import annotations

from pathlib import Path

import yaml

from grader_medium import grade as grader


TASK_PATH = Path(__file__).with_name("medium.yaml")

with TASK_PATH.open("r", encoding="utf-8") as handle:
    CONFIG = yaml.safe_load(handle) or {}


TASK = {
    "id": "medium",
    "name": "Coordinated Multi-Rescue",
    "difficulty": "medium",
    "description": CONFIG.get("description", "Denser rescue mission that rewards coordinated scouting and efficient energy use."),
    "objective": CONFIG.get("objective", "Rescue all victims while preserving battery and maximizing pre-rescue scouting."),
    "success_metric": CONFIG.get("success_metric", "Rescue every victim with strong coordination and energy efficiency."),
    "config_path": "tasks/medium.yaml",
    "grader_path": "grader_medium.grade",
    "max_steps": int(CONFIG.get("max_steps", 430)),
    "reward_range": [0.0, 1.0],
    "grader": grader,
}

grade = grader

__all__ = ["TASK", "CONFIG", "grader", "grade"]
