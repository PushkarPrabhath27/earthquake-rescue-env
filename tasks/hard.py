from __future__ import annotations

from pathlib import Path

import yaml

from grader_hard import grade as grader


TASK_PATH = Path(__file__).with_name("hard.yaml")

with TASK_PATH.open("r", encoding="utf-8") as handle:
    CONFIG = yaml.safe_load(handle) or {}


TASK = {
    "id": "hard",
    "name": "Full Disaster Clearance",
    "difficulty": "hard",
    "description": CONFIG.get("description", "Full-mission rescue scenario with tight routing demands and complete-clearance scoring."),
    "objective": CONFIG.get("objective", "Rescue all victims in a cluttered disaster zone as quickly as possible."),
    "success_metric": CONFIG.get("success_metric", "Rescue every victim while keeping time and battery usage low."),
    "config_path": "tasks/hard.yaml",
    "grader_path": "grader_hard.grade",
    "max_steps": int(CONFIG.get("max_steps", 420)),
    "reward_range": [0.0, 1.0],
    "grader": grader,
}

grade = grader

__all__ = ["TASK", "CONFIG", "grader", "grade"]
