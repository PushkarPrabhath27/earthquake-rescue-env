from __future__ import annotations

from pathlib import Path

import yaml

from grader_easy import grade as grader


TASK_PATH = Path(__file__).with_name("easy.yaml")

with TASK_PATH.open("r", encoding="utf-8") as handle:
    CONFIG = yaml.safe_load(handle) or {}


TASK = {
    "id": "easy",
    "name": "Easy Rescue Sweep",
    "difficulty": "easy",
    "description": CONFIG.get("description", "Introductory rescue mission with sparse rubble and two victims."),
    "objective": CONFIG.get("objective", "Rescue all victims while establishing the drone-to-rover workflow."),
    "success_metric": CONFIG.get("success_metric", "Rescue every victim before the step budget is exhausted."),
    "config_path": "tasks/easy.yaml",
    "grader_path": "grader_easy.grade",
    "max_steps": int(CONFIG.get("max_steps", 500)),
    "reward_range": [0.0, 1.0],
    "grader": grader,
}

grade = grader

__all__ = ["TASK", "CONFIG", "grader", "grade"]
