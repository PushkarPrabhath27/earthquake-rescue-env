TASK_REGISTRY = {
    "easy": {
        "config": "tasks/easy.yaml",
        "grader": "grader_easy.grader_easy",
    },
    "medium": {
        "config": "tasks/medium.yaml",
        "grader": "grader_medium.grader_medium",
    },
    "hard": {
        "config": "tasks/hard.yaml",
        "grader": "grader_hard.grader_hard",
    },
}

__all__ = ["TASK_REGISTRY"]
