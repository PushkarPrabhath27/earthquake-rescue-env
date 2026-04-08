def grader_easy(episode_result: dict) -> float:
    rescued = episode_result["victims_rescued"]
    total = episode_result["victims_total"]
    steps = episode_result["steps_used"]
    max_steps = episode_result["max_steps"]
    rescue_score = rescued / max(total, 1)
    step_score = max(0.0, 1.0 - (steps / max_steps))
    score = 0.6 * rescue_score + 0.4 * step_score
    return round(min(max(score, 0.0), 1.0), 4)
