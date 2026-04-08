def grader_hard(episode_result: dict) -> float:
    if episode_result["victims_rescued"] < episode_result["victims_total"]:
        return 0.0
    steps = episode_result["steps_used"]
    max_steps = episode_result["max_steps"]
    speed_score = max(0.0, 1.0 - (steps / max_steps))
    avg_battery_used = (
        episode_result["drone_battery_used"] + episode_result["rover_battery_used"]
    ) / 2
    energy_score = max(0.0, 1.0 - avg_battery_used)
    score = 0.5 + 0.3 * speed_score + 0.2 * energy_score
    return round(min(max(score, 0.0), 1.0), 4)
