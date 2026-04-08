def grader_medium(episode_result: dict) -> float:
    rescue_score = episode_result["victims_rescued"] / max(episode_result["victims_total"], 1)
    avg_battery_used = (
        episode_result["drone_battery_used"] + episode_result["rover_battery_used"]
    ) / 2
    energy_score = max(0.0, 1.0 - avg_battery_used)
    coordination_score = episode_result.get("coordination_score", 0.0)
    score = 0.5 * rescue_score + 0.3 * energy_score + 0.2 * coordination_score
    return round(min(max(score, 0.0), 1.0), 4)
