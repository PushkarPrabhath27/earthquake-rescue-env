from __future__ import annotations

from math import dist

from models.reward_model import RewardBreakdown


class RewardEngine:
    def __init__(self, task_config: dict) -> None:
        self.task_config = task_config

    def compute(self, env, prev_state: dict, step_meta: dict) -> tuple[float, RewardBreakdown]:
        distance_reward = self._distance_reward(prev_state, env)
        modality_reward = self._modality_reward(env, step_meta["actions"])
        energy_reward = self._energy_reward(env)
        loop_penalty = self._loop_penalty(step_meta)
        collision_penalty = self._collision_penalty(step_meta)
        completion_reward = self._completion_reward(env, step_meta["newly_rescued_count"])

        total = (
            distance_reward
            + modality_reward
            + energy_reward
            + loop_penalty
            + collision_penalty
            + completion_reward
        )
        total = round(max(-2.0, min(3.0, total)), 6)

        breakdown = RewardBreakdown(
            distance_reward=round(distance_reward, 6),
            modality_reward=round(modality_reward, 6),
            energy_reward=round(energy_reward, 6),
            loop_penalty=round(loop_penalty, 6),
            collision_penalty=round(collision_penalty, 6),
            completion_reward=round(completion_reward, 6),
            total=total,
        )
        return total, breakdown

    def _distance_reward(self, prev_state: dict, env) -> float:
        prev_targets = [victim for victim in prev_state["victims"] if not victim["rescued"]]
        if not prev_targets:
            return 0.0

        prev_drone = min(
            dist((victim["x"], victim["y"]), tuple(prev_state["drone_pos"])) for victim in prev_targets
        )
        curr_drone = min(
            dist((victim["x"], victim["y"]), tuple(env.drone_pos)) for victim in prev_targets
        )
        prev_rover = min(
            abs(victim["x"] - prev_state["rover_pos"][0]) + abs(victim["y"] - prev_state["rover_pos"][1])
            for victim in prev_targets
        )
        curr_rover = min(
            abs(victim["x"] - env.rover_pos[0]) + abs(victim["y"] - env.rover_pos[1])
            for victim in prev_targets
        )
        return ((prev_drone - curr_drone) + (prev_rover - curr_rover)) * 0.05

    def _modality_reward(self, env, actions: dict) -> float:
        switched = actions["drone"] == "switch_to_rover" or actions["rover"] == "switch_to_drone"
        if not switched:
            return 0.0

        useful = False
        scouted_targets = [victim for victim in env.victims if victim["scouted"] and not victim["rescued"]]
        if actions["drone"] == "switch_to_rover":
            useful = any(
                abs(victim["x"] - env.rover_pos[0]) + abs(victim["y"] - env.rover_pos[1]) <= 6
                for victim in scouted_targets
            )
        elif actions["rover"] == "switch_to_drone":
            useful = not scouted_targets

        if useful:
            return 0.2
        return -float(self.task_config.get("modality_switch_cost", 0.02))

    @staticmethod
    def _energy_reward(env) -> float:
        avg_spent = ((1.0 - env.drone_battery) + (1.0 - env.rover_battery)) / 2
        expected_spend = env.timestep / max(env.max_steps, 1)
        return (expected_spend - avg_spent) * 0.05

    @staticmethod
    def _loop_penalty(step_meta: dict) -> float:
        return -0.05 * (int(step_meta["drone_loop"]) + int(step_meta["rover_loop"]))

    @staticmethod
    def _collision_penalty(step_meta: dict) -> float:
        penalty = -0.3 * int(step_meta["rover_collide"])
        penalty += -0.05 * int(step_meta["drone_collide"])
        return penalty

    @staticmethod
    def _completion_reward(env, newly_rescued_count: int) -> float:
        reward = float(newly_rescued_count)
        if env.rescued_count == env.victims_total and env.timestep < 0.75 * env.max_steps:
            reward += 2.0
        return reward
