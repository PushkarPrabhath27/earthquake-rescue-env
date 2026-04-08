from __future__ import annotations

import numpy as np


class MotionPlanner:
    DRONE_MOVES = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    ROVER_MOVES = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def move(
        self, position: list[int], action: str, terrain: np.ndarray, flying: bool
    ) -> tuple[list[int], bool]:
        stationary_actions = {"hover", "wait", "switch_to_rover", "switch_to_drone"}
        if action in stationary_actions:
            return list(position), False

        moves = self.DRONE_MOVES if flying else self.ROVER_MOVES
        dx, dy = moves[action]
        nx, ny = position[0] + dx, position[1] + dy
        if not (0 <= nx < terrain.shape[0] and 0 <= ny < terrain.shape[1]):
            return list(position), True
        if not flying and terrain[nx, ny] == 1:
            return list(position), True
        return [nx, ny], False

    @staticmethod
    def battery_cost(action: str, flying: bool, multiplier: float) -> float:
        if flying:
            costs = {
                "up": 0.012,
                "down": 0.012,
                "left": 0.012,
                "right": 0.012,
                "hover": 0.006,
                "switch_to_rover": 0.004,
            }
        else:
            costs = {
                "up": 0.008,
                "down": 0.008,
                "left": 0.008,
                "right": 0.008,
                "wait": 0.003,
                "switch_to_drone": 0.004,
            }
        return round(costs[action] * multiplier, 6)
