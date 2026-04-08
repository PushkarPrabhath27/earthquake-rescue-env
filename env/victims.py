from __future__ import annotations

from math import dist

import numpy as np

from .terrain import TerrainSimulator


class VictimGenerator:
    START_BLOCKED = {(x, y) for x in range(3) for y in range(4)}

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def place(self, terrain: np.ndarray, count: int) -> list[dict]:
        reachable = TerrainSimulator.reachable_cells(terrain)
        candidates = [
            cell
            for cell in reachable
            if cell not in self.START_BLOCKED and cell[0] < 61 and cell[1] < 61
        ]
        if len(candidates) < count:
            raise ValueError("Not enough reachable cells to place victims")

        chosen_indexes = self.rng.choice(len(candidates), size=count, replace=False)
        chosen_positions = sorted(candidates[int(index)] for index in chosen_indexes)

        victims: list[dict] = []
        for victim_id, (x, y) in enumerate(chosen_positions):
            victims.append(
                {
                    "id": victim_id,
                    "x": x,
                    "y": y,
                    "strength": round(float(self.rng.uniform(0.55, 1.0)), 6),
                    "rescued": False,
                    "scouted": False,
                }
            )
        return victims

    @staticmethod
    def scout(victims: list[dict], drone_pos: list[int], radius: float) -> int:
        newly_scouted = 0
        for victim in victims:
            if victim["rescued"] or victim["scouted"]:
                continue
            if dist((victim["x"], victim["y"]), tuple(drone_pos)) <= radius:
                victim["scouted"] = True
                newly_scouted += 1
        return newly_scouted

    @staticmethod
    def rescue(victims: list[dict], rover_pos: list[int]) -> list[int]:
        rescued_ids: list[int] = []
        for victim in victims:
            if victim["rescued"]:
                continue
            if victim["x"] == rover_pos[0] and victim["y"] == rover_pos[1]:
                victim["rescued"] = True
                rescued_ids.append(victim["id"])
        return rescued_ids
