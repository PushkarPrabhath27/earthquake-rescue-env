from __future__ import annotations

from copy import deepcopy

import numpy as np

from models.info import StepInfo
from models.observation import AgentState, Observation, VictimSignal

from .motion import MotionPlanner
from .reward import RewardEngine
from .terrain import TerrainSimulator
from .victims import VictimGenerator


class EarthquakeRescueEnv:
    GRID_SIZE = 64

    def __init__(self, task_id: str, task_config: dict, seed: int = 42) -> None:
        self.task_id = task_id
        self.task = deepcopy(task_config)
        self.seed = int(seed)
        self.max_steps = int(self.task.get("max_steps", 500))
        self.rng = np.random.default_rng(self.seed)
        self.terrain_sim = TerrainSimulator(self.rng)
        self.victim_gen = VictimGenerator(self.rng)
        self.motion = MotionPlanner()
        self.reward_engine = RewardEngine(self.task)
        self.done = False
        self.reset(seed=self.seed)

    def reset(
        self,
        seed: int | None = None,
        task_id: str | None = None,
        task_config: dict | None = None,
    ) -> dict:
        if task_id is not None:
            self.task_id = task_id
        if task_config is not None:
            self.task = deepcopy(task_config)
            self.max_steps = int(self.task.get("max_steps", 500))
            self.reward_engine = RewardEngine(self.task)
        if seed is not None:
            self.seed = int(seed)

        self.rng = np.random.default_rng(self.seed)
        self.terrain_sim = TerrainSimulator(self.rng)
        self.victim_gen = VictimGenerator(self.rng)

        self.timestep = 0
        self.terrain = self.terrain_sim.generate(self.task.get("rubble_density", 0.25))
        self.drone_pos = [0, 0]
        self.rover_pos = [0, 2]
        self.drone_battery = 1.0
        self.rover_battery = 1.0
        self.drone_detection_radius = float(self.task.get("drone_detection_radius", 6.0))
        self.position_history = {
            "drone": {tuple(self.drone_pos)},
            "rover": {tuple(self.rover_pos)},
        }
        self.victims = self.victim_gen.place(self.terrain, int(self.task.get("victim_count", 2)))
        self.victims_total = len(self.victims)
        self.rescued_count = 0
        self.coordination_rescues = 0
        self.done = False
        self.last_info = self._build_info(False, False, False, None)
        return self._build_observation()

    def step(self, actions: dict) -> tuple[dict, float, bool, StepInfo]:
        if self.done:
            raise RuntimeError("Episode finished. Call /reset.")

        prev_state = {
            "drone_pos": list(self.drone_pos),
            "rover_pos": list(self.rover_pos),
            "victims": deepcopy(self.victims),
        }

        drone_action = actions["drone"]
        rover_action = actions["rover"]

        drone_new = list(self.drone_pos)
        rover_new = list(self.rover_pos)
        drone_collide = False
        rover_collide = False
        drone_cost = 0.0
        rover_cost = 0.0

        if self.drone_battery > 0.0:
            drone_new, drone_collide = self.motion.move(self.drone_pos, drone_action, self.terrain, flying=True)
            drone_cost = self.motion.battery_cost(
                drone_action, flying=True, multiplier=float(self.task.get("battery_drain_multiplier", 1.0))
            )
        if self.rover_battery > 0.0:
            rover_new, rover_collide = self.motion.move(self.rover_pos, rover_action, self.terrain, flying=False)
            rover_cost = self.motion.battery_cost(
                rover_action, flying=False, multiplier=float(self.task.get("battery_drain_multiplier", 1.0))
            )

        self.drone_pos = drone_new
        self.rover_pos = rover_new
        self.drone_battery = round(max(0.0, self.drone_battery - drone_cost), 6)
        self.rover_battery = round(max(0.0, self.rover_battery - rover_cost), 6)

        drone_loop = tuple(self.drone_pos) in self.position_history["drone"]
        rover_loop = tuple(self.rover_pos) in self.position_history["rover"]

        self.victim_gen.scout(self.victims, self.drone_pos, self.drone_detection_radius)
        rescued_ids = self.victim_gen.rescue(self.victims, self.rover_pos)
        rescued_lookup = {victim["id"]: victim for victim in self.victims}
        self.rescued_count += len(rescued_ids)
        self.coordination_rescues += sum(1 for victim_id in rescued_ids if rescued_lookup[victim_id]["scouted"])

        self.position_history["drone"].add(tuple(self.drone_pos))
        self.position_history["rover"].add(tuple(self.rover_pos))
        self.timestep += 1

        step_meta = {
            "actions": {"drone": drone_action, "rover": rover_action},
            "drone_collide": drone_collide,
            "rover_collide": rover_collide,
            "drone_loop": drone_loop,
            "rover_loop": rover_loop,
            "newly_rescued_count": len(rescued_ids),
        }
        reward, reward_breakdown = self.reward_engine.compute(self, prev_state, step_meta)

        terminated, truncated, success = self._check_termination()
        self.done = terminated or truncated
        observation = self._build_observation()
        info = self._build_info(terminated, truncated, success, reward_breakdown)
        self.last_info = info
        return observation, reward, self.done, info

    def _build_observation(self) -> dict:
        observation = Observation(
            terrain_map=self.terrain.astype(int).tolist(),
            drone=AgentState(
                x=self.drone_pos[0],
                y=self.drone_pos[1],
                battery=round(self.drone_battery, 6),
                modality="drone",
            ),
            rover=AgentState(
                x=self.rover_pos[0],
                y=self.rover_pos[1],
                battery=round(self.rover_battery, 6),
                modality="rover",
            ),
            victim_signals=[VictimSignal(**victim) for victim in self.victims],
            timestep=self.timestep,
        )
        return observation.model_dump(mode="json")

    def _build_info(self, terminated: bool, truncated: bool, success: bool, reward_breakdown) -> StepInfo:
        coordination_score = 0.0
        if self.rescued_count:
            coordination_score = self.coordination_rescues / self.rescued_count

        return StepInfo(
            task=self.task_id,
            victims_rescued=self.rescued_count,
            victims_total=self.victims_total,
            drone_battery=round(self.drone_battery, 6),
            rover_battery=round(self.rover_battery, 6),
            drone_battery_used=round(1.0 - self.drone_battery, 6),
            rover_battery_used=round(1.0 - self.rover_battery, 6),
            terminated=terminated,
            truncated=truncated,
            success=success,
            coordination_score=round(coordination_score, 6),
            steps_used=self.timestep,
            max_steps=self.max_steps,
            reward_breakdown=reward_breakdown,
        )

    def _check_termination(self) -> tuple[bool, bool, bool]:
        rescued_fraction = self.rescued_count / max(self.victims_total, 1)
        min_rescue_fraction = float(self.task.get("min_rescue_fraction", 1.0))
        all_rescued = self.rescued_count == self.victims_total
        batteries_empty = self.drone_battery <= 0.0 and self.rover_battery <= 0.0
        timed_out = self.timestep >= self.max_steps

        success = all_rescued or rescued_fraction >= min_rescue_fraction
        terminated = all_rescued or (success and not timed_out) or batteries_empty
        truncated = timed_out and not terminated
        return terminated, truncated, success
