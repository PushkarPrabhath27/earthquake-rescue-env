from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import httpx
import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graders.grader_easy import grader_easy
from graders.grader_hard import grader_hard
from graders.grader_medium import grader_medium


API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.environ.get("API_KEY")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")
TASK_ID = os.environ.get("TASK_ID", "easy")
BENCHMARK = os.environ.get("BENCHMARK", "earthquake-rescue-multimodal")
SEED = int(os.environ.get("SEED", "42"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "500"))

VALID_DRONE_ACTIONS = {"up", "down", "left", "right", "hover", "switch_to_rover"}
VALID_ROVER_ACTIONS = {"up", "down", "left", "right", "wait", "switch_to_drone"}

if API_KEY is None:
    raise ValueError("API_KEY environment variable is required")


def load_task_config(task_id: str) -> dict[str, Any]:
    path = ROOT / "tasks" / f"{task_id}.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


TASK_CONFIG = load_task_config(TASK_ID)
DRONE_DETECTION_RADIUS = float(TASK_CONFIG.get("drone_detection_radius", 6.0))


def required_total_rescues(total_victims: int) -> int:
    rescue_fraction = float(TASK_CONFIG.get("min_rescue_fraction", 1.0))
    return max(1, min(total_victims, math.ceil(rescue_fraction * total_victims)))


@dataclass(frozen=True)
class EpisodePlan:
    strategy: str
    rescue_order: list[int]
    scout_order: list[int]
    replan_mode: str
    llm_planned: bool


def print_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def print_step(step: int, action: dict, reward: float, done: bool, error: str | None = None) -> None:
    action_str = json.dumps(action, separators=(",", ":"), sort_keys=True)
    error_text = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_text}",
        flush=True,
    )


def print_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_csv = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_csv}",
        flush=True,
    )


def get_victim_lookup(obs: dict) -> dict[int, dict]:
    return {int(victim["id"]): victim for victim in obs["victim_signals"]}


def remaining_victim_ids(obs: dict) -> list[int]:
    return sorted(int(victim["id"]) for victim in obs["victim_signals"] if not victim["rescued"])


def step_toward(current: tuple[int, int], target: tuple[int, int], wait_action: str) -> str:
    if current[0] < target[0]:
        return "down"
    if current[0] > target[0]:
        return "up"
    if current[1] < target[1]:
        return "right"
    if current[1] > target[1]:
        return "left"
    return wait_action


class GridPlanner:
    def __init__(self, grid: list[list[int]]) -> None:
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
        self._path_cache: dict[tuple[tuple[int, int], tuple[int, int]], list[str] | None] = {}

    def shortest_path_actions(self, start: tuple[int, int], goal: tuple[int, int]) -> list[str] | None:
        key = (start, goal)
        if key in self._path_cache:
            cached = self._path_cache[key]
            return None if cached is None else list(cached)

        if start == goal:
            self._path_cache[key] = []
            return []

        queue = deque([start])
        parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
        directions = [((1, 0), "down"), ((-1, 0), "up"), ((0, 1), "right"), ((0, -1), "left")]

        while queue:
            x, y = queue.popleft()
            for (dx, dy), action in directions:
                nx, ny = x + dx, y + dy
                next_cell = (nx, ny)
                if not (0 <= nx < self.height and 0 <= ny < self.width):
                    continue
                if self.grid[nx][ny] == 1 or next_cell in parents:
                    continue
                parents[next_cell] = ((x, y), action)
                if next_cell == goal:
                    queue.clear()
                    break
                queue.append(next_cell)

        if goal not in parents:
            self._path_cache[key] = None
            return None

        actions: list[str] = []
        node = goal
        while node != start:
            parent_info = parents[node]
            if parent_info is None:
                break
            parent, action = parent_info
            actions.append(action)
            node = parent
        actions.reverse()
        self._path_cache[key] = list(actions)
        return actions

    def distance(self, start: tuple[int, int], goal: tuple[int, int]) -> int | None:
        actions = self.shortest_path_actions(start, goal)
        if actions is None:
            return None
        return len(actions)


def solve_optimal_rescue_order(
    obs: dict,
    planner: GridPlanner,
    preferred_order: list[int] | None = None,
) -> list[int]:
    victims = get_victim_lookup(obs)
    remaining = [victim_id for victim_id in remaining_victim_ids(obs) if victim_id in victims]
    if not remaining:
        return []

    rover_pos = (obs["rover"]["x"], obs["rover"]["y"])
    remaining.sort()
    preferred_rank = {victim_id: index for index, victim_id in enumerate(preferred_order or [])}

    start_distances = {}
    pair_distances = {}
    for victim_id in remaining:
        victim = victims[victim_id]
        start_distances[victim_id] = planner.distance(rover_pos, (victim["x"], victim["y"]))
    for src_id in remaining:
        src = victims[src_id]
        for dst_id in remaining:
            if src_id == dst_id:
                continue
            dst = victims[dst_id]
            pair_distances[(src_id, dst_id)] = planner.distance((src["x"], src["y"]), (dst["x"], dst["y"]))

    reachable = [victim_id for victim_id in remaining if start_distances[victim_id] is not None]
    if not reachable:
        return remaining
    if len(reachable) == 1:
        return reachable

    n = len(reachable)
    rescued_already = sum(1 for victim in obs["victim_signals"] if victim["rescued"])
    total_required = required_total_rescues(len(obs["victim_signals"]))
    target_count = max(1, min(n, total_required - rescued_already))
    index_to_victim = {index: victim_id for index, victim_id in enumerate(reachable)}
    victim_to_index = {victim_id: index for index, victim_id in index_to_victim.items()}

    dp: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
    for victim_id in reachable:
        distance = start_distances[victim_id]
        if distance is None:
            continue
        idx = victim_to_index[victim_id]
        tie_break = (preferred_rank.get(victim_id, len(reachable) + idx), victim_id)
        dp[(1 << idx, idx)] = (distance, tie_break)

    for mask in range(1, (1 << n)):
        for last in range(n):
            state = (mask, last)
            if state not in dp:
                continue
            current_cost, current_tie = dp[state]
            last_victim = index_to_victim[last]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                next_victim = index_to_victim[nxt]
                segment = pair_distances.get((last_victim, next_victim))
                if segment is None:
                    continue
                next_mask = mask | (1 << nxt)
                next_tie = current_tie + (preferred_rank.get(next_victim, len(reachable) + nxt), next_victim)
                candidate = (current_cost + segment, next_tie)
                prev = dp.get((next_mask, nxt))
                if prev is None or candidate < prev:
                    dp[(next_mask, nxt)] = candidate

    best_end = None
    for last in range(n):
        for mask in range(1, (1 << n)):
            if mask.bit_count() != target_count:
                continue
            state = (mask, last)
            if state not in dp:
                continue
            candidate = dp[state]
            if best_end is None or candidate < best_end[2]:
                best_end = (mask, last, candidate)

    if best_end is None:
        return reachable

    mask, last, _ = best_end
    order_indices: list[int] = [last]
    while mask:
        if mask == (1 << last):
            break
        last_victim = index_to_victim[last]
        found_prev = None
        prev_mask = mask ^ (1 << last)
        for prev in range(n):
            if not (prev_mask & (1 << prev)):
                continue
            prev_victim = index_to_victim[prev]
            segment = pair_distances.get((prev_victim, last_victim))
            if segment is None:
                continue
            prev_state = (prev_mask, prev)
            if prev_state not in dp:
                continue
            prev_cost, prev_tie = dp[prev_state]
            candidate = (prev_cost + segment, prev_tie + (preferred_rank.get(last_victim, len(reachable) + last), last_victim))
            if candidate == dp[(mask, last)]:
                found_prev = prev
                break
        if found_prev is None:
            break
        order_indices.append(found_prev)
        mask = prev_mask
        last = found_prev

    order_indices.reverse()
    order = [index_to_victim[index] for index in order_indices]
    return order


def build_planning_summary(obs: dict, planner: GridPlanner, deterministic_order: list[int]) -> dict[str, Any]:
    victims = get_victim_lookup(obs)
    rover_pos = (obs["rover"]["x"], obs["rover"]["y"])
    drone_pos = (obs["drone"]["x"], obs["drone"]["y"])

    victim_summaries = []
    for victim_id in remaining_victim_ids(obs):
        victim = victims[victim_id]
        rover_distance = planner.distance(rover_pos, (victim["x"], victim["y"]))
        drone_moves_to_scout = max(
            0,
            math.ceil(math.dist(drone_pos, (victim["x"], victim["y"])) - DRONE_DETECTION_RADIUS),
        )
        victim_summaries.append(
            {
                "id": victim_id,
                "x": victim["x"],
                "y": victim["y"],
                "scouted": victim["scouted"],
                "rescued": victim["rescued"],
                "rover_distance": rover_distance,
                "drone_moves_to_scout": drone_moves_to_scout,
            }
        )

    route_cost = 0
    current = rover_pos
    for victim_id in deterministic_order:
        victim = victims[victim_id]
        segment = planner.distance(current, (victim["x"], victim["y"]))
        route_cost += segment or 0
        current = (victim["x"], victim["y"])

    return {
        "task_id": TASK_ID,
        "benchmark": BENCHMARK,
        "drone": {
            "position": list(drone_pos),
            "battery": round(float(obs["drone"]["battery"]), 6),
        },
        "rover": {
            "position": list(rover_pos),
            "battery": round(float(obs["rover"]["battery"]), 6),
        },
        "task_hints": {
            "drone_detection_radius": DRONE_DETECTION_RADIUS,
            "max_steps": int(TASK_CONFIG.get("max_steps", MAX_STEPS)),
            "min_rescue_fraction": float(TASK_CONFIG.get("min_rescue_fraction", 1.0)),
        },
        "candidate_plan": {
            "rescue_order": deterministic_order,
            "estimated_rover_path_cost": route_cost,
            "focus": "speed" if TASK_ID == "hard" else "coordination",
        },
        "victims": victim_summaries,
        "response_schema": {
            "strategy": "string",
            "rescue_order": "list[int]",
            "scout_order": "list[int]",
            "replan_mode": "string",
        },
    }


def extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(raw[start : end + 1])


def sanitize_plan_response(response: dict[str, Any], deterministic_order: list[int]) -> EpisodePlan:
    valid_ids = set(deterministic_order)

    def normalize_id_list(key: str, default: list[int]) -> list[int]:
        value = response.get(key, default)
        if not isinstance(value, list):
            value = default
        cleaned: list[int] = []
        for item in value:
            try:
                victim_id = int(item)
            except (TypeError, ValueError):
                continue
            if victim_id in valid_ids and victim_id not in cleaned:
                cleaned.append(victim_id)
        for victim_id in default:
            if victim_id not in cleaned:
                cleaned.append(victim_id)
        return cleaned

    strategy = response.get("strategy", "hybrid_route")
    if not isinstance(strategy, str):
        strategy = "hybrid_route"
    replan_mode = response.get("replan_mode", "on_target_completion")
    if not isinstance(replan_mode, str):
        replan_mode = "on_target_completion"

    rescue_order = normalize_id_list("rescue_order", deterministic_order)
    if TASK_ID == "hard":
        scout_order: list[int] = []
    else:
        scout_order = normalize_id_list("scout_order", rescue_order)

    return EpisodePlan(
        strategy=strategy.strip() or "hybrid_route",
        rescue_order=rescue_order,
        scout_order=scout_order,
        replan_mode=replan_mode.strip() or "on_target_completion",
        llm_planned=True,
    )


def request_episode_plan(obs: dict, deterministic_order: list[int], planner: GridPlanner) -> EpisodePlan:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    summary = build_planning_summary(obs, planner, deterministic_order)
    system_prompt = (
        "You are planning a rescue episode for a drone and rover team. "
        "Return JSON only with keys strategy, rescue_order, scout_order, replan_mode. "
        "Use only victim ids listed in the summary."
    )

    last_error: Exception | None = None
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(summary, separators=(",", ":"), sort_keys=True)},
                ],
            )
            message = response.choices[0].message.content or ""
            plan_json = extract_json_object(message)
            return sanitize_plan_response(plan_json, deterministic_order)
        except Exception as exc:  # pragma: no cover - network/provider dependent
            last_error = exc

    raise RuntimeError(f"Failed to obtain valid proxy-backed plan: {last_error}")


def prune_plan(order: list[int], obs: dict) -> list[int]:
    remaining = remaining_victim_ids(obs)
    pruned = [victim_id for victim_id in order if victim_id in remaining]
    for victim_id in remaining:
        if victim_id not in pruned:
            pruned.append(victim_id)
    return pruned


def compute_drone_action(obs: dict, plan: EpisodePlan) -> str:
    if obs["drone"]["battery"] <= 0:
        return "switch_to_rover"
    if TASK_ID == "hard":
        return "switch_to_rover"

    victims = get_victim_lookup(obs)
    drone_pos = (obs["drone"]["x"], obs["drone"]["y"])
    scout_order = prune_plan(plan.scout_order, obs)

    for victim_id in scout_order:
        victim = victims[victim_id]
        if victim["rescued"] or victim["scouted"]:
            continue
        if math.dist(drone_pos, (victim["x"], victim["y"])) <= DRONE_DETECTION_RADIUS:
            continue
        return step_toward(drone_pos, (victim["x"], victim["y"]), "switch_to_rover")

    return "switch_to_rover"


def compute_rover_action(obs: dict, planner: GridPlanner, plan: EpisodePlan) -> str:
    victims = get_victim_lookup(obs)
    rescue_order = prune_plan(plan.rescue_order, obs)
    if not rescue_order:
        return "wait"

    rover_pos = (obs["rover"]["x"], obs["rover"]["y"])
    drone_pos = (obs["drone"]["x"], obs["drone"]["y"])
    target = victims[rescue_order[0]]
    target_pos = (target["x"], target["y"])
    path = planner.shortest_path_actions(rover_pos, target_pos)
    if path is None:
        fallback_order = solve_optimal_rescue_order(obs, planner)
        if not fallback_order:
            return "wait"
        target = victims[fallback_order[0]]
        target_pos = (target["x"], target["y"])
        path = planner.shortest_path_actions(rover_pos, target_pos)
        if path is None:
            return "wait"

    if TASK_ID == "medium" and not target["scouted"] and obs["drone"]["battery"] > 0:
        drone_moves_to_scout = max(0, math.ceil(math.dist(drone_pos, target_pos) - DRONE_DETECTION_RADIUS))
        if len(path) <= max(1, drone_moves_to_scout):
            return "switch_to_drone"

    if not path:
        return "wait"
    return path[0]


def validate_action(action: dict[str, str]) -> dict[str, str]:
    drone_action = action.get("drone", "switch_to_rover")
    rover_action = action.get("rover", "wait")
    if drone_action not in VALID_DRONE_ACTIONS:
        drone_action = "switch_to_rover"
    if rover_action not in VALID_ROVER_ACTIONS:
        rover_action = "wait"
    return {"drone": drone_action, "rover": rover_action}


def get_action(obs: dict, planner: GridPlanner, plan: EpisodePlan) -> dict[str, str]:
    preferred = prune_plan(plan.rescue_order, obs)
    deterministic_order = solve_optimal_rescue_order(obs, planner, preferred_order=preferred)
    execution_plan = EpisodePlan(
        strategy=plan.strategy,
        rescue_order=deterministic_order,
        scout_order=prune_plan(plan.scout_order or deterministic_order, obs),
        replan_mode=plan.replan_mode,
        llm_planned=plan.llm_planned,
    )
    return validate_action(
        {
            "drone": compute_drone_action(obs, execution_plan),
            "rover": compute_rover_action(obs, planner, execution_plan),
        }
    )


def compute_score(task_id: str, info: dict) -> float:
    if task_id == "medium":
        return grader_medium(info)
    if task_id == "hard":
        return grader_hard(info)
    return grader_easy(info)


def run() -> None:
    base = ENV_BASE_URL.rstrip("/")
    rewards: list[float] = []
    info: dict = {}
    steps_taken = 0
    success = False
    score = 0.0
    print_start(TASK_ID)
    with httpx.Client(timeout=20.0) as client:
        try:
            response = client.post(f"{base}/reset", json={"seed": SEED, "task_id": TASK_ID})
            response.raise_for_status()
            obs = response.json()
            planner = GridPlanner(obs["terrain_map"])
            deterministic_order = solve_optimal_rescue_order(obs, planner)
            plan = request_episode_plan(obs, deterministic_order, planner)

            for timestep in range(1, MAX_STEPS + 1):
                action = get_action(obs, planner, plan)
                response = client.post(f"{base}/step", json=action)
                response.raise_for_status()
                payload = response.json()
                obs = payload["observation"]
                reward = float(payload["reward"])
                done = bool(payload["done"])
                info = payload["info"]
                rewards.append(reward)
                steps_taken = timestep
                print_step(timestep, action, reward, done)
                if done:
                    success = bool(info.get("success", False))
                    break
        except Exception as exc:
            error_action = {"drone": "switch_to_rover", "rover": "wait"}
            print_step(max(steps_taken, 1), error_action, 0.0, True, str(exc))
            success = False
        finally:
            if info:
                score = compute_score(TASK_ID, info)
                steps_taken = steps_taken or int(info.get("steps_used", 0))
                success = success or bool(info.get("success", False))
            print_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    run()
