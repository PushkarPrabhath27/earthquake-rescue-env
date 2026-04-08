from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from collections import deque
from math import dist

import httpx

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graders.grader_easy import grader_easy
from graders.grader_hard import grader_hard
from graders.grader_medium import grader_medium


LLM_API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")
TASK_ID = os.environ.get("TASK_ID", "easy")
BENCHMARK = os.environ.get("BENCHMARK", "earthquake-rescue-multimodal")
SEED = int(os.environ.get("SEED", "42"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "500"))
USE_OPENAI = os.environ.get("USE_OPENAI", "0") == "1"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


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


def bfs_distance_and_next_action(
    grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]
) -> tuple[int | None, str]:
    if start == goal:
        return 0, "wait"

    queue = deque([start])
    parents = {start: None}
    distance = {start: 0}
    directions = [((1, 0), "down"), ((-1, 0), "up"), ((0, 1), "right"), ((0, -1), "left")]

    while queue:
        x, y = queue.popleft()
        for (dx, dy), _ in directions:
            nx, ny = x + dx, y + dy
            next_cell = (nx, ny)
            if not (0 <= nx < len(grid) and 0 <= ny < len(grid[0])):
                continue
            if grid[nx][ny] == 1 or next_cell in parents:
                continue
            parents[next_cell] = (x, y)
            distance[next_cell] = distance[(x, y)] + 1
            if next_cell == goal:
                queue.clear()
                break
            queue.append(next_cell)

    if goal not in parents:
        return None, "wait"

    node = goal
    while parents[node] != start:
        node = parents[node]
        if node is None:
            return None, "wait"

    dx = node[0] - start[0]
    dy = node[1] - start[1]
    mapping = {(1, 0): "down", (-1, 0): "up", (0, 1): "right", (0, -1): "left"}
    return distance[goal], mapping.get((dx, dy), "wait")


def heuristic_action(obs: dict) -> dict:
    victims = [victim for victim in obs["victim_signals"] if not victim["rescued"]]
    scouted = [victim for victim in victims if victim["scouted"]]
    unscouted = [victim for victim in victims if not victim["scouted"]]
    drone_pos = (obs["drone"]["x"], obs["drone"]["y"])
    rover_pos = (obs["rover"]["x"], obs["rover"]["y"])

    drone_action = "hover"
    if TASK_ID == "hard":
        drone_action = "switch_to_rover"
    elif unscouted:
        target = min(unscouted, key=lambda victim: dist((victim["x"], victim["y"]), drone_pos))
        drone_action = step_toward(drone_pos, (target["x"], target["y"]), "hover")
    elif scouted:
        drone_action = "switch_to_rover"

    rover_target_pool = victims if TASK_ID == "hard" else (scouted or victims)
    rover_action = "wait"
    if rover_target_pool:
        best_distance = None
        best_action = "wait"
        for victim in rover_target_pool:
            path_distance, candidate_action = bfs_distance_and_next_action(
                obs["terrain_map"], rover_pos, (victim["x"], victim["y"])
            )
            if path_distance is None:
                continue
            if best_distance is None or path_distance < best_distance:
                best_distance = path_distance
                best_action = candidate_action
        rover_action = best_action
        if rover_action == "wait" and not scouted:
            rover_action = "switch_to_drone"

    return {"drone": drone_action, "rover": rover_action}


def llm_action(obs: dict) -> dict:
    if OpenAI is None:
        raise RuntimeError("openai package is unavailable")
    client = OpenAI(base_url=LLM_API_BASE_URL, api_key=HF_TOKEN)
    prompt = (
        "You control a rescue drone and rover. "
        "Return compact JSON with keys 'drone' and 'rover'. "
        f"Observation: {json.dumps(obs, separators=(',', ':'))}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def get_action(obs: dict) -> dict:
    if USE_OPENAI:
        try:
            return llm_action(obs)
        except Exception:
            return heuristic_action(obs)
    return heuristic_action(obs)


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
    with httpx.Client(timeout=10.0) as client:
        try:
            response = client.post(f"{base}/reset", json={"seed": SEED, "task_id": TASK_ID})
            response.raise_for_status()
            obs = response.json()

            for timestep in range(1, MAX_STEPS + 1):
                action = get_action(obs)
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
                time.sleep(0.02)
        except Exception as exc:
            error_action = {"drone": "hover", "rover": "wait"}
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
