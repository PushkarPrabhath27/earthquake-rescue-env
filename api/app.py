from __future__ import annotations

import sys
from pathlib import Path

import yaml
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import os


ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
DEFAULT_TASK_ID = os.getenv("TASK_ID", "easy")
DEFAULT_SEED = 42

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.core import EarthquakeRescueEnv
from models.action import ResetRequest, StepAction
from models.tasking import GradeRequest, GradeResponse, TaskDescriptor, TasksResponse
from tasks import TASKS, TASK_REGISTRY
from grader_easy import grade as grade_easy
from grader_medium import grade as grade_medium
from grader_hard import grade as grade_hard


def load_task_config(task_id: str) -> dict:
    path = TASKS_DIR / f"{task_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown task_id '{task_id}'")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


current_task_id = DEFAULT_TASK_ID
env = EarthquakeRescueEnv(current_task_id, load_task_config(current_task_id), seed=DEFAULT_SEED)
current_obs = env._build_observation()

app = FastAPI(title="Earthquake Rescue OpenEnv")


GRADE_FNS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def public_task_descriptor(task_id: str) -> TaskDescriptor:
    task = TASK_REGISTRY[task_id]
    return TaskDescriptor(
        id=task["id"],
        name=task["name"],
        difficulty=task["difficulty"],
        description=task["description"],
        objective=task["objective"],
        success_metric=task["success_metric"],
        config_path=task["config_path"],
        grader_path=task["grader_path"],
        max_steps=task["max_steps"],
        reward_range=task["reward_range"],
    )


@app.get("/")
async def root():
    return {
        "name": "earthquake-rescue-multimodal",
        "status": "ok",
        "task": current_task_id,
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/grade"],
    }


@app.post("/reset")
async def reset(
    seed: int = Query(DEFAULT_SEED),
    task_id: str | None = Query(default=None),
    payload: ResetRequest | None = Body(default=None),
):
    global env, current_obs, current_task_id

    selected_seed = payload.seed if payload and payload.seed is not None else seed
    selected_task = payload.task_id if payload and payload.task_id is not None else (task_id or current_task_id)
    task_config = load_task_config(selected_task)
    if selected_task != current_task_id:
        env = EarthquakeRescueEnv(selected_task, task_config, seed=selected_seed)
    current_task_id = selected_task
    current_obs = env.reset(seed=selected_seed, task_id=selected_task, task_config=task_config)
    return JSONResponse(content=current_obs)


@app.post("/step")
async def step(action: StepAction):
    global current_obs

    if env.done:
        raise HTTPException(status_code=400, detail="Episode ended. Call /reset.")

    observation, reward, done, info = env.step(action.model_dump())
    current_obs = observation
    payload = {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info.model_dump(mode="json"),
    }
    return JSONResponse(content=payload)


@app.get("/state")
async def state():
    return JSONResponse(content=current_obs)


@app.get("/tasks")
async def tasks() -> JSONResponse:
    payload = TasksResponse(tasks=[public_task_descriptor(task["id"]) for task in TASKS])
    return JSONResponse(content=payload.model_dump(mode="json"))


@app.get("/tasks/{task_id}")
async def task_detail(task_id: str) -> JSONResponse:
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task_id '{task_id}'")
    payload = public_task_descriptor(task_id)
    return JSONResponse(content=payload.model_dump(mode="json"))


@app.post("/grade")
async def grade(payload: GradeRequest) -> JSONResponse:
    if payload.task_id not in GRADE_FNS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id '{payload.task_id}'")
    episode_result = payload.episode_result or payload.info
    if not episode_result:
        raise HTTPException(status_code=400, detail="Provide episode_result or info for grading.")
    score = GRADE_FNS[payload.task_id](episode_result)
    response = GradeResponse(task_id=payload.task_id, score=score)
    return JSONResponse(content=response.model_dump(mode="json"))


@app.get("/health")
async def health():
    return {"status": "ok", "task": current_task_id}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "api.app:app",
        app_dir=".",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        workers=1,
    )


if __name__ == "__main__":
    main()
