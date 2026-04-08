---
title: Earthquake Rescue Env
emoji: "🚁"
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: OpenEnv earthquake rescue with drone-rover agents
---

# Earthquake Rescue Multimodal Robot

OpenEnv-compatible environment for coordinated earthquake rescue using a drone and a ground rover. The project is designed for deterministic grading, FastAPI validation, Docker deployment, and Hugging Face Spaces hosting.

## Overview

- Grid size: `64x64`
- Agents: drone + rover acting every timestep
- Tasks: `easy`, `medium`, `hard`
- API endpoints: `/reset`, `/step`, `/state`, `/health`

## Action Space

- Drone: `up | down | left | right | hover | switch_to_rover`
- Rover: `up | down | left | right | wait | switch_to_drone`

## Observation Space

- `terrain_map`: `64x64` int grid where `0=clear`, `1=rubble`
- `drone`: `{x, y, battery, modality}`
- `rover`: `{x, y, battery, modality}`
- `victim_signals`: list of `{id, x, y, strength, rescued, scouted}`
- `timestep`: current step count

## Reward Components

- distance progress
- modality switching
- energy efficiency
- loop penalty
- collision penalty
- completion reward

Rewards are rounded to `6` decimals and clamped to `[-2.0, 3.0]`.

## Tasks

| Task | Rubble | Victims | Steps | Min Rescue |
| --- | --- | --- | --- | --- |
| easy | 15% | 2 | 500 | 50% |
| medium | 30% | 4 | 400 | 75% |
| hard | 40% | 5 | 350 | 100% |

## Local Run

```powershell
C:\Users\pushk\python310\python.exe -m pip install -r requirements.txt
C:\Users\pushk\python310\python.exe -m uvicorn api.app:app --app-dir . --host 0.0.0.0 --port 7860
```

## Smoke Test

```powershell
Invoke-WebRequest http://localhost:7860/health
Invoke-WebRequest -Method Post "http://localhost:7860/reset?seed=42&task_id=easy"
```

## Baseline Agent

The default `inference.py` policy is deterministic and heuristic-driven:

- drone moves toward the nearest unrescued victim and signals handoff once targets are scouted
- rover uses BFS shortest-path routing toward the nearest scouted victim, else the nearest reachable victim
- on `hard`, the baseline switches to full-rescue rover-first routing for stronger completion reliability

Required inference env vars for submission compatibility:

- `API_BASE_URL`: LLM endpoint for OpenAI-compatible calls
- `MODEL_NAME`: model identifier
- `HF_TOKEN`: API token

Environment runner vars used by this repo:

- `ENV_BASE_URL`: environment URL, defaults to `http://127.0.0.1:7860`
- `TASK_ID`: task tier
- `SEED`: rollout seed

`inference.py` emits strict `[START]`, `[STEP]`, and `[END]` stdout lines for submission logging.

## Deployment

```powershell
docker build -t earthquake-rescue-env .
docker run -p 7860:7860 earthquake-rescue-env
```

Live Space URL:

`https://pushkar27-earthquake-rescue-env.hf.space`
