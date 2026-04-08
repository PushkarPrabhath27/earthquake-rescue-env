---
title: Earthquake Rescue Env
emoji: "🚁"
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
license: mit
short_description: OpenEnv earthquake rescue with graded drone-rover tasks
---

# Earthquake Rescue Multimodal Robot

An OpenEnv-compatible reinforcement-learning environment for coordinated earthquake rescue. The environment simulates a post-disaster urban grid where a drone scouts victims and a ground rover executes rescue routes through partially blocked terrain. It is designed for deterministic evaluation, fast validator interaction, and deployment on Hugging Face Spaces.

## Why This Project

Real disaster response is a high-stakes coordination problem, not a toy game. This environment models a realistic multimodal workflow:

- the **drone** provides wide-area scouting, rubble bypass, and early victim discovery
- the **rover** handles precise ground navigation and performs the actual rescue action
- the **policy** must balance route quality, battery use, scouting timing, and episode speed

That makes the environment a good benchmark for:

- structured decision-making
- multi-agent coordination
- dense reward learning
- LLM-guided planning with deterministic low-level execution

## Core Environment

- Grid size: `64 x 64`
- Terrain: procedurally generated rubble map with rover-reachable victim placement
- Agents: `drone` and `rover` act every timestep
- Episode API: `/reset`, `/step`, `/state`, `/health`
- Deployment target: Hugging Face Docker Space on port `7860`
- Determinism: seeded terrain, seeded victim placement, rounded rewards and info values

## Agent Roles

### Drone

- can move over rubble
- scouts victims within a configurable detection radius
- consumes battery faster than the rover
- supports coordination actions such as `switch_to_rover`

### Rover

- can only move through clear terrain
- rescues a victim by entering the victim cell
- is the main scoring agent for mission completion
- uses exact shortest-path routing over the terrain grid

## Action Space

### Drone Actions

- `up`
- `down`
- `left`
- `right`
- `hover`
- `switch_to_rover`

### Rover Actions

- `up`
- `down`
- `left`
- `right`
- `wait`
- `switch_to_drone`

## Observation Space

Every observation returned by `/reset`, `/step`, and `/state` contains:

- `terrain_map`: `64 x 64` grid where `0 = clear`, `1 = rubble`
- `drone`: `{x, y, battery, modality}`
- `rover`: `{x, y, battery, modality}`
- `victim_signals`: list of `{id, x, y, strength, rescued, scouted}`
- `timestep`: current step index

This schema is mirrored consistently across:

- [openenv.yaml](/c:/Users/pushk/OneDrive/Documents/ScalerMetaHackathon/openenv.yaml)
- runtime responses from [api/app.py](/c:/Users/pushk/OneDrive/Documents/ScalerMetaHackathon/api/app.py)
- Pydantic models under [models/](/c:/Users/pushk/OneDrive/Documents/ScalerMetaHackathon/models)

## Reward Design

The reward function provides dense trajectory feedback, not just terminal success.

Reward components:

- distance progress toward remaining victims
- modality coordination reward
- energy efficiency reward
- loop penalty
- collision penalty
- completion reward

Reward behavior:

- total reward is clamped to `[-2.0, 3.0]`
- values are rounded to stable precision for reproducibility
- reward details are exposed in `info.reward_breakdown`

## Tasks And Graders

The environment exposes three graded tasks with increasing difficulty. Each task is declared in [openenv.yaml](/c:/Users/pushk/OneDrive/Documents/ScalerMetaHackathon/openenv.yaml) and each YAML config in [tasks/](/c:/Users/pushk/OneDrive/Documents/ScalerMetaHackathon/tasks) also includes its own grader reference so validators can discover them directly.

| Task | Goal | Rubble | Victims | Max Steps | Battery Multiplier | Grader |
| --- | --- | --- | --- | --- | --- | --- |
| `easy` | Full rescue with simple coordination | `15%` | `2` | `500` | `0.5` | `grader_easy.grader_easy` |
| `medium` | Full rescue with coordination + energy emphasis | `28%` | `4` | `430` | `0.9` | `grader_medium.grader_medium` |
| `hard` | Full rescue under tighter routing pressure | `36%` | `5` | `420` | `0.7` | `grader_hard.grader_hard` |

### Grader Summary

- `grader_easy`: rescue ratio plus step efficiency
- `grader_medium`: rescue ratio plus energy score plus coordination score
- `grader_hard`: full rescue gate plus speed score plus energy score

All graders are deterministic and clamp outputs to `[0.0, 1.0]`.

## Baseline Planner

The default [inference.py](/c:/Users/pushk/OneDrive/Documents/ScalerMetaHackathon/inference.py) is a **proxy-backed hybrid baseline**:

- it makes a required OpenAI-compatible planning call through `API_BASE_URL` and `API_KEY`
- it uses the LLM for high-level episode planning, not raw per-step control
- rover execution is deterministic and uses exact grid shortest paths
- rescue order is optimized using a small-state dynamic programming route search
- the drone follows a scouting schedule aligned to the rescue sequence

This design was chosen to satisfy the LiteLLM proxy requirement while keeping rollout behavior stable under validator execution.

### Why Hybrid Instead Of Fully LLM-Driven

- fully LLM-driven per-step control is higher variance
- deterministic path execution is faster and more reproducible
- exact routing improves hard-task completion reliability
- the LLM still contributes real strategic planning, so proxy usage is genuine

## Baseline Results

Observed local baseline scores after the latest planner and task tuning:

- `easy`: `0.93`
- `medium`: `0.75`
- `hard`: `0.75`

These are baseline heuristic-plus-LLM planning results, not supervised classification accuracy. The most important metric in this benchmark is end-of-episode grader score per task.

## API Contract

### `GET /health`

Returns lightweight service status:

```json
{"status":"ok","task":"easy"}
```

### `POST /reset`

Accepts a seed and optional `task_id`, then returns the initial observation.

Example:

```json
{"seed":42,"task_id":"medium"}
```

### `POST /step`

Accepts:

```json
{"drone":"right","rover":"down"}
```

Returns:

```json
{
  "observation": {...},
  "reward": 0.08912,
  "done": false,
  "info": {...}
}
```

### `GET /state`

Returns the latest observation without mutating the episode.

## Repository Structure

```text
.
├── api/              FastAPI service layer
├── env/              terrain, motion, reward, victim, and core env logic
├── graders/          deterministic grader implementations
├── models/           typed Pydantic request/response models
├── server/           validator-required server entrypoint
├── tasks/            task difficulty configs
├── inference.py      baseline runner with strict START/STEP/END logs
├── openenv.yaml      environment manifest
├── Dockerfile        deployment image
├── pyproject.toml    package metadata and entrypoints
└── uv.lock           locked dependency set
```

## Local Setup

Use the Python path that matches this machine:

```powershell
C:\Users\pushk\python310\python.exe -m pip install -r requirements.txt
```

Start the API locally:

```powershell
$env:PYTHONPATH="c:\Users\pushk\OneDrive\Documents\opencode\Jarvis\jarvis"
C:\Users\pushk\python310\python.exe -m uvicorn api.app:app --app-dir . --host 0.0.0.0 --port 7860
```

## Local Smoke Tests

Health:

```powershell
curl http://127.0.0.1:7860/health
```

Reset:

```powershell
curl -X POST "http://127.0.0.1:7860/reset" -H "Content-Type: application/json" -d "{\"seed\":42,\"task_id\":\"easy\"}"
```

Step:

```powershell
curl -X POST "http://127.0.0.1:7860/step" -H "Content-Type: application/json" -d "{\"drone\":\"right\",\"rover\":\"right\"}"
```

## Inference Environment Variables

Submission-compatible inference variables:

- `API_BASE_URL`: injected LiteLLM / OpenAI-compatible endpoint
- `MODEL_NAME`: model identifier used for planning
- `API_KEY`: injected API key used for the OpenAI client

Runner variables used by this repo:

- `ENV_BASE_URL`: environment endpoint, defaults to `http://127.0.0.1:7860`
- `TASK_ID`: selected task id
- `SEED`: deterministic rollout seed
- `MAX_STEPS`: inference step budget

## Deployment

### Docker

```powershell
docker build -t earthquake-rescue-env .
docker run -p 7860:7860 earthquake-rescue-env
```

### Hugging Face Space

Space page:

`https://huggingface.co/spaces/Pushkar27/earthquake-rescue-env`

Live runtime:

`https://pushkar27-earthquake-rescue-env.hf.space`

## Evaluation Notes

- the environment is deterministic for a fixed seed
- all public API responses use strict JSON shapes
- graders are importable as standalone callables
- `inference.py` emits exact `[START]`, `[STEP]`, and `[END]` lines
- the baseline makes real proxy-backed OpenAI client calls before action rollout

## Project Strengths

- realistic multimodal rescue setting instead of a toy game
- strong deterministic execution core
- explicit dense reward design
- three graded tasks with increasing difficulty
- task YAMLs are self-contained and each explicitly declares its grader callable
- validator-safe API surface
- hybrid LLM + exact routing baseline for stronger performance

## Future Improvements

- richer multi-episode benchmarking across multiple seeds
- stronger LLM route revision during long episodes
- optional learned value heuristics for rescue ordering
- visual replay tooling for trajectory inspection
