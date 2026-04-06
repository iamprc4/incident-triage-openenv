<<<<<<< HEAD
# Incident Triage OpenEnv

A real-world OpenEnv environment for **IT incident triage**.  
Agents receive incident tickets and must decide category, priority, owner team, runbook response, and customer communication.

## Why this environment is useful

This models a real production workflow used by SRE/support/security teams:
- classify incidents accurately,
- escalate to the right team,
- generate high-quality mitigation steps,
- communicate safely with customers.

It is suitable for evaluating action quality under time pressure with dense, interpretable reward signals.

## Tasks (easy -> hard)

1. `easy_password_reset` (Auth recovery)
2. `medium_db_latency` (Migration-caused performance degradation)
3. `hard_security_breach` (Potential compromise and data exfiltration)

Each task has deterministic grading and returns a score/reward in `[0.0, 1.0]`.

## API

Environment supports the standard:
- `reset()`
- `step(action)`
- `state()`

HTTP endpoints for deployment:
- `POST /reset`
- `POST /step`
- `GET /state`

## Action space

Structured object:
- `summary: str`
- `category: network|database|auth|billing|infra|security`
- `priority: low|medium|high|critical`
- `owner_team: sre|platform|db|security|support`
- `runbook_steps: list[str]`
- `customer_message: str`

## Observation space

- `task_name, ticket_id, ticket_text`
- `context` (service metadata, SLA, impact context)
- `attempts_used, attempts_remaining`
- `target_outcome`
- `feedback_signals` (per-component partial credit)
- `last_action_error`

## Reward design

Per-step reward is the mean of six deterministic sub-scores:
- category correctness
- priority correctness
- owner team correctness
- runbook keyword coverage
- customer message quality
- summary quality

This gives dense guidance, not sparse binary-only success.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run API server:

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

## Baseline inference

`inference.py` is provided at repo root and prints required lines:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`
- `[AGGREGATE] ...` (when evaluating multiple tasks)

Run deterministic baseline over all 3 tasks:

```bash
python inference.py
```

Optional environment variables:
- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`
- `LOCAL_IMAGE_NAME`
- `POLICY_MODE` (`heuristic|llm`, default `heuristic` for reproducibility)
- `INCIDENT_TRIAGE_TASK` (`easy_password_reset|medium_db_latency|hard_security_breach`) to run a single task

The script is deterministic by default and uses a fixed heuristic policy so scores are reproducible across runs.  
Set `POLICY_MODE=llm` to query a hosted model.

## Docker

```bash
docker build -t incident-triage-openenv .
docker run --rm -p 7860:7860 incident-triage-openenv
```

## Hugging Face Spaces deployment

Use Docker Space mode and this repository root as app source.  
Health and API checks:

- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`

## Validation

Use:

```bash
./scripts/validate-submission.sh https://<your-space>.hf.space .
```

Also run:

```bash
openenv validate
```
=======
---
title: Incident Triage Openenv
emoji: 🔥
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 00e4149c1f6a12a28f63a755f5c71c76e2f8c7fc
