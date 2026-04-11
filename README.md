---
title: Incident Triage OpenEnv
emoji: 🚨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

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
- runbook quality: **all** required phrases (per task) must appear across `runbook_steps` (substring match, case-insensitive)
- customer message quality: **all** required phrases must appear in `customer_message`
- summary quality: combines length (targets ~12+ words) with **summary terms** that must reflect the ticket (e.g. MFA/phone, migration/latency, token/billing)

Episode success uses `reward >= 0.92` after shaping (see `incident_triage_env.py`).  
Noisy ticket text is intentional: agents must ignore distractors and still hit exact grader phrases—strong models that paraphrase everything can score **below** a careful baseline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run API server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Baseline inference

`inference.py` lives at the repo root and uses the **OpenAI** client only for LLM calls.

### Hackathon stdout format (strict)

Only these line types are printed (no extra lines such as `[AGGREGATE]`):

1. **`[START]`** once per episode: `task=`, `env=`, `model=`
2. **`[STEP]`** once per `env.step()`: `step=`, `action=`, `reward=` (2 decimals), `done=` (`true`|`false`), `error=` (raw string or `null`)
3. **`[END]`** once after `env.close()`: `success=`, `steps=`, `rewards=` (comma-separated, 2 decimals each)

Example shape:

```text
[START] task=easy_password_reset env=incident_triage_openenv model=gpt-4.1-mini
[STEP] step=1 action={...} reward=0.85 done=true error=null
[END] success=true steps=1 rewards=0.85
```

Three tasks run as **three episodes** back-to-back (each episode is START → STEP(s) → END).

### Environment variables

Per submission guidelines:

- **`API_BASE_URL`** — LLM API base URL (**has a default** in `inference.py`)
- **`MODEL_NAME`** — model id (**has a default** in `inference.py`)
- **`HF_TOKEN`** — required; script raises if missing. If the evaluator only sets **`API_KEY`** (LiteLLM proxy), that value is accepted as the client credential when `HF_TOKEN` is unset.

Optional:

- **`POLICY_MODE`** — `llm` (default) or `heuristic` (no `chat.completions` calls; no real token required)
- **`LOCAL_IMAGE_NAME`**
- **`INCIDENT_TRIAGE_TASK`** — run a single task by name

```bash
python inference.py
```

Reward values in logs are clamped to **strictly** between `0` and `1` when Phase-2-style checks apply.

## Waitlist / resubmit checklist

Official troubleshooting and guidelines: [Meta OpenEnv Hackathon: Guidelines](https://docs.google.com/document/d/1nth7bAacQOQEpVk6oIHV917YuRcLOowcSS1Ed-uNQVQ/edit?tab=t.0) (open while signed into Google if prompted).

Before you resubmit:

1. Push the same commit to **both** GitHub and your **HF Space** repo so the live Space matches GitHub.
2. Confirm `POST /reset` works **with an empty body** and with `{"task_name":"..."}`.
3. Run `inference.py` with **`HF_TOKEN`** (or **`API_KEY`** if that is what the evaluator injects), **`API_BASE_URL`**, and **`MODEL_NAME`** as required by the run; do not hardcode secrets.
4. Re-run `openenv validate` and your pre-submission script after each change.

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
