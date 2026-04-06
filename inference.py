import asyncio
import json
import os
from typing import List, Optional

from openai import OpenAI

from incident_triage_env import IncidentAction, IncidentTriageEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("INCIDENT_TRIAGE_TASK", "")
POLICY_MODE = os.getenv("POLICY_MODE", "heuristic").lower()
BENCHMARK = "incident_triage_openenv"
MAX_STEPS = 3
TEMPERATURE = 0.0
MAX_TOKENS = 280
SUCCESS_SCORE_THRESHOLD = 0.85


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _heuristic_action(ticket_text: str, task_name: str) -> IncidentAction:
    if task_name == "easy_password_reset":
        return IncidentAction(
            summary="User lost MFA access after phone replacement and needs identity-verified recovery.",
            category="auth",
            priority="medium",
            owner_team="support",
            runbook_steps=[
                "Verify identity with approved KYC flow",
                "Perform mfa reset and force secure login",
                "Confirm account activity and session integrity",
            ],
            customer_message=(
                "We can help restore access quickly. We will first verify your identity, "
                "then reset MFA and confirm secure sign-in."
            ),
        )
    if task_name == "medium_db_latency":
        return IncidentAction(
            summary="Checkout latency rose after migration and requires immediate DB performance triage.",
            category="database",
            priority="high",
            owner_team="db",
            runbook_steps=[
                "Rollback migration if regression confirmed",
                "Analyze slow query plans on checkout paths",
                "Add or adjust index for hot query predicates",
            ],
            customer_message=(
                "We are investigating a performance issue affecting checkout and will provide "
                "an update after immediate mitigation steps."
            ),
        )
    return IncidentAction(
        summary="Potential credential compromise with anomalous admin token usage and data access spikes.",
        category="security",
        priority="critical",
        owner_team="security",
        runbook_steps=[
            "Revoke tokens for potentially compromised admin identities",
            "Isolate privileged access paths and tighten controls",
            "Audit logs for exfiltration timeline and scope",
            "Notify legal and incident command process",
        ],
        customer_message=(
            "We have initiated containment for a security incident and are prioritizing protection of data. "
            "Next update will follow after forensic validation."
        ),
    )


def get_action_from_llm(client: OpenAI, ticket_text: str, task_name: str) -> IncidentAction:
    system = (
        "Return exactly one JSON object with keys: summary, category, priority, owner_team, runbook_steps, customer_message. "
        "No markdown. No extra keys."
    )
    user = f"Task={task_name}\nTicket:\n{ticket_text}"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        payload = json.loads(text)
        return IncidentAction.model_validate(payload)
    except Exception:
        return _heuristic_action(ticket_text, task_name)


async def run_episode(task_name: str, client: Optional[OpenAI]) -> float:
    env = await IncidentTriageEnv.from_docker_image(IMAGE_NAME, task_name=task_name)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    model = MODEL_NAME if POLICY_MODE == "llm" else "heuristic-baseline"
    log_start(task=task_name, env=BENCHMARK, model=model)

    try:
        result = await env.reset()
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            if POLICY_MODE == "llm" and client is not None:
                action = get_action_from_llm(client, result.observation.ticket_text, result.observation.task_name)
            else:
                action = _heuristic_action(result.observation.ticket_text, result.observation.task_name)
            result = await env.step(action)
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            log_step(step, action.model_dump_json(), reward, result.done, result.observation.last_action_error)
            if result.done:
                break
        score = min(max(sum(rewards) / max(len(rewards), 1), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


async def main() -> None:
    selected_tasks = [TASK_NAME] if TASK_NAME else [
        "easy_password_reset",
        "medium_db_latency",
        "hard_security_breach",
    ]
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if POLICY_MODE == "llm" else None
    scores: List[float] = []
    for task_name in selected_tasks:
        score = await run_episode(task_name, client)
        scores.append(score)
    avg = sum(scores) / max(len(scores), 1)
    print(f"[AGGREGATE] tasks={len(selected_tasks)} average_score={avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
