from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IncidentAction(BaseModel):
    summary: str = Field(min_length=8, max_length=240)
    category: Literal["network", "database", "auth", "billing", "infra", "security"]
    priority: Literal["low", "medium", "high", "critical"]
    owner_team: Literal["sre", "platform", "db", "security", "support"]
    runbook_steps: List[str] = Field(default_factory=list, min_length=1, max_length=6)
    customer_message: str = Field(min_length=10, max_length=500)


class IncidentObservation(BaseModel):
    task_name: str
    ticket_id: str
    ticket_text: str
    context: Dict[str, Any]
    attempts_used: int
    attempts_remaining: int
    target_outcome: str
    feedback_signals: Dict[str, float]
    last_action_error: Optional[str] = None


class StepResult(BaseModel):
    observation: IncidentObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GradeSignals(BaseModel):
    category: float = Field(ge=0.0, le=1.0)
    priority: float = Field(ge=0.0, le=1.0)
    owner_team: float = Field(ge=0.0, le=1.0)
    runbook_quality: float = Field(ge=0.0, le=1.0)
    customer_message_quality: float = Field(ge=0.0, le=1.0)
    summary_quality: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    ticket_id: str
    ticket_text: str
    context: Dict[str, Any]
    target: Dict[str, Any]
    max_attempts: int


TASKS: Dict[str, TaskSpec] = {
    "easy_password_reset": TaskSpec(
        name="easy_password_reset",
        ticket_id="INC-1001",
        ticket_text=(
            "Customer cannot log in after phone replacement. MFA codes no longer arrive. "
            "No suspicious logins. Account created yesterday."
        ),
        context={"region": "eu-west", "recent_changes": ["new device"], "sla_minutes": 120},
        target={
            "category": "auth",
            "priority": "medium",
            "owner_team": "support",
            "keywords": ["mfa reset", "verify identity"],
            "message_keywords": ["help", "verify", "restore access"],
        },
        max_attempts=2,
    ),
    "medium_db_latency": TaskSpec(
        name="medium_db_latency",
        ticket_id="INC-2042",
        ticket_text=(
            "Checkout API p95 latency jumped from 280ms to 4.1s after a schema migration. "
            "Error rate is low, but queue depth is increasing every minute."
        ),
        context={"service": "checkout", "deployment": "2026.04.06.3", "sla_minutes": 45},
        target={
            "category": "database",
            "priority": "high",
            "owner_team": "db",
            "keywords": ["rollback migration", "analyze slow query", "index"],
            "message_keywords": ["investigating", "performance", "update"],
        },
        max_attempts=3,
    ),
    "hard_security_breach": TaskSpec(
        name="hard_security_breach",
        ticket_id="INC-9007",
        ticket_text=(
            "SIEM alerts indicate anomalous admin token usage from two geographies in 8 minutes. "
            "Data export spikes observed on billing tables. Potential credential compromise."
        ),
        context={"service": "billing", "customers_impacted": 7300, "sla_minutes": 15},
        target={
            "category": "security",
            "priority": "critical",
            "owner_team": "security",
            "keywords": ["revoke tokens", "isolate access", "audit logs", "legal"],
            "message_keywords": ["containment", "security incident", "next update"],
        },
        max_attempts=3,
    ),
}


class IncidentTriageEnv:
    benchmark_name = "incident_triage_openenv"

    def __init__(self, task_name: str = "easy_password_reset") -> None:
        if task_name not in TASKS:
            raise ValueError(f"unknown task_name={task_name!r}")
        self.task_name = task_name
        self.task = TASKS[task_name]
        self._attempts_used = 0
        self._done = False
        self._last_action_error: Optional[str] = None
        self._last_signals: Dict[str, float] = {}
        self._best_reward: float = 0.0

    @classmethod
    async def from_docker_image(cls, _image_name: Optional[str] = None, task_name: str = "easy_password_reset") -> "IncidentTriageEnv":
        # Local fallback for baseline reproducibility when running without a container runtime.
        return cls(task_name=task_name)

    async def reset(self) -> StepResult:
        self._attempts_used = 0
        self._done = False
        self._last_action_error = None
        self._last_signals = {}
        self._best_reward = 0.0
        obs = self.state()
        return StepResult(observation=obs, reward=0.0, done=False, info={"reset": True})

    async def step(self, action: IncidentAction) -> StepResult:
        if self._done:
            return StepResult(
                observation=self.state(),
                reward=0.0,
                done=True,
                info={"warning": "episode_already_done"},
            )

        self._attempts_used += 1
        signals = grade_action(self.task, action)
        base_reward = sum(signals.model_dump().values()) / 6.0
        improvement_bonus = max(base_reward - self._best_reward, 0.0) * 0.15
        attempt_penalty = max(self._attempts_used - 1, 0) * 0.02
        reward = round(min(max(base_reward + improvement_bonus - attempt_penalty, 0.0), 1.0), 4)
        self._best_reward = max(self._best_reward, reward)
        self._last_signals = signals.model_dump()

        maxed_attempts = self._attempts_used >= self.task.max_attempts
        solved = reward >= 0.95
        self._done = solved or maxed_attempts
        obs = self.state()
        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={
                "solved": solved,
                "max_attempts": maxed_attempts,
                "base_reward": round(base_reward, 4),
                "improvement_bonus": round(improvement_bonus, 4),
                "attempt_penalty": round(attempt_penalty, 4),
            },
        )

    def state(self) -> IncidentObservation:
        return IncidentObservation(
            task_name=self.task.name,
            ticket_id=self.task.ticket_id,
            ticket_text=self.task.ticket_text,
            context=self.task.context,
            attempts_used=self._attempts_used,
            attempts_remaining=max(self.task.max_attempts - self._attempts_used, 0),
            target_outcome=(
                "Produce an actionable triage decision with appropriate severity, ownership, "
                "runbook steps, and customer-safe communication."
            ),
            feedback_signals=self._last_signals,
            last_action_error=self._last_action_error,
        )

    async def close(self) -> None:
        return None

    def render(self) -> str:
        return json.dumps(self.state().model_dump(), indent=2)


def _keyword_coverage(text: str, keywords: List[str], max_hits: int) -> float:
    lowered = text.lower()
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return min(hits / max_hits, 1.0) if max_hits else 1.0


def grade_action(task: TaskSpec, action: IncidentAction) -> GradeSignals:
    """Deterministic agent grader with partial credit in [0.0, 1.0]."""
    target = task.target
    runbook_text = " ".join(action.runbook_steps).lower()

    category_score = 1.0 if action.category.lower() == target["category"] else 0.0
    priority_score = 1.0 if action.priority.lower() == target["priority"] else 0.0
    owner_score = 1.0 if action.owner_team.lower() == target["owner_team"] else 0.0
    runbook_score = _keyword_coverage(runbook_text, target["keywords"], max_hits=max(len(target["keywords"]) - 1, 1))
    customer_msg_score = _keyword_coverage(
        action.customer_message,
        target["message_keywords"],
        max_hits=max(len(target["message_keywords"]) - 1, 1),
    )
    summary_score = 1.0 if len(action.summary.split()) >= 6 else 0.4

    return GradeSignals(
        category=round(category_score, 4),
        priority=round(priority_score, 4),
        owner_team=round(owner_score, 4),
        runbook_quality=round(runbook_score, 4),
        customer_message_quality=round(customer_msg_score, 4),
        summary_quality=round(summary_score, 4),
    )
