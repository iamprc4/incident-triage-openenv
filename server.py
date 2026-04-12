from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from incident_triage_env import IncidentAction, IncidentTriageEnv

app = FastAPI(title="Incident Triage OpenEnv")
env = IncidentTriageEnv()

STRICT_MIN_SCORE = 0.01
STRICT_MAX_SCORE = 0.99


def _clamp(value: float) -> float:
    return min(max(float(value), STRICT_MIN_SCORE), STRICT_MAX_SCORE)


def _clamp_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp reward in the step result dict to strictly open (0, 1)."""
    if "reward" in data:
        data["reward"] = _clamp(data["reward"])
    return data


class ResetRequest(BaseModel):
    task_name: str = "easy_password_reset"


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok", "benchmark": env.benchmark_name}


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None) -> Dict[str, Any]:
    global env
    task_name = req.task_name if req else "easy_password_reset"
    env = IncidentTriageEnv(task_name=task_name)
    result = await env.reset()
    return _clamp_result(result.model_dump())


@app.post("/step")
async def step(req: StepRequest) -> Dict[str, Any]:
    try:
        action = IncidentAction.model_validate(req.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid action: {exc}") from exc
    result = await env.step(action)
    return _clamp_result(result.model_dump())


@app.get("/state")
async def state() -> Dict[str, Any]:
    return env.state().model_dump()


def main() -> None:
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860)