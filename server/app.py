from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from incident_triage_env import IncidentAction, IncidentTriageEnv

app = FastAPI(title="Incident Triage OpenEnv")
env = IncidentTriageEnv()


class ResetRequest(BaseModel):
    task_name: str = "easy_password_reset"


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/")
async def health() -> Dict[str, str]:
    return {"status": "ok", "benchmark": env.benchmark_name}


@app.post("/reset")
async def reset(req: ResetRequest) -> Dict[str, Any]:
    global env
    env = IncidentTriageEnv(task_name=req.task_name)
    result = await env.reset()
    return result.model_dump()


@app.post("/step")
async def step(req: StepRequest) -> Dict[str, Any]:
    try:
        action = IncidentAction.model_validate(req.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid action: {exc}") from exc
    result = await env.step(action)
    return result.model_dump()


@app.get("/state")
async def state() -> Dict[str, Any]:
    return env.state().model_dump()


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
