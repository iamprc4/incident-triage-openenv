import asyncio

from incident_triage_env import IncidentAction, IncidentTriageEnv


def _baseline_action(task_name: str) -> IncidentAction:
    if task_name == "easy_password_reset":
        return IncidentAction(
            summary=(
                "User cannot receive MFA codes after replacing their phone and needs verified identity "
                "recovery with support assistance per policy."
            ),
            category="auth",
            priority="medium",
            owner_team="support",
            runbook_steps=["verify identity", "mfa reset", "restore access"],
            customer_message="We will help verify your identity and restore access safely.",
        )
    if task_name == "medium_db_latency":
        return IncidentAction(
            summary=(
                "Checkout API latency degraded sharply after the schema migration and needs database "
                "performance investigation on checkout paths."
            ),
            category="database",
            priority="high",
            owner_team="db",
            runbook_steps=["rollback migration", "analyze slow query", "add index"],
            customer_message="We are investigating checkout performance and will provide an update shortly.",
        )
    return IncidentAction(
        summary=(
            "Anomalous admin token use with billing table export spikes indicates a serious incident "
            "requiring immediate security containment and review."
        ),
        category="security",
        priority="critical",
        owner_team="security",
        runbook_steps=["revoke tokens", "isolate access", "audit logs", "notify legal"],
        customer_message=(
            "We have initiated containment for this security incident and will share a next update "
            "after validation."
        ),
    )


def test_reset_and_step_reward_bounds() -> None:
    async def run() -> None:
        for task_name in ("easy_password_reset", "medium_db_latency", "hard_security_breach"):
            env = IncidentTriageEnv(task_name=task_name)
            result = await env.reset()
            assert result.reward == 0.0
            assert not result.done

            action = _baseline_action(task_name)
            result2 = await env.step(action)
            assert 0.0 <= result2.reward <= 1.0
            assert isinstance(result2.done, bool)
            assert "base_reward" in result2.info
            assert "improvement_bonus" in result2.info
            assert "attempt_penalty" in result2.info

    asyncio.run(run())
