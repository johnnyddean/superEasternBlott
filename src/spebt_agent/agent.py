"""Interactive Agent — main orchestration entry point.

This is the top-level API that ties together:
- ToolRegistry (tool discovery & invocation)
- AgentSession (state tracking)
- Planner (LLM → structured plan)
- Executor (step-by-step execution with failure recovery)

Usage:
    result = run_interactive_agent("先检查环境再跑稳定模式设计", team_name="MyTeam")
    print(result["final_summary"])
"""

from __future__ import annotations

from spebt_agent.agent_state import AgentSession
from spebt_agent.brain import build_llm_client
from spebt_agent.brain.executor import execute_plan
from spebt_agent.brain.planner import generate_plan
from spebt_agent.brain.prompts import format_system_prompt
from spebt_agent.config import load_default_configs
from spebt_agent.tools.registry import ToolRegistry, build_default_registry


def run_interactive_agent(
    task: str,
    team_name: str | None = None,
    *,
    profile: str = "stable",
    session_id: str | None = None,
    max_stage2_candidates: int | None = None,
    esmfold_top_k: int | None = None,
    registry: ToolRegistry | None = None,
) -> dict:
    """Run the interactive agent on a natural-language task.

    This is the primary Python API. It:
    1. Loads the tool registry
    2. Creates an agent session
    3. Has the LLM generate a structured plan
    4. Executes the plan step by step with failure recovery
    5. Returns a structured result with summary

    Args:
        task: Natural language task description (Chinese or English).
        team_name: Team name for submission.
        profile: Design profile — "stable" (default) or "full_search".
        session_id: Optional session ID to resume an existing session.
        max_stage2_candidates: Override max stage2 candidates (for degraded runs).
        esmfold_top_k: Override ESMFold top-K (for degraded runs).
        registry: Optional pre-built ToolRegistry. If None, uses default.

    Returns:
        A dict with keys: ok, session_id, goal, task_type, steps_executed,
        tool_history_summary, final_summary, artifacts, run_ids, warnings, failure_reason.
    """
    # ── load config and LLM client ──────────────────────────────────────
    configs = load_default_configs()
    llm_cfg = configs["model_paths"].get("llm", {})
    llm_client = build_llm_client(llm_cfg)
    effective_team = team_name or configs["competition"].get("team_name", "")

    # ── build / use registry ────────────────────────────────────────────
    if registry is None:
        registry = build_default_registry()

    # ── create session ──────────────────────────────────────────────────
    session = AgentSession(
        task=task,
        team_name=effective_team,
        session_id=session_id,
        llm_model=getattr(llm_client, "model", "none"),
    )

    # ── plan ────────────────────────────────────────────────────────────
    tools_list = registry.list_tools()
    constraints = {
        "team_name": effective_team,
        "profile": profile,
        "max_stage2_candidates": max_stage2_candidates,
        "esmfold_top_k": esmfold_top_k,
    }
    plan = generate_plan(
        task=task,
        tools=tools_list,
        llm_client=llm_client,
        constraints=constraints,
        recent_state=session.summary_dict if session_id else None,
    )

    # Inject team_name into run_design/resume_run/export_submission steps
    for step in plan.get("steps", []):
        if step.get("tool") in ("run_design", "resume_run", "export_submission"):
            inputs = step.setdefault("inputs", {})
            if "team_name" in inputs and not inputs["team_name"]:
                inputs["team_name"] = effective_team

    # ── execute ─────────────────────────────────────────────────────────
    result = execute_plan(plan=plan, registry=registry, session=session, llm_client=llm_client)

    return result
