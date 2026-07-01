"""Executor / Reflector — executes a plan step-by-step with hard-coded failure recovery.

The LLM is NOT allowed to override these recovery rules. The executor applies them
deterministically based on tool name and failure context.
"""

from __future__ import annotations

import copy
from typing import Any

from spebt_agent.agent_state import AgentSession
from spebt_agent.brain.llm import LLMClient
from spebt_agent.brain.prompts import format_summarizer_prompt
from spebt_agent.tools.registry import ToolRegistry

# ── failure recovery policy ─────────────────────────────────────────────────


def _recovery_for_tool(tool_name: str, result: dict, step_inputs: dict, session: AgentSession) -> dict:
    """Apply hard-coded recovery rules for a failed tool call.

    Returns a dict with:
      - action: "abort" | "retry" | "degrade" | "skip" | "continue"
      - reason: explanation
      - modified_inputs: new inputs for retry/degrade (if applicable)
    """
    failure_msg = result.get("summary", "")
    warnings = result.get("warnings", [])

    # ── prepare_data: always abort ──
    if tool_name == "prepare_data":
        return {"action": "abort", "reason": "数据准备失败，无法继续。请检查原始数据文件是否完整。"}

    # ── health_check: abort if non-optional modules fail ──
    if tool_name == "health_check":
        # If all failed modules are optional (ProteinMPNN, ThermoMPNN, ESMFold, RaSP), continue
        # Otherwise abort for critical failures
        optional_keywords = ["proteinmpnn", "thermompnn", "esmfold", "rasp"]
        all_warnings_lower = " ".join(warnings).lower()
        # A failure is critical if it mentions any module NOT in the optional list
        # Simple heuristic: if warnings contain common critical module names (esmc, brightness, stability, candidates)
        critical_indicators = ["esmc", "brightness", "stability", "candidates", "prepare", "关键"]
        is_critical = any(indicator in all_warnings_lower for indicator in critical_indicators)
        if is_critical:
            return {"action": "abort", "reason": "关键工具模块不可用，请先运行 'python -m spebt_agent.cli.setup_tool_envs' 然后再试。"}
        else:
            return {"action": "continue", "reason": "仅可选工具模块不可用，继续运行设计（可能降级）。"}

    # ── run_design / resume_run: sophisticated recovery ──
    if tool_name in ("run_design", "resume_run"):
        return _run_design_recovery(failure_msg, step_inputs, session)

    # ── read_* tools: skip on failure (non-critical) ──
    if tool_name.startswith("read_") or tool_name.startswith("inspect_"):
        return {"action": "skip", "reason": f"读取工具失败（{failure_msg[:100]}），跳过此步继续。"}

    # ── export_submission: abort ──
    if tool_name == "export_submission":
        return {"action": "abort", "reason": f"导出提交文件失败: {failure_msg[:200]}"}

    # ── train_brightness: abort ──
    if tool_name == "train_brightness":
        return {"action": "abort", "reason": f"亮度模型训练失败: {failure_msg[:200]}"}

    # ── default: abort on unknown tools ──
    return {"action": "abort", "reason": f"工具 '{tool_name}' 执行失败: {failure_msg[:200]}"}


def _run_design_recovery(failure_msg: str, step_inputs: dict, session: AgentSession) -> dict:
    """Specific recovery logic for run_design failures."""
    msg_lower = failure_msg.lower()

    # Check if submission.csv already exists — treat as success
    artifacts = session.state.get("artifacts", {})
    if artifacts.get("submission_csv") or artifacts.get("submission"):
        return {"action": "continue", "reason": "submission.csv 已生成，即使运行中有 warning，也视为成功。"}

    # stage1_ranked failure → abort (critical model failure)
    if "stage1_ranked" in msg_lower or "stage1" in msg_lower and "rank" in msg_lower:
        return {"action": "abort", "reason": "阶段1排名失败，可能是 ESMC 模型权重缺失或亮度/稳定性评分模型不可用。请检查 model_paths.yaml 配置。"}

    # stage2_stability failure → resume → reduce esmfold_top_k → reduce max_stage2_candidates
    if "stage2_stability" in msg_lower:
        run_id = session.active_run_id
        new_inputs = copy.deepcopy(step_inputs)
        new_inputs["resume"] = True
        if run_id:
            new_inputs["run_id"] = run_id

        # First: try plain resume
        if not session.state.get("tool_history"):
            return {"action": "retry", "reason": "stage2_stability 失败，尝试从检查点恢复。", "modified_inputs": new_inputs}

        # Second: reduce esmfold_top_k
        current_esmfold = new_inputs.get("esmfold_top_k", 999)
        if current_esmfold and current_esmfold > 6:
            new_inputs["esmfold_top_k"] = max(6, current_esmfold // 2)
            return {"action": "degrade", "reason": f"降低 esmfold_top_k 到 {new_inputs['esmfold_top_k']} 重试。", "modified_inputs": new_inputs}

        # Third: reduce max_stage2_candidates
        current_max = new_inputs.get("max_stage2_candidates", 9999)
        if current_max and current_max > 800:
            new_inputs["max_stage2_candidates"] = max(800, current_max // 2)
            return {"action": "degrade", "reason": f"降低 max_stage2_candidates 到 {new_inputs['max_stage2_candidates']} 重试。", "modified_inputs": new_inputs}

        return {"action": "abort", "reason": "stage2_stability 多次降级后仍然失败，请检查 ESMFold/ThermoMPNN 环境。"}

    # stage2_brightness failure → resume → reduce max_stage2_candidates
    if "stage2_brightness" in msg_lower:
        run_id = session.active_run_id
        new_inputs = copy.deepcopy(step_inputs)
        new_inputs["resume"] = True
        if run_id:
            new_inputs["run_id"] = run_id

        # First: try plain resume
        last_tool_calls = [h for h in session.state.get("tool_history", []) if h["tool"] in ("run_design", "resume_run")]
        if len(last_tool_calls) <= 1:
            return {"action": "retry", "reason": "stage2_brightness 失败，尝试从检查点恢复。", "modified_inputs": new_inputs}

        # Second: reduce max_stage2_candidates
        current_max = new_inputs.get("max_stage2_candidates", 9999)
        if current_max and current_max > 800:
            new_inputs["max_stage2_candidates"] = max(800, current_max // 2)
            return {"action": "degrade", "reason": f"降低 max_stage2_candidates 到 {new_inputs['max_stage2_candidates']} 重试。", "modified_inputs": new_inputs}

        return {"action": "abort", "reason": "stage2_brightness 多次降级后仍然失败。"}

    # ESMFold unavailable → continue with warning (already handled inside run_design)
    if "esmfold" in msg_lower and "unavailable" in msg_lower:
        return {"action": "continue", "reason": "ESMFold 不可用，使用 ESMC+ThermoMPNN 继续（已在运行中重新加权）。"}

    # ThermoMPNN unavailable → continue
    if "thermompnn" in msg_lower and "unavailable" in msg_lower:
        return {"action": "continue", "reason": "ThermoMPNN 不可用，使用 ESMC+ESMFold 继续。"}

    # ProteinMPNN unavailable → continue
    if "proteinmpnn" in msg_lower and "unavailable" in msg_lower:
        return {"action": "continue", "reason": "ProteinMPNN 不可用，使用 stage1+combinatorial 路线继续。"}

    # Default: try resume once
    run_id = session.active_run_id
    new_inputs = copy.deepcopy(step_inputs)
    new_inputs["resume"] = True
    if run_id:
        new_inputs["run_id"] = run_id
    last_tool_calls = [h for h in session.state.get("tool_history", []) if h["tool"] in ("run_design", "resume_run")]
    if len(last_tool_calls) <= 1:
        return {"action": "retry", "reason": "运行失败，尝试从检查点恢复一次。", "modified_inputs": new_inputs}

    return {"action": "abort", "reason": f"运行失败且已重试过: {failure_msg[:200]}"}


# ── main execution loop ─────────────────────────────────────────────────────


def execute_plan(
    plan: dict,
    registry: ToolRegistry,
    session: AgentSession,
    llm_client: LLMClient,
    *,
    max_recovery_loops: int = 5,
) -> dict:
    """Execute a structured plan step by step.

    For each step:
    1. Call the tool via registry
    2. Record the result in session
    3. If ok: continue to next step
    4. If not ok: apply failure recovery → retry/degrade/skip/abort

    Args:
        plan: The structured plan from planner.generate_plan().
        registry: ToolRegistry with registered tools.
        session: AgentSession for state tracking.
        llm_client: LLM client for final summarization.
        max_recovery_loops: Maximum number of recovery actions before aborting.

    Returns:
        A dict with final status, summary, and artifacts.
    """
    steps = plan.get("steps", [])
    session.set_plan(plan)
    session.set_status("executing")

    recovery_count = 0
    step_index = 0

    while step_index < len(steps):
        step = steps[step_index]
        tool_name = step.get("tool", "")
        inputs = step.get("inputs", {})

        # inject team_name into inputs if available
        if "team_name" in inputs and not inputs["team_name"] and session.state.get("team_name"):
            inputs["team_name"] = session.state["team_name"]

        # inject run_id for resume_run
        if tool_name == "resume_run" and not inputs.get("run_id"):
            if session.active_run_id:
                inputs["run_id"] = session.active_run_id

        # Call the tool
        result = registry.call(tool_name, inputs)
        session.record_tool_call(tool_name, inputs, result)

        if result.get("ok"):
            # Success — advance to next step
            step_record = {**step, "status": "success", "result_summary": result.get("summary", "")[:300]}
            session.add_step(step_record)
            step_index += 1
            recovery_count = 0  # reset recovery counter on success
        else:
            # Failure — apply recovery policy
            recovery = _recovery_for_tool(tool_name, result, inputs, session)
            recovery_count += 1

            if recovery_count > max_recovery_loops:
                session.set_status("failed", f"超过最大恢复尝试次数 ({max_recovery_loops})，停止执行。")
                break

            action = recovery["action"]

            if action == "abort":
                session.set_status("failed", recovery["reason"])
                step_record = {**step, "status": "failed", "result_summary": recovery["reason"]}
                session.add_step(step_record)
                break

            elif action == "retry":
                # Retry with same or modified inputs
                if "modified_inputs" in recovery:
                    steps[step_index]["inputs"] = recovery["modified_inputs"]
                # Don't advance step_index — will retry same step
                continue

            elif action == "degrade":
                # Retry with degraded parameters
                if "modified_inputs" in recovery:
                    steps[step_index]["inputs"] = recovery["modified_inputs"]
                continue

            elif action == "skip":
                # Skip this step and continue
                step_record = {**step, "status": "skipped", "result_summary": recovery["reason"]}
                session.add_step(step_record)
                step_index += 1
                continue

            elif action == "continue":
                # Treat as success despite warnings
                step_record = {**step, "status": "success_with_warnings", "result_summary": recovery["reason"]}
                session.add_step(step_record)
                step_index += 1
                continue

    # ── final summary ───────────────────────────────────────────────────
    if session.status != "failed":
        session.set_status("success")

    # Generate LLM summary
    try:
        summary = _call_summarizer(llm_client, session)
    except Exception:
        summary = _build_fallback_summary(session)

    session.finalize(summary)

    return {
        "ok": session.status == "success",
        "session_id": session.session_id,
        **session.summary_dict,
        "final_summary": summary,
    }


def _call_summarizer(llm_client: LLMClient, session: AgentSession) -> str:
    """Call the LLM to generate a user-facing summary."""
    from spebt_agent.brain.llm import NullLLMClient

    if isinstance(llm_client, NullLLMClient):
        return _build_fallback_summary(session)

    prompt = format_summarizer_prompt(
        task=session.state.get("user_task", ""),
        plan=session.state.get("plan"),
        tool_history=session.state.get("tool_history", []),
        artifacts=session.state.get("artifacts", {}),
        warnings=session.state.get("warnings", []),
        status=session.state.get("status", "unknown"),
    )
    if hasattr(llm_client, "_complete"):
        return llm_client._complete(prompt)
    return llm_client.write_report({"context": prompt})


def _build_fallback_summary(session: AgentSession) -> str:
    """Build a deterministic summary when LLM is unavailable."""
    state = session.state
    status = state.get("status", "unknown")
    steps = state.get("completed_steps", [])
    tool_hist = state.get("tool_history", [])
    artifacts = state.get("artifacts", {})
    warnings = state.get("warnings", [])

    lines = [
        "# spEBT Agent 执行报告",
        "",
        f"**会话 ID**：{session.session_id}",
        f"**状态**：{status}",
        f"**任务**：{state.get('user_task', '')}",
        "",
        "## 执行步骤",
    ]
    for s in steps:
        status_label = s.get("status", "?")
        lines.append(f"- [{status_label}] {s.get('tool', s.get('id', '?'))}: {s.get('result_summary', '')[:200]}")

    if warnings:
        lines.append("\n## 警告")
        for w in warnings:
            lines.append(f"- {w}")

    if artifacts:
        lines.append("\n## 产出文件")
        for k, v in artifacts.items():
            if k.endswith("_dir") or k in ("run_id", "count", "total_count"):
                continue
            lines.append(f"- **{k}**：`{v}`")

    if status == "failed":
        lines.append(f"\n## 失败原因\n{state.get('failure_reason', '未知错误')}")
        lines.append("\n## 下一步建议")
        lines.append("- 查看上方警告信息定位问题")
        lines.append("- 使用 `python -m spebt_agent.cli.run_agent --task \"继续上一次失败的运行\"` 尝试恢复")
    else:
        lines.append("\n## 下一步建议")
        lines.append("- 查看 `outputs/latest/final_report.md` 了解详细报告")
        lines.append("- 使用 `python -m spebt_agent.cli.run_agent --task \"读取最新结果并解释\"` 查看结果解读")

    return "\n".join(lines)
