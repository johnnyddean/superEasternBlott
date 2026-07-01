"""Planner module — converts a natural-language task into a structured execution plan.

Uses the LLM to generate a plan, with template-based fallback when LLM is unavailable
or returns unparseable output.
"""

from __future__ import annotations

import json
import re
from typing import Any

from spebt_agent.brain.llm import LLMClient, NullLLMClient
from spebt_agent.brain.prompts import format_planner_prompt

# ── plan validation ─────────────────────────────────────────────────────────

REQUIRED_PLAN_FIELDS = ["goal", "task_type", "steps", "completion_criteria"]
VALID_TASK_TYPES = {"health_and_design", "design_run", "resume_failed_run", "inspect_results", "export_only", "train_then_design"}
VALID_ON_FAILURE = {"abort", "retry", "skip", "degrade"}


def validate_plan(plan: dict) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors = []
    for field in REQUIRED_PLAN_FIELDS:
        if field not in plan:
            errors.append(f"Missing required field: {field}")
    if plan.get("task_type") not in VALID_TASK_TYPES:
        errors.append(f"Invalid task_type: {plan.get('task_type')}. Must be one of {VALID_TASK_TYPES}")
    steps = plan.get("steps", [])
    if not isinstance(steps, list) or len(steps) == 0:
        errors.append("Plan must have at least one step.")
    for i, step in enumerate(steps):
        if "tool" not in step:
            errors.append(f"Step {i + 1}: missing 'tool' field")
        if step.get("on_failure") not in VALID_ON_FAILURE:
            errors.append(f"Step {i + 1}: invalid on_failure '{step.get('on_failure')}'. Must be one of {VALID_ON_FAILURE}")
    return errors


# ── built-in task templates (fallback when LLM can't plan) ──────────────────

TASK_TEMPLATES: dict[str, dict] = {
    "health_and_design": {
        "goal": "先检查工具环境健康状态，确认就绪后运行一轮稳定模式 GFP 蛋白设计",
        "task_type": "health_and_design",
        "assumptions": ["工具环境已安装", "数据已准备"],
        "steps": [
            {"id": "1", "tool": "health_check", "reason": "确认所有工具模块可用", "inputs": {}, "expected_output": "所有模块通过或仅有可选工具缺失", "on_failure": "abort"},
            {"id": "2", "tool": "run_design", "reason": "运行稳定模式设计流水线", "inputs": {"team_name": "", "profile": "stable", "resume": True}, "expected_output": "生成 submission.csv 和 final_report.md", "on_failure": "degrade"},
        ],
        "completion_criteria": "submission.csv 已生成，包含 6 条序列",
        "fallback_rules": ["如果 run_design 失败且卡在 stage2_stability，尝试 resume 或降级参数"],
    },
    "design_run": {
        "goal": "运行一轮稳定模式 GFP 蛋白设计",
        "task_type": "design_run",
        "assumptions": ["工具环境已就绪", "数据已准备"],
        "steps": [
            {"id": "1", "tool": "run_design", "reason": "执行设计流水线", "inputs": {"team_name": "", "profile": "stable", "resume": True}, "expected_output": "生成 submission.csv", "on_failure": "degrade"},
        ],
        "completion_criteria": "submission.csv 已生成",
        "fallback_rules": ["失败时尝试 resume，或降级参数重试"],
    },
    "resume_failed_run": {
        "goal": "从检查点恢复之前失败的运行",
        "task_type": "resume_failed_run",
        "assumptions": ["存在失败的 run_id", "配置文件未变"],
        "steps": [
            {"id": "1", "tool": "read_run_state", "reason": "查看失败位置和原因", "inputs": {}, "expected_output": "显示失败阶段和原因", "on_failure": "abort"},
            {"id": "2", "tool": "resume_run", "reason": "从检查点恢复运行", "inputs": {"run_id": "", "team_name": "", "profile": "stable"}, "expected_output": "从失败点继续并完成", "on_failure": "degrade"},
        ],
        "completion_criteria": "运行状态变为 success，submission.csv 生成",
        "fallback_rules": ["如果 resume 失败，尝试降低 max_stage2_candidates 或 esmfold_top_k 重新运行"],
    },
    "inspect_results": {
        "goal": "读取最近一次设计运行的结果并向用户解释",
        "task_type": "inspect_results",
        "assumptions": ["outputs/latest/ 存在有效结果"],
        "steps": [
            {"id": "1", "tool": "inspect_outputs_latest", "reason": "查看产出文件列表", "inputs": {}, "expected_output": "列出所有文件", "on_failure": "abort"},
            {"id": "2", "tool": "read_selected_top6", "reason": "读取最终入选的 6 条序列", "inputs": {}, "expected_output": "展示序列和得分", "on_failure": "skip"},
            {"id": "3", "tool": "read_ranked_variants", "reason": "读取排名靠前的变体", "inputs": {"top_n": 10}, "expected_output": "展示前 10 名变体", "on_failure": "skip"},
        ],
        "completion_criteria": "成功读取并展示结果",
        "fallback_rules": ["如果某个文件不存在，跳过并告知用户"],
    },
    "export_only": {
        "goal": "从已有的 selected_top6.csv 导出竞赛提交文件",
        "task_type": "export_only",
        "assumptions": ["selected_top6.csv 已存在"],
        "steps": [
            {"id": "1", "tool": "read_selected_top6", "reason": "确认入选序列", "inputs": {}, "expected_output": "展示 6 条序列", "on_failure": "abort"},
            {"id": "2", "tool": "export_submission", "reason": "导出 submission.csv", "inputs": {"selected_csv": "outputs/latest/selected_top6.csv", "team_name": ""}, "expected_output": "生成提交文件", "on_failure": "abort"},
        ],
        "completion_criteria": "submission.csv 已生成",
        "fallback_rules": ["如果 selected_top6.csv 不存在，需要先运行设计"],
    },
    "train_then_design": {
        "goal": "先训练亮度预测模型，再运行一轮设计",
        "task_type": "train_then_design",
        "assumptions": ["训练数据已准备", "ESMC 模型权重可用"],
        "steps": [
            {"id": "1", "tool": "train_brightness", "reason": "训练亮度预测器", "inputs": {"target": "brightness"}, "expected_output": "模型保存，输出 R² 和 MAE", "on_failure": "abort"},
            {"id": "2", "tool": "run_design", "reason": "使用新训练的模型跑设计", "inputs": {"team_name": "", "profile": "stable", "resume": True}, "expected_output": "生成 submission.csv", "on_failure": "degrade"},
        ],
        "completion_criteria": "模型训练完成且 submission.csv 已生成",
        "fallback_rules": ["如果训练失败，检查数据文件是否存在；如果设计失败，按标准降级策略处理"],
    },
}


def _classify_task(task: str) -> str:
    """Simple keyword-based task classification for template fallback."""
    t = task.lower()
    if any(kw in t for kw in ["健康", "health", "检查", "check", "就绪", "ready", "诊断", "diagnose"]):
        if any(kw in t for kw in ["设计", "design", "跑", "运行", "run"]) and not any(kw in t for kw in ["环境", "模块", "工具", "env", "module", "tool"]):
            return "health_and_design"
        return "health_and_design"  # "检查环境" alone → health_and_design (will abort after health_check if fails)
    if any(kw in t for kw in ["恢复", "继续", "resume", "重试", "retry", "失败"]):
        return "resume_failed_run"
    if any(kw in t for kw in ["查看", "读取", "结果", "inspect", "read", "解释", "最新"]):
        return "inspect_results"
    if any(kw in t for kw in ["导出", "export", "提交", "submission"]):
        return "export_only"
    if any(kw in t for kw in ["训练", "train", "brightness"]) and any(kw in t for kw in ["设计", "design"]):
        return "train_then_design"
    if any(kw in t for kw in ["设计", "design", "跑", "运行", "run"]):
        return "design_run"
    return "design_run"  # default


# ── main planner function ───────────────────────────────────────────────────


def generate_plan(
    task: str,
    tools: list[dict],
    llm_client: LLMClient,
    constraints: dict | None = None,
    recent_state: dict | None = None,
) -> dict:
    """Generate a structured execution plan from a natural language task.

    Args:
        task: User's natural language task.
        tools: List of tool summaries from ToolRegistry.list_tools().
        llm_client: LLM client for plan generation.
        constraints: Optional constraints dict.
        recent_state: Optional recent agent state for context.

    Returns:
        A validated plan dict.
    """
    # Try LLM-based planning first
    if not isinstance(llm_client, NullLLMClient):
        try:
            prompt = format_planner_prompt(task, constraints, recent_state)
            # Use the LLM's internal _complete or similar; for OpenAICompatibleLLMClient we call directly
            raw = _call_llm(llm_client, prompt)
            plan = _parse_plan_json(raw)
            errors = validate_plan(plan)
            if not errors:
                # merge tool list info isn't needed since plan just references tool names
                return plan
        except Exception:
            pass  # fall through to template

    # Template-based fallback
    task_type = _classify_task(task)
    template = TASK_TEMPLATES.get(task_type, TASK_TEMPLATES["design_run"])
    plan = dict(template)  # shallow copy
    plan["goal"] = f"[自动识别: {task_type}] {plan['goal']}"
    # inject team_name if provided
    team_name = (constraints or {}).get("team_name", "")
    for step in plan["steps"]:
        if "team_name" in step.get("inputs", {}):
            step["inputs"]["team_name"] = team_name
    return plan


def _call_llm(llm_client: LLMClient, prompt: str) -> str:
    """Call the LLM for plan generation. Tries _complete first, falls back to summarize_strategy."""
    if hasattr(llm_client, "_complete"):
        return llm_client._complete(prompt)
    # fallback: use existing interface
    return llm_client.summarize_strategy({"prompt": prompt})


def _parse_plan_json(raw: str) -> dict:
    """Extract JSON from LLM output, handling markdown fences and stray text."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse plan JSON from LLM output: {raw[:200]}")
