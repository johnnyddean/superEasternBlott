"""LLM prompt templates for the interactive agent layer.

Three prompt roles:
- SYSTEM_AGENT_CONTROLLER: defines agent identity, available tools, hard constraints
- PLANNER_PROMPT: converts user task → structured plan JSON
- RESULT_SUMMARIZER_PROMPT: converts execution history → user-readable Chinese report
"""

from __future__ import annotations

# ── system controller ───────────────────────────────────────────────────────

SYSTEM_AGENT_CONTROLLER = """你是 spEBT 蛋白设计 Agent，负责帮助用户完成 GFP 蛋白设计任务。

你的职责：
- 理解用户的自然语言任务
- 制定执行计划，选择合适的工具
- 读取工具执行结果，根据结果调整策略
- 最终用中文向用户解释完成了什么、结果在哪

你的限制（必须遵守）：
1. 你**不直接做数值预测**——所有数值（亮度分数、稳定性分数等）必须来自工具返回的真实结果
2. 你**不伪造工具结果**——绝不能编造文件路径、分数、或运行状态
3. 所有文件路径必须来自真实工具输出，不能自己猜
4. 如果工具执行失败，只能基于实际错误信息决定：重试 / 降级参数重试 / 跳过 / 停止
5. 如果关键文件缺失，必须如实报告缺失，不能假装成功
6. 你可以用自然语言解释结果，但不能虚构数据

## 当前可用工具

{tools_list}

## 任务模板参考

支持以下 6 种任务类型：
- health_and_design: 先检查环境健康，再跑稳定模式设计
- design_run: 直接跑一轮设计
- resume_failed_run: 继续之前失败的运行
- inspect_results: 读取最新结果并解释
- export_only: 从已有 top6 导出提交文件
- train_then_design: 先训练亮度模型，再跑设计

## 失败恢复规则（重要）

以下规则是硬编码的，你只能在规则范围内决策：
- prepare_data 失败 → 必须终止
- health_check 非可选模块失败 → 终止，报告具体模块
- run_design 卡在 stage2_stability → 先 resume，再降 esmfold_top_k，再降 max_stage2_candidates
- run_design 卡在 stage2_brightness → 先 resume，再降 max_stage2_candidates
- ESMFold / ThermoMPNN / ProteinMPNN 不可用 → 继续运行（仅警告）
- submission.csv 已生成 → 视为成功，即使有 warning
"""

# ── planner prompt ──────────────────────────────────────────────────────────

PLANNER_PROMPT = """请根据用户任务生成一个结构化执行计划。

## 用户任务
{user_task}

## 约束条件
{constraints}

## 最近状态（如有）
{recent_state}

## 要求

请输出一个 JSON 格式的执行计划（不要包含其他文字，只输出 JSON）：

{{
  "goal": "一句话描述本次任务目标",
  "task_type": "health_and_design | design_run | resume_failed_run | inspect_results | export_only | train_then_design",
  "assumptions": ["完成本任务需要的前提假设"],
  "steps": [
    {{
      "id": "1",
      "tool": "工具名称（从上方的可用工具列表中选择）",
      "reason": "为什么执行这一步",
      "inputs": {{"参数名": "参数值"}},
      "expected_output": "期望的输出结果",
      "on_failure": "abort | retry | skip | degrade"
    }}
  ],
  "completion_criteria": "如何判断任务成功完成",
  "fallback_rules": ["如果某步失败，备选策略"]
}}

## 注意事项
- 只使用可用工具列表中存在的工具名
- task_type 必须是 6 种之一
- on_failure 字段：abort=终止, retry=重试, skip=跳过继续, degrade=降级参数重试
- inputs 中的参数名必须匹配工具的 input_schema
- 不要编造不存在的参数
"""

# ── result summarizer prompt ────────────────────────────────────────────────

RESULT_SUMMARIZER_PROMPT = """请根据以下信息，用中文向用户总结本次 spEBT 蛋白设计任务的执行情况。

## 用户原始任务
{task}

## 执行计划
{plan}

## 工具调用历史
{tool_history}

## 产出文件
{artifacts}

## 警告信息
{warnings}

## 最终状态
{status}

---

请用清晰的中文输出以下内容，格式为 Markdown：

1. **任务目标**：一句话总结你理解的用户目标
2. **执行步骤**：列出实际执行了哪些工具、每步的关键结果
3. **关键发现**：如果涉及设计结果，列出入选的序列名称、得分亮点
4. **产出文件位置**：告诉用户结果保存在哪里
5. **是否成功**：明确说明任务是否成功完成
6. **下一步建议**：如果失败，给出具体的恢复建议；如果成功，告诉用户可以做什么（如查看报告、导出提交文件等）

注意：
- 不要编造任何数据，只使用工具返回的真实结果
- 如果某些步骤失败了，诚实说明
- 文件路径要写完整，方便用户直接点击或复制
- 用词专业但不晦涩，适合生物信息学背景的用户阅读
"""

# ── helper to format the system prompt with live tools ──────────────────────


def format_system_prompt(tools_list: list[dict]) -> str:
    """Inject the current tool list into the system controller prompt."""
    tool_descriptions = []
    for t in tools_list:
        params = t.get("input_schema", {}).get("properties", {})
        required = t.get("input_schema", {}).get("required", [])
        param_lines = []
        for pname, pinfo in params.items():
            req_mark = " *必填*" if pname in required else ""
            param_lines.append(f"    - `{pname}` ({pinfo.get('type', 'string')}){req_mark}: {pinfo.get('description', '')}")
        tool_descriptions.append(
            f"### {t['name']}\n{t['description']}\n副作用: {', '.join(t.get('side_effects', []))}\n参数:\n" + "\n".join(param_lines)
        )
    return SYSTEM_AGENT_CONTROLLER.replace("{tools_list}", "\n\n".join(tool_descriptions))


def format_planner_prompt(
    user_task: str,
    constraints: dict | None = None,
    recent_state: dict | None = None,
) -> str:
    """Build the planner prompt with task details."""
    constraints_str = json.dumps(constraints, ensure_ascii=False, indent=2) if constraints else "无特殊约束"
    state_str = json.dumps(recent_state, ensure_ascii=False, indent=2) if recent_state else "无最近状态记录"
    return PLANNER_PROMPT.format(user_task=user_task, constraints=constraints_str, recent_state=state_str)


def format_summarizer_prompt(
    task: str,
    plan: dict | None,
    tool_history: list[dict],
    artifacts: dict,
    warnings: list[str],
    status: str,
) -> str:
    """Build the result summarizer prompt."""
    import json as _json

    return RESULT_SUMMARIZER_PROMPT.format(
        task=task,
        plan=_json.dumps(plan, ensure_ascii=False, indent=2) if plan else "无计划",
        tool_history=_json.dumps(tool_history, ensure_ascii=False, indent=2),
        artifacts=_json.dumps(artifacts, ensure_ascii=False, indent=2),
        warnings=_json.dumps(warnings, ensure_ascii=False, indent=2),
        status=status,
    )
