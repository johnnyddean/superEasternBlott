"""Tests for the Planner module — plan generation and validation."""

import json
import pytest
from spebt_agent.brain.planner import (
    validate_plan,
    generate_plan,
    _classify_task,
    _parse_plan_json,
    TASK_TEMPLATES,
)
from spebt_agent.brain.llm import NullLLMClient


# ── plan validation ────────────────────────────────────────────────────────


def test_validate_valid_plan():
    plan = {
        "goal": "测试目标",
        "task_type": "design_run",
        "steps": [{"id": "1", "tool": "run_design", "reason": "设计", "inputs": {}, "expected_output": "完成", "on_failure": "abort"}],
        "completion_criteria": "完成",
    }
    assert validate_plan(plan) == []


def test_validate_missing_fields():
    plan = {"goal": "test"}
    errors = validate_plan(plan)
    assert len(errors) > 0
    assert any("task_type" in e for e in errors)
    assert any("steps" in e for e in errors)


def test_validate_invalid_task_type():
    plan = {
        "goal": "test",
        "task_type": "invalid_type",
        "steps": [{"id": "1", "tool": "x", "on_failure": "abort"}],
        "completion_criteria": "done",
    }
    errors = validate_plan(plan)
    assert any("task_type" in e for e in errors)


def test_validate_invalid_on_failure():
    plan = {
        "goal": "test",
        "task_type": "design_run",
        "steps": [{"id": "1", "tool": "x", "on_failure": "panic"}],
        "completion_criteria": "done",
    }
    errors = validate_plan(plan)
    assert any("on_failure" in e for e in errors)


def test_validate_empty_steps():
    plan = {
        "goal": "test",
        "task_type": "design_run",
        "steps": [],
        "completion_criteria": "done",
    }
    errors = validate_plan(plan)
    assert any("step" in e.lower() for e in errors)


def test_validate_missing_tool_in_step():
    plan = {
        "goal": "test",
        "task_type": "design_run",
        "steps": [{"id": "1", "on_failure": "abort"}],
        "completion_criteria": "done",
    }
    errors = validate_plan(plan)
    assert any("tool" in e for e in errors)


# ── task classification ────────────────────────────────────────────────────


def test_classify_health_and_design():
    assert _classify_task("先检查环境再跑设计") == "health_and_design"
    assert _classify_task("health check then run design") == "health_and_design"


def test_classify_resume():
    assert _classify_task("继续上一次失败的运行") == "resume_failed_run"
    assert _classify_task("resume the run") == "resume_failed_run"


def test_classify_inspect():
    assert _classify_task("读取最新结果并解释") == "inspect_results"
    assert _classify_task("查看结果") == "inspect_results"


def test_classify_export():
    assert _classify_task("导出提交文件") == "export_only"
    assert _classify_task("export submission") == "export_only"


def test_classify_train_then_design():
    assert _classify_task("先训练亮度模型再设计") == "train_then_design"
    assert _classify_task("train brightness then design") == "train_then_design"


def test_classify_design_default():
    assert _classify_task("跑一轮设计") == "design_run"
    assert _classify_task("run design") == "design_run"
    assert _classify_task("随便什么不匹配的文本") == "design_run"


# ── JSON parsing ───────────────────────────────────────────────────────────


def test_parse_plain_json():
    raw = '{"goal":"test","task_type":"design_run","steps":[],"completion_criteria":"done"}'
    result = _parse_plan_json(raw)
    assert result["goal"] == "test"


def test_parse_json_in_markdown_fence():
    raw = '''Here is the plan:
```json
{"goal":"test","task_type":"design_run","steps":[],"completion_criteria":"done"}
```
That's it.'''
    result = _parse_plan_json(raw)
    assert result["goal"] == "test"


def test_parse_json_brace_block():
    raw = '''some text {"goal":"test","task_type":"design_run","steps":[],"completion_criteria":"done"} more text'''
    result = _parse_plan_json(raw)
    assert result["goal"] == "test"


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError):
        _parse_plan_json("this is not json at all")


# ── template-based plan generation ─────────────────────────────────────────


def test_generate_plan_with_null_llm():
    """With NullLLMClient, falls back to template-based plan."""
    llm = NullLLMClient()
    tools = [{"name": "health_check", "description": "检查环境", "input_schema": {}, "side_effects": [], "resume_safe": True},
             {"name": "run_design", "description": "跑设计", "input_schema": {"properties": {"team_name": {"type": "string", "description": "队伍名"}}}, "side_effects": ["writes_files"], "resume_safe": True}]

    plan = generate_plan("先检查环境再跑设计", tools=tools, llm_client=llm)
    assert validate_plan(plan) == []
    assert plan["task_type"] == "health_and_design"
    assert len(plan["steps"]) == 2
    assert plan["steps"][0]["tool"] == "health_check"
    assert plan["steps"][1]["tool"] == "run_design"


def test_generate_plan_injects_team_name():
    """Team name from constraints is injected into template steps."""
    llm = NullLLMClient()
    tools = [{"name": "run_design", "description": "跑设计", "input_schema": {"properties": {"team_name": {"type": "string"}}}, "side_effects": [], "resume_safe": True}]

    plan = generate_plan("跑设计", tools=tools, llm_client=llm, constraints={"team_name": "MyTeam"})
    assert plan["steps"][0]["inputs"]["team_name"] == "MyTeam"


def test_generate_plan_design_run_template():
    llm = NullLLMClient()
    tools = [{"name": "run_design", "description": "跑设计", "input_schema": {"properties": {"team_name": {"type": "string"}}}, "side_effects": [], "resume_safe": True}]

    plan = generate_plan("run design please", tools=tools, llm_client=llm)
    assert plan["task_type"] == "design_run"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["tool"] == "run_design"


def test_all_templates_are_valid():
    """Every built-in template should pass validation."""
    for task_type, plan in TASK_TEMPLATES.items():
        errors = validate_plan(dict(plan))
        assert errors == [], f"Template '{task_type}' has validation errors: {errors}"


def test_all_templates_have_tools():
    """Template steps reference tools that make sense."""
    valid_tools = {"health_check", "run_design", "resume_run", "read_run_state",
                   "read_selected_top6", "read_ranked_variants", "export_submission",
                   "inspect_outputs_latest", "train_brightness"}
    for task_type, plan in TASK_TEMPLATES.items():
        for step in plan["steps"]:
            assert step["tool"] in valid_tools, f"Template '{task_type}' references unknown tool '{step['tool']}'"
