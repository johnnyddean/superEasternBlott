"""Tests for the interactive agent CLI and orchestration."""

import json
import sys
import pytest
from spebt_agent.agent import run_interactive_agent
from spebt_agent.brain.llm import NullLLMClient
from spebt_agent.tools.registry import ToolRegistry, Tool, tool_result


# ── helpers ────────────────────────────────────────────────────────────────


def _mock_registry_with_tools(tool_names: list[str]) -> ToolRegistry:
    """Build a registry with simple mock tools that always succeed."""
    registry = ToolRegistry()
    for name in tool_names:

        def make_callable(n):
            def _call(**kw):
                return tool_result(True, f"{n} completed", artifacts={f"{n}_result": "ok"}, warnings=[], next_hints=[])

            return _call

        registry.register(
            Tool(
                name=name,
                description=f"Mock {name}",
                input_schema={"type": "object", "properties": {}},
                callable=make_callable(name),
                side_effects=[],
                resume_safe=True,
            )
        )
    return registry


# ── agent orchestration tests ──────────────────────────────────────────────


def test_run_agent_health_and_design(tmp_path, monkeypatch):
    """Agent correctly identifies health_and_design and executes tools."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    # Ensure outputs directory exists
    (tmp_path / "outputs" / "agent_sessions").mkdir(parents=True, exist_ok=True)

    registry = _mock_registry_with_tools(["health_check", "run_design"])

    result = run_interactive_agent(
        task="先检查环境再跑稳定设计",
        team_name="TestTeam",
        profile="stable",
        registry=registry,
    )

    assert result["ok"] is True
    assert result["task_type"] == "health_and_design"
    assert len(result["steps_executed"]) == 2
    assert "health_check" in result["steps_executed"]
    assert "run_design" in result["steps_executed"]
    assert result["session_id"] is not None
    assert len(result["run_ids"]) == 0  # mock tools don't produce run_ids
    assert result["final_summary"] != ""


def test_run_agent_design_run(tmp_path, monkeypatch):
    """Agent identifies design_run and executes single tool."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    (tmp_path / "outputs" / "agent_sessions").mkdir(parents=True, exist_ok=True)

    registry = _mock_registry_with_tools(["run_design"])

    result = run_interactive_agent(
        task="跑一轮设计",
        team_name="TestTeam",
        registry=registry,
    )

    assert result["ok"] is True
    assert result["task_type"] == "design_run"
    assert len(result["steps_executed"]) == 1
    assert result["steps_executed"][0] == "run_design"


def test_run_agent_inspect_results(tmp_path, monkeypatch):
    """Agent identifies inspect_results and reads latest."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    (tmp_path / "outputs" / "agent_sessions").mkdir(parents=True, exist_ok=True)

    registry = _mock_registry_with_tools(["inspect_outputs_latest", "read_selected_top6", "read_ranked_variants"])

    result = run_interactive_agent(
        task="读取最新结果并解释",
        team_name="TestTeam",
        registry=registry,
    )

    assert result["ok"] is True
    assert result["task_type"] == "inspect_results"
    assert "inspect_outputs_latest" in result["steps_executed"]
    assert "read_selected_top6" in result["steps_executed"]


def test_run_agent_export_only(tmp_path, monkeypatch):
    """Agent identifies export_only task."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    (tmp_path / "outputs" / "agent_sessions").mkdir(parents=True, exist_ok=True)

    registry = _mock_registry_with_tools(["read_selected_top6", "export_submission"])

    result = run_interactive_agent(
        task="导出提交文件",
        team_name="TestTeam",
        registry=registry,
    )

    assert result["ok"] is True
    assert result["task_type"] == "export_only"


def test_run_agent_resume_failed_run(tmp_path, monkeypatch):
    """Agent identifies resume task."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    (tmp_path / "outputs" / "agent_sessions").mkdir(parents=True, exist_ok=True)

    registry = _mock_registry_with_tools(["read_run_state", "resume_run"])

    result = run_interactive_agent(
        task="继续上一次失败的运行",
        team_name="TestTeam",
        registry=registry,
    )

    assert result["ok"] is True
    assert result["task_type"] == "resume_failed_run"
    assert "read_run_state" in result["steps_executed"]
    assert "resume_run" in result["steps_executed"]


def test_run_agent_train_then_design(tmp_path, monkeypatch):
    """Agent identifies train_then_design task."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    (tmp_path / "outputs" / "agent_sessions").mkdir(parents=True, exist_ok=True)

    registry = _mock_registry_with_tools(["train_brightness", "run_design"])

    result = run_interactive_agent(
        task="先训练亮度模型再设计",
        team_name="TestTeam",
        registry=registry,
    )

    assert result["ok"] is True
    assert result["task_type"] == "train_then_design"
    assert "train_brightness" in result["steps_executed"]
    assert "run_design" in result["steps_executed"]


# ── failure handling ───────────────────────────────────────────────────────


def test_run_agent_tool_failure_causes_recovery(tmp_path, monkeypatch):
    """When a tool fails with abort, agent stops and reports failure."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    (tmp_path / "outputs" / "agent_sessions").mkdir(parents=True, exist_ok=True)

    registry = ToolRegistry()

    # health_check succeeds
    def mock_health(**kw):
        return tool_result(True, "all good")

    # run_design fails
    def mock_design(**kw):
        return tool_result(False, "stage2_stability failure: ESMFold OOM", warnings=["oom"])

    registry.register(Tool(name="health_check", description="Health check", input_schema={}, callable=mock_health, side_effects=[], resume_safe=True))
    registry.register(Tool(name="run_design", description="Run design", input_schema={}, callable=mock_design, side_effects=["writes_files"], resume_safe=True))

    result = run_interactive_agent(
        task="先检查再跑设计",
        team_name="TestTeam",
        registry=registry,
    )

    # The recovery should try retry/degrade — since it's stage2_stability
    # It will try recovery steps up to max_recovery_loops
    assert result["session_id"] is not None
    # The agent should not succeed since the mock keeps failing
    assert result["ok"] is False or len(result["tool_history_summary"]) > 2


def test_run_agent_health_failure_aborts(tmp_path, monkeypatch):
    """When health_check fails with critical error, agent aborts."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    (tmp_path / "outputs" / "agent_sessions").mkdir(parents=True, exist_ok=True)

    registry = ToolRegistry()

    def mock_health(**kw):
        return tool_result(False, "关键模块失败", warnings=["ESMC failed"])

    def mock_design(**kw):
        return tool_result(True, "design ok")

    registry.register(Tool(name="health_check", description="HC", input_schema={}, callable=mock_health, side_effects=[], resume_safe=True))
    registry.register(Tool(name="run_design", description="RD", input_schema={}, callable=mock_design, side_effects=["writes_files"], resume_safe=True))

    result = run_interactive_agent(
        task="先检查再跑设计",
        team_name="TestTeam",
        registry=registry,
    )

    # Health check failure should abort before reaching run_design
    assert result["ok"] is False
    assert result["failure_reason"] != ""


# ── CLI help ───────────────────────────────────────────────────────────────


def test_cli_help_output():
    """CLI --help shows all options."""
    import subprocess
    import os

    # This test verifies the CLI can be invoked and shows help
    # We run it as a module
    project_dir = os.path.dirname(os.path.dirname(__file__))
    src_dir = os.path.join(project_dir, "src")

    result = subprocess.run(
        [sys.executable, "-m", "spebt_agent.cli.run_agent", "--help"],
        capture_output=True,
        text=True,
        cwd=project_dir,
        env={**os.environ, "PYTHONPATH": src_dir},
    )
    assert result.returncode == 0
    assert "--task" in result.stdout
    assert "--team-name" in result.stdout
    assert "--profile" in result.stdout
    assert "--json" in result.stdout
    assert "--task-file" in result.stdout


def test_cli_requires_task():
    """CLI errors when no task is provided."""
    import subprocess
    import os

    project_dir = os.path.dirname(os.path.dirname(__file__))
    src_dir = os.path.join(project_dir, "src")

    result = subprocess.run(
        [sys.executable, "-m", "spebt_agent.cli.run_agent"],
        capture_output=True,
        text=True,
        cwd=project_dir,
        env={**os.environ, "PYTHONPATH": src_dir},
    )
    assert result.returncode != 0
