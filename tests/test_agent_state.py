"""Tests for AgentSession — agent session state management."""

import json
import pytest
from spebt_agent.agent_state import AgentSession


def test_session_creation(tmp_path, monkeypatch):
    """Session creates its state file on init."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)

    session = AgentSession(task="测试任务", team_name="TestTeam", llm_model="test-model")

    assert session.session_id is not None
    assert session.state["user_task"] == "测试任务"
    assert session.state["team_name"] == "TestTeam"
    assert session.state["status"] == "created"

    # State file should exist
    state_path = tmp_path / "outputs" / "agent_sessions" / session.session_id / "agent_state.json"
    assert state_path.exists()

    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    assert loaded["session_id"] == session.session_id


def test_session_save_and_load(tmp_path, monkeypatch):
    """Session can be saved and reloaded."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)

    session = AgentSession(task="保存测试", team_name="TeamA", llm_model="m1")
    session.set_plan({"goal": "测试目标", "task_type": "design_run", "steps": []})
    session.record_tool_call("health_check", {}, {"ok": True, "summary": "pass", "artifacts": {}, "warnings": []})
    session.set_status("success")

    # Reload
    loaded = AgentSession.load(session.session_id)
    assert loaded is not None
    assert loaded.state["user_task"] == "保存测试"
    assert loaded.state["plan"]["goal"] == "测试目标"
    assert loaded.state["status"] == "success"
    assert len(loaded.state["tool_history"]) == 1


def test_session_load_nonexistent(tmp_path, monkeypatch):
    """Loading a nonexistent session returns None."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)

    result = AgentSession.load("nonexistent-id")
    assert result is None


def test_record_tool_call(tmp_path, monkeypatch):
    """Tool calls are recorded with inputs and results."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)

    session = AgentSession(task="工具测试", team_name="T")
    session.record_tool_call(
        "run_design",
        {"profile": "stable", "team_name": "T"},
        {"ok": True, "summary": "设计完成", "artifacts": {"run_id": "abc123", "submission_csv": "/tmp/sub.csv"}, "warnings": ["可选工具缺失"]},
    )

    assert len(session.state["tool_history"]) == 1
    entry = session.state["tool_history"][0]
    assert entry["tool"] == "run_design"
    assert entry["ok"] is True
    assert entry["inputs"] == {"profile": "stable", "team_name": "T"}

    # run_id should be tracked
    assert "abc123" in session.state["related_run_ids"]
    assert session.state["active_run_id"] == "abc123"

    # warnings should be accumulated
    assert "可选工具缺失" in session.state["warnings"]

    # artifacts should be accumulated
    assert session.state["artifacts"]["run_id"] == "abc123"


def test_add_step(tmp_path, monkeypatch):
    """Plan steps are tracked."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)

    session = AgentSession(task="步骤测试", team_name="T")
    session.add_step({"id": "1", "tool": "health_check", "status": "success", "result_summary": "通过"})
    session.add_step({"id": "2", "tool": "run_design", "status": "success", "result_summary": "完成"})

    assert len(session.state["completed_steps"]) == 2
    assert session.state["completed_steps"][0]["tool"] == "health_check"
    assert session.state["completed_steps"][0]["status"] == "success"
    assert "finished_at" in session.state["completed_steps"][0]


def test_status_transitions(tmp_path, monkeypatch):
    """Status transitions work correctly."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)

    session = AgentSession(task="状态测试", team_name="T")

    session.set_status("planning")
    assert session.status == "planning"

    session.set_status("executing")
    assert session.status == "executing"

    session.set_status("success")
    assert session.status == "success"
    assert session.state["finished_at"] != ""

    session.set_status("failed", "测试失败原因")
    assert session.is_failed
    assert session.state["failure_reason"] == "测试失败原因"


def test_finalize(tmp_path, monkeypatch):
    """Finalize sets success and stores summary."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)

    session = AgentSession(task="最终化测试", team_name="T")
    session.finalize("任务全部完成 ✅", {"report": "/tmp/report.md"})
    assert session.status == "success"
    assert session.state["final_summary"] == "任务全部完成 ✅"
    assert session.state["artifacts"]["report"] == "/tmp/report.md"


def test_summary_dict(tmp_path, monkeypatch):
    """summary_dict returns a compact view."""
    from spebt_agent import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)

    session = AgentSession(task="摘要测试", team_name="T")
    session.set_plan({"goal": "测试目标", "task_type": "design_run", "steps": []})
    session.add_step({"id": "1", "tool": "health_check", "status": "success", "result_summary": "通过"})
    session.record_tool_call("health_check", {}, {"ok": True, "summary": "全部通过", "artifacts": {}, "warnings": []})
    session.finalize("完成")

    d = session.summary_dict
    assert d["session_id"] == session.session_id
    assert d["status"] == "success"
    assert d["goal"] == "测试目标"
    assert d["task_type"] == "design_run"
    assert "health_check" in d["steps_executed"]
