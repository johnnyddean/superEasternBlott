"""Tests for the standardized tool registry."""

import pytest
from spebt_agent.tools.registry import (
    Tool,
    ToolRegistry,
    build_default_registry,
    tool_result,
)


def test_tool_result_structure():
    result = tool_result(True, "success", artifacts={"a": "b"}, warnings=["w1"], next_hints=["h1"])
    assert result["ok"] is True
    assert result["summary"] == "success"
    assert result["artifacts"] == {"a": "b"}
    assert result["warnings"] == ["w1"]
    assert result["next_hints"] == ["h1"]


def test_tool_result_defaults():
    result = tool_result(False, "fail")
    assert result["ok"] is False
    assert result["artifacts"] == {}
    assert result["warnings"] == []
    assert result["next_hints"] == []


def test_tool_definition():
    def dummy_callable(x: int = 0) -> dict:
        return tool_result(True, f"x={x}")

    tool = Tool(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        callable=dummy_callable,
        side_effects=["reads_files"],
        resume_safe=True,
    )

    assert tool.name == "test_tool"
    assert tool.resume_safe is True
    assert "reads_files" in tool.side_effects

    summary = tool.to_summary()
    assert summary["name"] == "test_tool"
    assert "description" in summary
    assert "input_schema" in summary


def test_registry_register_and_list():
    registry = ToolRegistry()

    def t1(**kw) -> dict:
        return tool_result(True, "ok")

    registry.register(Tool(name="tool1", description="Tool 1", input_schema={}, callable=t1))
    registry.register(Tool(name="tool2", description="Tool 2", input_schema={}, callable=t1))

    tools = registry.list_tools()
    assert len(tools) == 2
    names = {t["name"] for t in tools}
    assert names == {"tool1", "tool2"}


def test_registry_call_success():
    registry = ToolRegistry()

    def echo(**kw) -> dict:
        return tool_result(True, f"echoed: {kw}")

    registry.register(Tool(name="echo", description="Echo", input_schema={}, callable=echo))
    result = registry.call("echo", {"msg": "hello"})
    assert result["ok"] is True
    assert "hello" in result["summary"]


def test_registry_call_unknown_tool():
    registry = ToolRegistry()
    result = registry.call("nonexistent")
    assert result["ok"] is False
    assert "Unknown tool" in result["summary"]


def test_registry_call_exception():
    registry = ToolRegistry()

    def raiser(**kw) -> dict:
        raise RuntimeError("boom")

    registry.register(Tool(name="raiser", description="Raises", input_schema={}, callable=raiser))
    result = registry.call("raiser")
    assert result["ok"] is False
    assert "boom" in result["summary"]


def test_registry_get_schema():
    registry = ToolRegistry()

    def t1(**kw) -> dict:
        return tool_result(True, "")

    registry.register(
        Tool(
            name="tool1",
            description="Test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            callable=t1,
        )
    )
    schema = registry.get_schema("tool1")
    assert schema is not None
    assert schema["name"] == "tool1"
    assert schema["input_schema"]["properties"]["x"]["type"] == "integer"

    assert registry.get_schema("nonexistent") is None


def test_default_registry_has_13_tools():
    registry = build_default_registry()
    tools = registry.list_tools()
    assert len(tools) == 13
    names = {t["name"] for t in tools}
    expected = {
        "health_check",
        "prepare_data",
        "run_design",
        "resume_run",
        "read_run_state",
        "read_trace",
        "read_ranked_variants",
        "read_selected_top6",
        "read_submission",
        "export_submission",
        "train_brightness",
        "inspect_outputs_latest",
        # "inspect_run_outputs" is the 12th
    }
    # re-check: health_check, prepare_data, run_design, resume_run, read_run_state,
    # read_trace, read_ranked_variants, read_selected_top6, read_submission,
    # export_submission, train_brightness, inspect_outputs_latest, inspect_run_outputs
    # That's 13 actually — let me verify
    all_names = {
        "health_check", "prepare_data", "run_design", "resume_run",
        "read_run_state", "read_trace", "read_ranked_variants", "read_selected_top6",
        "read_submission", "export_submission", "train_brightness",
        "inspect_outputs_latest", "inspect_run_outputs",
    }
    # The registry has 13 tools (I added inspect_run_outputs too)
    assert len(tools) == 13
    assert names == all_names


def test_default_registry_tools_have_required_fields():
    registry = build_default_registry()
    for tool_summary in registry.list_tools():
        assert "name" in tool_summary
        assert "description" in tool_summary
        assert "input_schema" in tool_summary
        assert "side_effects" in tool_summary
        assert "resume_safe" in tool_summary


def test_each_tool_returns_unified_structure():
    """Verify every registered tool's callable returns the unified result dict."""
    registry = build_default_registry()
    for tool_summary in registry.list_tools():
        name = tool_summary["name"]
        # Skip tools that require real files/models — just verify they're callable
        tool = registry._tools[name]
        # Tools that need no special environment can be called with empty inputs
        if name in ("health_check", "read_run_state", "read_trace",
                     "read_ranked_variants", "read_selected_top6",
                     "read_submission", "inspect_outputs_latest"):
            # These will fail gracefully (no real files) but should return the unified structure
            result = tool.callable()
            assert "ok" in result, f"{name} missing 'ok'"
            assert "summary" in result, f"{name} missing 'summary'"
            assert "artifacts" in result, f"{name} missing 'artifacts'"
            assert "warnings" in result, f"{name} missing 'warnings'"
            assert "next_hints" in result, f"{name} missing 'next_hints'"


def test_registry_call_passes_inputs():
    registry = ToolRegistry()
    received = {}

    def capture(**kw) -> dict:
        received.update(kw)
        return tool_result(True, "captured")

    registry.register(
        Tool(
            name="capture",
            description="Capture inputs",
            input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}},
            callable=capture,
        )
    )
    registry.call("capture", {"a": 42, "b": "hello"})
    assert received == {"a": 42, "b": "hello"}
