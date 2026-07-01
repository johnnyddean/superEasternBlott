"""Agent session state — tracks an entire interactive agent conversation.

An agent session can span multiple run_design invocations (e.g. health_check →
run_design → failed → resume with reduced params → export). This module provides
the AgentSession class that persists to outputs/agent_sessions/<session_id>/agent_state.json.

Separate from run_state.json which tracks a single run_design pipeline execution.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spebt_agent import paths as _paths


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentSession:
    """Tracks a complete interactive agent task from planning through execution."""

    def __init__(
        self,
        task: str,
        team_name: str | None = None,
        *,
        session_id: str | None = None,
        llm_model: str = "",
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._root = _paths.project_root()
        self._session_dir = self._root / "outputs" / "agent_sessions" / self.session_id
        self._state_path = self._session_dir / "agent_state.json"

        if self._state_path.exists():
            self.state = json.loads(self._state_path.read_text(encoding="utf-8"))
        else:
            self.state: dict[str, Any] = {
                "session_id": self.session_id,
                "user_task": task,
                "team_name": team_name,
                "llm_model": llm_model,
                "plan": None,
                "completed_steps": [],
                "tool_history": [],
                "active_run_id": None,
                "related_run_ids": [],
                "status": "created",
                "final_summary": "",
                "artifacts": {},
                "warnings": [],
                "failure_reason": "",
                "started_at": _now_iso(),
                "finished_at": "",
            }
            self.save()

    # ── persistence ──────────────────────────────────────────────────────

    def save(self) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, session_id: str) -> AgentSession | None:
        root = _paths.project_root()
        state_path = root / "outputs" / "agent_sessions" / session_id / "agent_state.json"
        if not state_path.exists():
            return None
        state = json.loads(state_path.read_text(encoding="utf-8"))
        session = cls(task=state.get("user_task", ""), team_name=state.get("team_name"), session_id=session_id, llm_model=state.get("llm_model", ""))
        session.state = state
        return session

    # ── plan ─────────────────────────────────────────────────────────────

    def set_plan(self, plan: dict) -> None:
        self.state["plan"] = plan
        self.set_status("planning")

    # ── steps ────────────────────────────────────────────────────────────

    def add_step(self, step: dict) -> None:
        """Record a completed plan step."""
        step["finished_at"] = _now_iso()
        self.state["completed_steps"].append(step)
        self.save()

    # ── tool calls ───────────────────────────────────────────────────────

    def record_tool_call(self, name: str, inputs: dict | None, result: dict) -> None:
        entry = {
            "tool": name,
            "inputs": inputs or {},
            "ok": result.get("ok", False),
            "summary": result.get("summary", "")[:500],
            "artifacts": result.get("artifacts", {}),
            "warnings": result.get("warnings", []),
            "called_at": _now_iso(),
        }
        self.state["tool_history"].append(entry)

        # track run_ids
        rid = result.get("artifacts", {}).get("run_id")
        if rid and rid not in self.state["related_run_ids"]:
            self.state["related_run_ids"].append(rid)
            self.state["active_run_id"] = rid

        # accumulate warnings
        for w in result.get("warnings", []):
            if w not in self.state["warnings"]:
                self.state["warnings"].append(w)

        # accumulate artifacts
        for k, v in result.get("artifacts", {}).items():
            self.state["artifacts"][k] = v

        self.save()

    # ── status ───────────────────────────────────────────────────────────

    def set_status(self, status: str, failure_reason: str = "") -> None:
        self.state["status"] = status
        self.state["failure_reason"] = failure_reason
        if status in ("success", "failed"):
            self.state["finished_at"] = _now_iso()
        self.save()

    def finalize(self, summary: str, artifacts: dict[str, str] | None = None) -> None:
        self.state["final_summary"] = summary
        if artifacts:
            self.state["artifacts"].update(artifacts)
        if self.state.get("status") not in ("failed",):
            self.set_status("success")

    # ── properties ───────────────────────────────────────────────────────

    @property
    def is_failed(self) -> bool:
        return self.state.get("status") == "failed"

    @property
    def status(self) -> str:
        return self.state.get("status", "unknown")

    @property
    def active_run_id(self) -> str | None:
        return self.state.get("active_run_id")

    @property
    def summary_dict(self) -> dict:
        """Return a compact summary for inclusion in the final result."""
        return {
            "session_id": self.session_id,
            "status": self.state["status"],
            "goal": (self.state.get("plan") or {}).get("goal", self.state["user_task"]),
            "task_type": (self.state.get("plan") or {}).get("task_type", ""),
            "steps_executed": [s.get("tool", s.get("id", "?")) for s in self.state.get("completed_steps", [])],
            "tool_history_summary": [
                {"tool": h["tool"], "ok": h["ok"], "summary": h["summary"][:200]} for h in self.state.get("tool_history", [])
            ],
            "run_ids": self.state.get("related_run_ids", []),
            "warnings": self.state.get("warnings", []),
            "failure_reason": self.state.get("failure_reason", ""),
            "final_summary": self.state.get("final_summary", ""),
            "artifacts": self.state.get("artifacts", {}),
        }
