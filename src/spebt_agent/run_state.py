from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spebt_agent.memory.logger import append_trace


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint_config(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class RunTracker:
    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        latest_dir: Path,
        profile: str,
        team_name: str,
        config_fingerprint: str,
        allow_resume: bool,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.latest_dir = latest_dir
        self.profile = profile
        self.team_name = team_name
        self.config_fingerprint = config_fingerprint
        self.trace_path = self.run_dir / "agent_trace.jsonl"
        self.state_path = self.run_dir / "run_state.json"
        self._stage_start_times: dict[str, float] = {}
        self.resumed = False
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

        if allow_resume and self.state_path.exists():
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (
                loaded.get("config_fingerprint") == self.config_fingerprint
                and loaded.get("profile") == self.profile
                and loaded.get("team_name") == self.team_name
            ):
                self.state = loaded
                self.resumed = True
            else:
                self.state = self._fresh_state()
                self._reset_files()
        else:
            self.state = self._fresh_state()
            self._reset_files()
        self.save()

    def _fresh_state(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "profile": self.profile,
            "team_name": self.team_name,
            "config_fingerprint": self.config_fingerprint,
            "status": "running",
            "artifacts": {},
            "stage_status": {},
            "timings": {},
            "warnings": [],
            "failure_reason": "",
            "run_dir": str(self.run_dir),
            "latest_dir": str(self.latest_dir),
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "finished_at": "",
        }

    def _reset_files(self) -> None:
        if self.trace_path.exists():
            self.trace_path.unlink()
        if self.state_path.exists():
            self.state_path.unlink()

    def save(self) -> None:
        self.state["updated_at"] = now_iso()
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")

    def set_status(self, status: str, failure_reason: str = "") -> None:
        self.state["status"] = status
        self.state["failure_reason"] = failure_reason
        self.state["finished_at"] = now_iso() if status in {"success", "failed"} else ""
        self.save()

    def add_warning(self, source: str, message: str) -> None:
        self.state.setdefault("warnings", []).append({"source": source, "message": message, "timestamp": now_iso()})
        self.save()

    def trace(self, node: str, status: str, **extra: Any) -> None:
        append_trace(self.trace_path, {"run_id": self.run_id, "node": node, "status": status, **extra})

    def stage_start(self, stage: str, **details: Any) -> None:
        self._stage_start_times[stage] = time.perf_counter()
        self.state["stage_status"][stage] = {
            "status": "running",
            "started_at": now_iso(),
            "finished_at": "",
            "details": details,
        }
        self.save()
        self.trace(stage, "start", **details)

    def stage_success(self, stage: str, artifacts: dict[str, str] | None = None, **details: Any) -> None:
        started = self._stage_start_times.pop(stage, None)
        stage_entry = self.state["stage_status"].setdefault(stage, {})
        stage_entry.update(
            {
                "status": "success",
                "finished_at": now_iso(),
                "details": details,
            }
        )
        if "started_at" not in stage_entry:
            stage_entry["started_at"] = now_iso()
        if started is not None:
            self.state["timings"][stage] = round(time.perf_counter() - started, 3)
        if artifacts:
            self.state["artifacts"].update(artifacts)
            stage_entry["artifacts"] = artifacts
        self.save()
        self.trace(stage, "success", **details, **(artifacts or {}))

    def stage_failed(self, stage: str, error: str, **details: Any) -> None:
        started = self._stage_start_times.pop(stage, None)
        stage_entry = self.state["stage_status"].setdefault(stage, {})
        stage_entry.update(
            {
                "status": "failed",
                "finished_at": now_iso(),
                "details": {**details, "error": error},
            }
        )
        if started is not None:
            self.state["timings"][stage] = round(time.perf_counter() - started, 3)
        self.state["failure_reason"] = error
        self.save()
        self.trace(stage, "failed", error=error, **details)

    def fail_running_stage(self, error: str) -> str | None:
        running = [name for name, meta in self.state.get("stage_status", {}).items() if meta.get("status") == "running"]
        if not running:
            return None
        stage = running[-1]
        self.stage_failed(stage, error)
        return stage

    def stage_resume_hit(self, stage: str, **details: Any) -> None:
        self.trace(stage, "resume_hit", **details)

    def stage_can_resume(self, stage: str, required_artifacts: list[str] | None = None) -> bool:
        stage_entry = self.state.get("stage_status", {}).get(stage, {})
        if stage_entry.get("status") != "success":
            return False
        for key in required_artifacts or []:
            path = self.state.get("artifacts", {}).get(key)
            if not path or not Path(path).exists():
                return False
        return True

    def artifact_path(self, filename: str) -> Path:
        return self.run_dir / filename

    def register_artifact(self, key: str, path: str | Path) -> str:
        resolved = str(Path(path))
        self.state["artifacts"][key] = resolved
        self.save()
        return resolved

    def write_json(self, filename: str, payload: Any) -> Path:
        path = self.artifact_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def publish_latest(self, keys: list[str]) -> None:
        for key in keys:
            src = self.state.get("artifacts", {}).get(key)
            if not src:
                continue
            src_path = Path(src)
            if not src_path.exists():
                continue
            dst = self.latest_dir / src_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
