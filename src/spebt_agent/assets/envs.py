from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from spebt_agent.assets.registry import ToolEnvironment, list_tool_environments
from spebt_agent.paths import project_root


def env_dir(env: ToolEnvironment, env_root: str | Path = ".venvs") -> Path:
    return project_root() / env_root / env.env_name


def env_python(env_path: Path) -> Path:
    if os.name == "nt":
        return env_path / "Scripts" / "python.exe"
    return env_path / "bin" / "python"


def create_tool_environment(env: ToolEnvironment, env_root: str | Path = ".venvs", install_requirements: bool = False) -> Path:
    path = env_dir(env, env_root)
    if not env_python(path).exists():
        subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)
    if install_requirements and env.requirements:
        subprocess.run([str(env_python(path)), "-m", "pip", "install", *env.requirements], check=True)
    return path


def create_all_tool_environments(env_root: str | Path = ".venvs", install_requirements: bool = False) -> list[Path]:
    return [create_tool_environment(env, env_root, install_requirements) for env in list_tool_environments()]
