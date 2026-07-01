from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spebt_agent.config import load_yaml


@dataclass(frozen=True)
class ToolEnvironment:
    module: str
    env_name: str
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class ModelAsset:
    asset_id: str
    tool_module: str
    kind: str
    target: str
    size_bytes: int
    size_source: str
    description: str
    spec: dict[str, Any]

    @property
    def size_human(self) -> str:
        value = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TB"


def load_asset_config(path: str | Path = "configs/model_assets.yaml") -> dict[str, Any]:
    return load_yaml(path)


def list_tool_environments(config: dict[str, Any] | None = None) -> list[ToolEnvironment]:
    cfg = config or load_asset_config()
    modules = cfg.get("environments", {}).get("modules", {})
    return [
        ToolEnvironment(module=name, env_name=spec["env_name"], requirements=tuple(spec.get("requirements", [])))
        for name, spec in sorted(modules.items())
    ]


def list_model_assets(config: dict[str, Any] | None = None, tool_module: str | None = None) -> list[ModelAsset]:
    cfg = config or load_asset_config()
    assets = []
    for spec in cfg.get("assets", []):
        if tool_module and spec.get("tool_module") != tool_module:
            continue
        target = spec.get("target_dir") or spec.get("target_path") or ""
        assets.append(
            ModelAsset(
                asset_id=spec["asset_id"],
                tool_module=spec["tool_module"],
                kind=spec["kind"],
                target=target,
                size_bytes=int(spec.get("size_bytes", 0)),
                size_source=spec.get("size_source", "unknown"),
                description=spec.get("description", ""),
                spec=spec,
            )
        )
    return assets


def list_manual_assets(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or load_asset_config()
    return list(cfg.get("manual_assets", []))
