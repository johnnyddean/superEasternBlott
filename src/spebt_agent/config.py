import copy
from pathlib import Path
from typing import Any

import yaml

from spebt_agent.paths import project_root


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_default_configs() -> dict[str, dict[str, Any]]:
    return {
        "competition": load_yaml("configs/competition.yaml"),
        "model_paths": load_yaml("configs/model_paths.yaml"),
        "generation": load_yaml("configs/generation.yaml"),
        "scoring": load_yaml("configs/scoring_weights.yaml"),
    }


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def build_runtime_configs(
    *,
    profile: str = "stable",
    max_stage2_candidates: int | None = None,
    esmfold_top_k: int | None = None,
) -> dict[str, dict[str, Any]]:
    configs = copy.deepcopy(load_default_configs())
    generation_profiles = configs["generation"].get("design_profiles", {})
    model_profiles = configs["model_paths"].get("design_profiles", {})
    if profile in generation_profiles:
        deep_update(configs["generation"], generation_profiles[profile])
    if profile in model_profiles:
        deep_update(configs["model_paths"], model_profiles[profile])
    configs["generation"]["active_profile"] = profile
    configs["model_paths"]["active_profile"] = profile
    if max_stage2_candidates is not None:
        configs["generation"].setdefault("stage2", {})["max_total_candidates"] = int(max_stage2_candidates)
    if esmfold_top_k is not None:
        configs["model_paths"]["stability"]["esmfold_top_k"] = int(esmfold_top_k)
    return configs
