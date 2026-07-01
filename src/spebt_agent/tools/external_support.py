from __future__ import annotations

from pathlib import Path

from spebt_agent.assets.envs import env_dir, env_python
from spebt_agent.assets.registry import list_tool_environments
from spebt_agent.paths import project_root, resolve_path


def resolve_tool_python(module: str) -> Path | None:
    for env in list_tool_environments():
        if env.module == module:
            py = env_python(env_dir(env))
            return py if py.exists() else None
    return None


def resolve_structure_path(parent_pdb: str | None, search_dirs: list[str] | None = None) -> Path | None:
    if not parent_pdb:
        return None

    candidates = []
    pdb_name = str(parent_pdb)
    if pdb_name.lower().endswith(".pdb"):
        filenames = [pdb_name]
    else:
        filenames = [f"{pdb_name}.pdb", pdb_name]

    for search_dir in search_dirs or []:
        base = resolve_path(search_dir)
        for filename in filenames:
            candidates.append(base / filename)

    for filename in filenames:
        candidates.append(project_root() / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def diff_mutations(parent_sequence: str, sequence: str) -> list[str]:
    if len(parent_sequence) != len(sequence):
        return []
    mutations = []
    for idx, (wt, mut) in enumerate(zip(parent_sequence, sequence), start=1):
        if wt != mut:
            mutations.append(f"{wt}{idx}{mut}")
    return mutations
