from __future__ import annotations

from typing import Any, TypedDict


class GFPDesignState(TypedDict, total=False):
    run_id: str
    profile: str
    team_name: str
    configs: dict[str, Any]
    parent_sequence: str
    parent_name: str
    processed_summary: dict[str, Any]
    variants: list[dict[str, Any]]
    stage1_ranked: list[dict[str, Any]]
    final_ranked: list[dict[str, Any]]
    selected_top6: list[dict[str, Any]]
    stage2_candidates: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    outputs: dict[str, str]
    llm_strategy: str
    report_text: str
