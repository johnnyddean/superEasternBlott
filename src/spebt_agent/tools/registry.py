"""Standardized tool registry for the interactive agent layer.

Every tool has a uniform definition (name, description, input/output schema,
callable) and returns a uniform result dict so the LLM planner and executor can
discover and invoke tools without knowing internal details.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from spebt_agent.assets.envs import env_dir, env_python
from spebt_agent.assets.registry import list_tool_environments
from spebt_agent.config import load_default_configs
from spebt_agent.paths import project_root

# ── unified result ──────────────────────────────────────────────────────────


def tool_result(
    ok: bool,
    summary: str = "",
    *,
    artifacts: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    next_hints: list[str] | None = None,
) -> dict:
    return {
        "ok": ok,
        "summary": summary,
        "artifacts": artifacts or {},
        "warnings": warnings or [],
        "next_hints": next_hints or [],
    }


# ── tool definition ─────────────────────────────────────────────────────────


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    callable: Callable[..., dict]
    side_effects: list[str] = field(default_factory=list)
    resume_safe: bool = True

    def to_summary(self) -> dict:
        """Return a lightweight dict suitable for inclusion in an LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "side_effects": self.side_effects,
            "resume_safe": self.resume_safe,
        }


# ── registry ────────────────────────────────────────────────────────────────


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict]:
        return [t.to_summary() for t in self._tools.values()]

    def get_schema(self, name: str) -> dict | None:
        tool = self._tools.get(name)
        return tool.to_summary() if tool else None

    def call(self, name: str, inputs: dict | None = None) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return tool_result(False, f"Unknown tool: {name}", warnings=[f"Tool '{name}' is not registered."])
        try:
            return tool.callable(**(inputs or {}))
        except Exception as exc:
            return tool_result(False, f"Tool '{name}' raised an exception: {exc}", warnings=[str(exc)])


# ── tool implementations ────────────────────────────────────────────────────


def _tool_health_check(modules: list[str] | None = None, strict: bool = False) -> dict:
    """Run smoke tests for every configured tool module."""
    from spebt_agent.cli.health_check import run_module_smoke

    if modules is None:
        modules = [env.module for env in list_tool_environments()] or ["brightness", "esmc", "inverse_folding", "stability"]

    results = [run_module_smoke(m) for m in modules]
    all_ok = all(r["ok"] for r in results)
    failed = [r["module"] for r in results if not r["ok"]]
    summary_lines = [f"health_check: {'PASS' if all_ok else 'FAIL'} ({len(results)} modules)"]
    for r in results:
        summary_lines.append(f"  {r['module']}: {r['status']}")
    warnings = [] if all_ok else [f"Failed modules: {', '.join(failed)}"]
    hints = []
    if not all_ok:
        hints.append("Review failed modules above. Optional tool failures may not block design runs.")
        hints.append("Run 'python -m spebt_agent.cli.setup_tool_envs' to recreate environments.")
        if strict:
            hints.append("Strict mode enabled — aborting due to module failure.")
    return tool_result(all_ok, "\n".join(summary_lines), warnings=warnings, next_hints=hints)


def _tool_prepare_data() -> dict:
    """Re-run data preparation from raw competition files."""
    from spebt_agent.data.prepare_competition_data import prepare_data

    root = project_root()
    configs = load_default_configs()
    raw = configs["competition"]["raw_data"]
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary = prepare_data(
        str(root / raw["refs"]),
        str(root / raw["gfp_xlsx"]),
        str(root / raw["exclusion_csv"]),
        processed_dir,
    )
    summary_path = processed_dir / "summary.json"
    return tool_result(
        True,
        f"prepare_data: completed. {summary.get('total_variants', '?')} variants prepared.",
        artifacts={"processed_dir": str(processed_dir), "summary_json": str(summary_path)},
    )


def _tool_run_design(
    team_name: str = "",
    profile: str = "stable",
    resume: bool = True,
    run_id: str | None = None,
    max_stage2_candidates: int | None = None,
    esmfold_top_k: int | None = None,
) -> dict:
    """Run the full spEBT design pipeline (stable or full_search)."""
    from spebt_agent.graph import run_design

    result = run_design(
        team_name=team_name or None,
        profile=profile,
        resume=resume,
        run_id=run_id,
        max_stage2_candidates=max_stage2_candidates,
        esmfold_top_k=esmfold_top_k,
    )
    outputs = result.get("outputs", {})
    ok = result.get("status") == "success" or bool(outputs.get("submission"))
    summary = (
        f"run_design: {'SUCCESS' if ok else 'FAILED'}\n"
        f"  run_id: {result.get('run_id', '?')}\n"
        f"  profile: {profile}\n"
        f"  run_dir: {result.get('run_dir', '?')}\n"
        f"  selected: {outputs.get('selected_top6', 'N/A')}\n"
        f"  submission: {outputs.get('submission', 'N/A')}"
    )
    return tool_result(
        ok,
        summary,
        artifacts={
            "run_id": result.get("run_id", ""),
            "run_dir": result.get("run_dir", ""),
            "latest_dir": result.get("latest_dir", ""),
            **outputs,
        },
        warnings=result.get("warnings", []),
        next_hints=(
            []
            if ok
            else [
                "Check run_state.json for the failed stage.",
                "Use resume_run with the same run_id to retry from checkpoint.",
                "Consider reducing max_stage2_candidates or esmfold_top_k.",
            ]
        ),
    )


def _tool_resume_run(run_id: str, team_name: str = "", profile: str = "stable") -> dict:
    """Resume a previously failed design run from its checkpoint."""
    return _tool_run_design(team_name=team_name, profile=profile, resume=True, run_id=run_id)


def _tool_read_run_state(run_dir: str | None = None, run_id: str | None = None) -> dict:
    """Read run_state.json and return a structured summary."""
    root = project_root()
    if run_dir:
        state_path = Path(run_dir) / "run_state.json"
    elif run_id:
        state_path = root / "outputs" / "runs" / run_id / "run_state.json"
    else:
        # try latest
        state_path = root / "outputs" / "latest" / "run_state.json"

    if not state_path.exists():
        return tool_result(False, f"run_state.json not found at {state_path}", next_hints=["Provide a specific run_id or run_dir."])

    state = json.loads(state_path.read_text(encoding="utf-8"))
    stage_summary = []
    for name, meta in state.get("stage_status", {}).items():
        stage_summary.append(f"  {name}: {meta.get('status', '?')}")

    failed_stages = [n for n, m in state.get("stage_status", {}).items() if m.get("status") == "failed"]
    running_stages = [n for n, m in state.get("stage_status", {}).items() if m.get("status") == "running"]

    summary = (
        f"Run {state.get('run_id', '?')}\n"
        f"  status: {state.get('status', '?')}\n"
        f"  profile: {state.get('profile', '?')}\n"
        + "\n".join(stage_summary)
    )
    warnings = state.get("warnings", [])
    hints = []
    if failed_stages:
        hints.append(f"Failed stages: {', '.join(failed_stages)}. Use resume_run to continue.")
    if running_stages:
        hints.append(f"Running stages: {', '.join(running_stages)}. The run may still be in progress.")
    if state.get("failure_reason"):
        hints.append(f"Failure reason: {state['failure_reason']}")

    return tool_result(
        True,
        summary,
        artifacts={"run_state_json": str(state_path)},
        warnings=[f"{w['source']}: {w['message']}" for w in warnings],
        next_hints=hints,
    )


def _tool_read_trace(run_dir: str | None = None, run_id: str | None = None, tail: int = 50) -> dict:
    """Read the last N events from agent_trace.jsonl."""
    root = project_root()
    if run_dir:
        trace_path = Path(run_dir) / "agent_trace.jsonl"
    elif run_id:
        trace_path = root / "outputs" / "runs" / run_id / "agent_trace.jsonl"
    else:
        trace_path = root / "outputs" / "latest" / "agent_trace.jsonl"

    if not trace_path.exists():
        return tool_result(False, f"agent_trace.jsonl not found at {trace_path}", next_hints=["Provide a specific run_id or run_dir."])

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-tail:] if len(lines) > tail else lines
    events = [json.loads(line) for line in recent]

    summary_lines = [f"Trace ({len(events)} events shown, {len(lines)} total):"]
    for e in events:
        summary_lines.append(f"  [{e.get('node','?')}] {e.get('status','?')}")
    failed = [e for e in events if e.get("status") == "failed"]
    hints = []
    if failed:
        hints.append(f"{len(failed)} failure events in trace. Check run_state for recovery options.")

    return tool_result(True, "\n".join(summary_lines), next_hints=hints)


def _tool_read_ranked_variants(source: str = "latest", top_n: int = 10, run_id: str | None = None) -> dict:
    """Read ranked_variants.csv and return top-N summary."""
    root = project_root()
    if run_id:
        csv_path = root / "outputs" / "runs" / run_id / "ranked_variants.csv"
    elif source == "latest":
        csv_path = root / "outputs" / "latest" / "ranked_variants.csv"
    else:
        csv_path = Path(source)

    if not csv_path.exists():
        return tool_result(False, f"ranked_variants.csv not found at {csv_path}", next_hints=["Run design first or provide a specific run_id."])

    df = pd.read_csv(csv_path)
    top = df.head(top_n)
    lines = [f"Top {len(top)} of {len(df)} ranked variants:"]
    for _, row in top.iterrows():
        vid = row.get("variant_id", "?")
        score = row.get("final_score", row.get("combined_score", "?"))
        brightness = row.get("predicted_relative_brightness", "?")
        retention = row.get("predicted_retention72", "?")
        lines.append(f"  {vid}: score={score}, brightness={brightness}, retention72={retention}")

    return tool_result(True, "\n".join(lines), artifacts={"ranked_variants_csv": str(csv_path), "total_count": str(len(df))})


def _tool_read_selected_top6(source: str = "latest", run_id: str | None = None) -> dict:
    """Read selected_top6.csv and return the final 6 chosen sequences."""
    root = project_root()
    if run_id:
        csv_path = root / "outputs" / "runs" / run_id / "selected_top6.csv"
    elif source == "latest":
        csv_path = root / "outputs" / "latest" / "selected_top6.csv"
    else:
        csv_path = Path(source)

    if not csv_path.exists():
        return tool_result(False, f"selected_top6.csv not found at {csv_path}", next_hints=["Run design first or provide a specific run_id."])

    df = pd.read_csv(csv_path)
    lines = [f"Selected top {len(df)} sequences:"]
    for _, row in df.iterrows():
        vid = row.get("variant_id", "?")
        seq = row.get("sequence", "")
        score = row.get("final_score", row.get("combined_score", "?"))
        mutations = row.get("mutations", "")
        lines.append(f"  {vid}: {seq[:30]}... score={score}, mutations={mutations}")

    return tool_result(True, "\n".join(lines), artifacts={"selected_top6_csv": str(csv_path), "count": str(len(df))})


def _tool_read_submission(source: str = "latest", run_id: str | None = None) -> dict:
    """Read submission.csv and return its contents."""
    root = project_root()
    if run_id:
        csv_path = root / "outputs" / "runs" / run_id / "submission.csv"
    elif source == "latest":
        csv_path = root / "outputs" / "latest" / "submission.csv"
    else:
        csv_path = Path(source)

    if not csv_path.exists():
        return tool_result(False, f"submission.csv not found at {csv_path}", next_hints=["Run design and export first, or provide a specific run_id."])

    df = pd.read_csv(csv_path)
    lines = [f"Submission ({len(df)} sequences):"]
    for _, row in df.iterrows():
        lines.append(f"  {row.get('Team_Name', '?')} Seq_{row.get('Seq_ID', '?')}: {row.get('Sequence', '')[:30]}...")

    return tool_result(True, "\n".join(lines), artifacts={"submission_csv": str(csv_path)})


def _tool_export_submission(selected_csv: str, team_name: str, out_csv: str = "") -> dict:
    """Export submission.csv from a selected_top6 CSV file."""
    from spebt_agent.tools.submission import export_submission_csv

    root = project_root()
    selected_path = Path(selected_csv)
    if not selected_path.is_absolute():
        selected_path = root / selected_path
    if not selected_path.exists():
        return tool_result(False, f"Selected CSV not found: {selected_path}")

    records = pd.read_csv(selected_path).to_dict("records")
    out = Path(out_csv) if out_csv else (root / "outputs" / "submission.csv")
    result_path = export_submission_csv(records, team_name, out)
    return tool_result(
        True,
        f"export_submission: written to {result_path} ({len(records)} sequences)",
        artifacts={"submission_csv": str(result_path)},
    )


def _tool_train_brightness(
    train_csv: str = "data/processed/brightness_train.csv",
    valid_csv: str = "data/processed/brightness_valid.csv",
    target: str = "brightness",
) -> dict:
    """Train the brightness predictor model."""
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    from spebt_agent.tools.brightness import build_feature_matrix
    from spebt_agent.tools.esmc import embed_sequences_real

    root = project_root()
    configs = load_default_configs()
    esmc_cfg = configs["model_paths"]["esmc"]
    brightness_cfg = configs["model_paths"]["brightness"]
    default_out = brightness_cfg["abs_model"] if target == "brightness" else brightness_cfg["delta_model"]
    out = root / default_out
    metrics_out = root / "artifacts" / "reports" / "brightness_training_metrics.json"

    train = pd.read_csv(root / train_csv)
    valid = pd.read_csv(root / valid_csv)

    seqs = pd.concat([train["sequence"], valid["sequence"]], ignore_index=True).astype(str).drop_duplicates().tolist()
    embeddings_map = embed_sequences_real(
        sequences=seqs,
        model_dir=str(root / esmc_cfg["model_dir"]),
        model_name=esmc_cfg["model_name"],
        cache_dir=str(root / esmc_cfg["embedding_cache_dir"]),
        pooling=esmc_cfg.get("pooling", "mean"),
        batch_size=int(esmc_cfg.get("batch_size", 8)),
        device=esmc_cfg.get("device", "auto"),
        dtype=esmc_cfg.get("dtype", "float32"),
    )
    seq_to_emb = {s: e for s, e in zip(seqs, embeddings_map)}

    parent_categories = sorted(train["parent"].astype(str).unique().tolist())

    def _features(df):
        recs = [{"sequence": str(r["sequence"]), "num_mutations": int(r["num_mutations"]), "parent": str(r["parent"])} for _, r in df.iterrows()]
        embs = np.vstack([seq_to_emb[str(r["sequence"])] for _, r in df.iterrows()]).astype(np.float32)
        return build_feature_matrix(recs, embs, parent_categories)

    x_train = _features(train)
    x_valid = _features(valid)
    y_train = train[target].astype(float).to_numpy()
    y_valid = valid[target].astype(float).to_numpy()

    estimator = HistGradientBoostingRegressor(max_iter=200, max_depth=8, learning_rate=0.05, l2_regularization=0.01, random_state=42)
    estimator.fit(x_train, y_train)
    pred = estimator.predict(x_valid)

    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "estimator": estimator,
        "target": target,
        "parent_categories": parent_categories,
        "feature_dim": int(x_train.shape[1]),
        "esmc": {
            "model_name": esmc_cfg["model_name"],
            "model_dir": str(root / esmc_cfg["model_dir"]),
            "embedding_cache_dir": str(root / esmc_cfg["embedding_cache_dir"]),
            "pooling": esmc_cfg.get("pooling", "mean"),
            "batch_size": int(esmc_cfg.get("batch_size", 8)),
            "device": esmc_cfg.get("device", "auto"),
            "dtype": esmc_cfg.get("dtype", "float32"),
            "embedding_dim": int(esmc_cfg.get("embedding_dim", x_train.shape[1] - 2 - len(parent_categories))),
        },
    }
    joblib.dump(bundle, out)

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if metrics_out.exists():
        metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    metrics[target] = {
        "out": str(out),
        "r2": float(r2_score(y_valid, pred)),
        "mae": float(mean_absolute_error(y_valid, pred)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "feature_dim": int(x_train.shape[1]),
    }
    metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return tool_result(
        True,
        f"train_brightness ({target}): R²={metrics[target]['r2']:.4f}, MAE={metrics[target]['mae']:.4f}",
        artifacts={"model": str(out), "metrics": str(metrics_out)},
    )


def _tool_inspect_outputs_latest() -> dict:
    """List all files under outputs/latest/."""
    root = project_root()
    latest = root / "outputs" / "latest"
    if not latest.exists():
        return tool_result(False, "outputs/latest/ does not exist.", next_hints=["Run design first to generate outputs."])

    files = sorted(latest.rglob("*"))
    file_list = [str(f.relative_to(root)) for f in files if f.is_file()]
    if not file_list:
        return tool_result(True, "outputs/latest/ exists but is empty.", next_hints=["Run design first."])

    summary = f"outputs/latest/ contains {len(file_list)} files:\n" + "\n".join(f"  {f}" for f in file_list)
    return tool_result(True, summary, artifacts={f: str(latest / f) for f in file_list})


def _tool_inspect_run_outputs(run_id: str) -> dict:
    """List all files under outputs/runs/<run_id>/."""
    root = project_root()
    run_dir = root / "outputs" / "runs" / run_id
    if not run_dir.exists():
        return tool_result(False, f"Run directory not found: {run_dir}", next_hints=["Check the run_id is correct."])

    files = sorted(run_dir.rglob("*"))
    file_list = [str(f.relative_to(root)) for f in files if f.is_file()]
    summary = f"outputs/runs/{run_id}/ contains {len(file_list)} files:\n" + "\n".join(f"  {f}" for f in file_list)
    return tool_result(True, summary, artifacts={f: str(run_dir / f) for f in file_list})


# ── build the default registry ──────────────────────────────────────────────


def build_default_registry() -> ToolRegistry:
    """Create a ToolRegistry populated with all 12 standard tools."""
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="health_check",
            description="检查所有工具模块环境是否就绪。运行各模块冒烟测试，返回通过/失败状态。在跑设计之前应该先做健康检查。",
            input_schema={
                "type": "object",
                "properties": {
                    "modules": {"type": "array", "items": {"type": "string"}, "description": "要检查的模块列表，默认全部"},
                    "strict": {"type": "boolean", "description": "失败时是否视为致命错误"},
                },
            },
            callable=_tool_health_check,
            side_effects=["reads_files", "subprocess"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="prepare_data",
            description="从原始竞赛文件（FASTA、Excel、排除列表）重建训练数据。一般在初次使用或数据更新时调用。",
            input_schema={"type": "object", "properties": {}},
            callable=_tool_prepare_data,
            side_effects=["writes_files"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="run_design",
            description="运行完整的 spEBT 蛋白设计流水线：生成突变、过滤、亮度/稳定性评分、排名、多样性选择前6、导出提交文件。支持 stable（稳定模式）和 full_search（完整搜索）两种配置。支持从失败点 resume。",
            input_schema={
                "type": "object",
                "properties": {
                    "team_name": {"type": "string", "description": "队伍名称"},
                    "profile": {"type": "string", "enum": ["stable", "full_search"], "description": "运行配置：stable 适合快速迭代，full_search 搜索更广"},
                    "resume": {"type": "boolean", "description": "是否从检查点恢复"},
                    "run_id": {"type": "string", "description": "指定 run_id（恢复时必填）"},
                    "max_stage2_candidates": {"type": "integer", "description": "阶段2最大候选数（失败降级时使用）"},
                    "esmfold_top_k": {"type": "integer", "description": "ESMFold 评分的 top-K 数量（失败降级时使用）"},
                },
                "required": ["team_name"],
            },
            callable=_tool_run_design,
            side_effects=["writes_files", "reads_files", "network", "subprocess", "llm_call"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="resume_run",
            description="从检查点恢复之前失败的 run_design。需要提供失败的 run_id。",
            input_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "要恢复的 run_id"},
                    "team_name": {"type": "string", "description": "队伍名称"},
                    "profile": {"type": "string", "enum": ["stable", "full_search"], "description": "运行配置"},
                },
                "required": ["run_id"],
            },
            callable=_tool_resume_run,
            side_effects=["writes_files", "reads_files", "network", "subprocess", "llm_call"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="read_run_state",
            description="读取 run_state.json，返回运行状态摘要：各阶段完成情况、失败原因、警告信息。用于判断失败恢复策略。",
            input_schema={
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string", "description": "运行目录路径"},
                    "run_id": {"type": "string", "description": "run_id（自动定位 outputs/runs/<run_id>）"},
                },
            },
            callable=_tool_read_run_state,
            side_effects=["reads_files"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="read_trace",
            description="读取 agent_trace.jsonl 的最后 N 条事件，用于了解流水线执行细节。",
            input_schema={
                "type": "object",
                "properties": {
                    "run_dir": {"type": "string", "description": "运行目录路径"},
                    "run_id": {"type": "string", "description": "run_id"},
                    "tail": {"type": "integer", "description": "显示最后 N 条，默认 50"},
                },
            },
            callable=_tool_read_trace,
            side_effects=["reads_files"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="read_ranked_variants",
            description="读取 ranked_variants.csv 排名结果，展示前 N 条变体的得分、亮度、稳定性。",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "数据来源：'latest'（默认）或具体路径"},
                    "top_n": {"type": "integer", "description": "展示前 N 条，默认 10"},
                    "run_id": {"type": "string", "description": "指定 run_id"},
                },
            },
            callable=_tool_read_ranked_variants,
            side_effects=["reads_files"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="read_selected_top6",
            description="读取 selected_top6.csv，展示最终入选的 6 条序列及其得分和突变信息。",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "数据来源：'latest'（默认）或具体路径"},
                    "run_id": {"type": "string", "description": "指定 run_id"},
                },
            },
            callable=_tool_read_selected_top6,
            side_effects=["reads_files"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="read_submission",
            description="读取 submission.csv，展示最终提交的竞赛文件内容。",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "数据来源：'latest'（默认）或具体路径"},
                    "run_id": {"type": "string", "description": "指定 run_id"},
                },
            },
            callable=_tool_read_submission,
            side_effects=["reads_files"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="export_submission",
            description="从 selected_top6.csv 导出竞赛提交文件 submission.csv。",
            input_schema={
                "type": "object",
                "properties": {
                    "selected_csv": {"type": "string", "description": "selected_top6.csv 路径"},
                    "team_name": {"type": "string", "description": "队伍名称"},
                    "out_csv": {"type": "string", "description": "输出路径（可选）"},
                },
                "required": ["selected_csv", "team_name"],
            },
            callable=_tool_export_submission,
            side_effects=["writes_files"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="train_brightness",
            description="在训练数据上训练亮度预测模型（HistGradientBoostingRegressor）。训练完成后保存模型和评估指标。",
            input_schema={
                "type": "object",
                "properties": {
                    "train_csv": {"type": "string", "description": "训练集 CSV 路径"},
                    "valid_csv": {"type": "string", "description": "验证集 CSV 路径"},
                    "target": {"type": "string", "enum": ["brightness", "delta_brightness_vs_parent"], "description": "预测目标"},
                },
            },
            callable=_tool_train_brightness,
            side_effects=["writes_files", "reads_files", "network"],
            resume_safe=False,
        )
    )

    registry.register(
        Tool(
            name="inspect_outputs_latest",
            description="列出 outputs/latest/ 下的所有文件，了解最近一次成功运行的产出。",
            input_schema={"type": "object", "properties": {}},
            callable=_tool_inspect_outputs_latest,
            side_effects=["reads_files"],
            resume_safe=True,
        )
    )

    registry.register(
        Tool(
            name="inspect_run_outputs",
            description="列出 outputs/runs/<run_id>/ 下的所有文件，了解特定运行的产出。",
            input_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "要查看的 run_id"},
                },
                "required": ["run_id"],
            },
            callable=_tool_inspect_run_outputs,
            side_effects=["reads_files"],
            resume_safe=True,
        )
    )

    return registry
