from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from spebt_agent.brain import build_llm_client
from spebt_agent.config import build_runtime_configs
from spebt_agent.data.prepare_competition_data import prepare_data
from spebt_agent.data.reference_loader import load_parent_sequence
from spebt_agent.memory.db import connect, record_agent_run
from spebt_agent.paths import project_root, resolve_path
from spebt_agent.run_state import RunTracker, fingerprint_config
from spebt_agent.tools.brightness import predict_brightness_batch
from spebt_agent.tools.candidates import generate_combinatorial_mutants, generate_single_mutants
from spebt_agent.tools.filters import chromophore_risk_score, exclusion_filter, hard_rule_filter, load_exclusion_set
from spebt_agent.tools.inverse_folding import run_proteinmpnn_adapter
from spebt_agent.tools.ranking import rank_variants, select_top_with_diversity
from spebt_agent.tools.stability import predict_retention72_proxy_batch
from spebt_agent.tools.submission import export_submission_csv


def build_graph():
    """Build a LangGraph wrapper around the deterministic pipeline."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install the graph extra first: python -m pip install -e .[graph]") from exc

    def run_pipeline_node(state: dict) -> dict:
        result = run_design(
            team_name=state.get("team_name"),
            profile=state.get("profile", "stable"),
            resume=state.get("resume", True),
            run_id=state.get("run_id"),
        )
        return {**state, **result}

    graph = StateGraph(dict)
    graph.add_node("run_spEBT_design_pipeline", run_pipeline_node)
    graph.add_edge(START, "run_spEBT_design_pipeline")
    graph.add_edge("run_spEBT_design_pipeline", END)
    return graph.compile()


def _risk_scores(variants: list[dict[str, Any]], competition_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"variant_id": v["variant_id"], **chromophore_risk_score(v.get("mutations", []), competition_cfg["protected_positions"], competition_cfg["pocket_positions"])}
        for v in variants
    ]


def _records_from_csv(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    frame = pd.read_csv(p)
    return frame.to_dict(orient="records")


def _write_records(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _stage_progress(tracker: RunTracker, node: str, **extra: Any) -> None:
    # Filter out keys that conflict with trace()'s own parameters
    safe_extra = {k: v for k, v in extra.items() if k not in ("run_id", "node", "status")}
    tracker.trace(node, "progress", **safe_extra)


def _report_warning_summary(stability_rows: list[dict[str, Any]]) -> list[str]:
    warning_lines: list[str] = []
    missing_esmfold = sum(1 for row in stability_rows if row.get("esmfold_zero_shot_score") is None)
    missing_thermompnn = sum(1 for row in stability_rows if row.get("thermompnn_ddg_pred") is None)
    if missing_esmfold:
        warning_lines.append(f"- ESMFold zero-shot unavailable for {missing_esmfold} candidates; remaining signals were reweighted.")
    if missing_thermompnn:
        warning_lines.append(f"- ThermoMPNN ddG unavailable for {missing_thermompnn} candidates; stability score fell back to available signals.")
    return warning_lines


def _current_stage2_candidates(stage1: list[dict[str, Any]], combos: list[dict[str, Any]], inverse_variants: list[dict[str, Any]], stage2_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seed_limit = int(stage2_cfg.get("stage1_seed_limit", 200))
    total_limit = int(stage2_cfg.get("max_total_candidates", seed_limit + len(combos) + len(inverse_variants)))
    # Always include inverse_folding variants — they carry structural diversity
    base = stage1[:seed_limit] + combos
    budget_for_combos = max(0, total_limit - len(inverse_variants))
    candidates = base[:budget_for_combos] + inverse_variants
    return candidates[:total_limit]


def run_design(
    team_name: str | None = None,
    *,
    profile: str = "stable",
    resume: bool = True,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    max_stage2_candidates: int | None = None,
    esmfold_top_k: int | None = None,
) -> dict:
    root = project_root()
    runtime_configs = build_runtime_configs(
        profile=profile,
        max_stage2_candidates=max_stage2_candidates,
        esmfold_top_k=esmfold_top_k,
    )
    effective_team_name = team_name or runtime_configs["competition"]["team_name"]
    effective_run_id = run_id or str(uuid.uuid4())
    outputs_root = resolve_path(output_dir, root=root) if output_dir else root / "outputs"
    run_dir = outputs_root / "runs" / effective_run_id
    latest_dir = outputs_root / "latest"
    config_fingerprint = fingerprint_config(
        {
            "profile": profile,
            "team_name": effective_team_name,
            "configs": runtime_configs,
            "max_stage2_candidates": max_stage2_candidates,
            "esmfold_top_k": esmfold_top_k,
        }
    )
    tracker = RunTracker(
        run_id=effective_run_id,
        run_dir=run_dir,
        latest_dir=latest_dir,
        profile=profile,
        team_name=effective_team_name,
        config_fingerprint=config_fingerprint,
        allow_resume=resume,
    )
    state: dict[str, Any] = {
        "run_id": effective_run_id,
        "configs": runtime_configs,
        "team_name": effective_team_name,
        "trace": [],
        "profile": profile,
    }

    conn = connect(root / "memory" / "gfp_agent.sqlite")
    record_agent_run(conn, effective_run_id, runtime_configs, "running")

    try:
        raw = runtime_configs["competition"]["raw_data"]
        processed_dir = root / "data" / "processed"
        summary_path = processed_dir / "summary.json"
        tracker.stage_start("prepare_or_load_processed_data")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = prepare_data(resolve_path(raw["refs"]), resolve_path(raw["gfp_xlsx"]), resolve_path(raw["exclusion_csv"]), processed_dir)
        state["processed_summary"] = summary
        tracker.stage_success("prepare_or_load_processed_data", {"summary_json": str(summary_path)}, summary_path=str(summary_path))

        tracker.stage_start("load_config_and_data")
        parent_name = runtime_configs["competition"]["parent_name"]
        parent_seq = load_parent_sequence(processed_dir, parent_name)
        state["parent_name"] = parent_name
        state["parent_sequence"] = parent_seq
        llm = build_llm_client(runtime_configs["model_paths"]["llm"])
        state["llm_strategy"] = llm.summarize_strategy({"parent_name": parent_name, "team_name": effective_team_name})
        tracker.stage_success("load_config_and_data", parent_name=parent_name, parent_length=len(parent_seq))

        gen_cfg = runtime_configs["generation"]
        comp_cfg = runtime_configs["competition"]

        if tracker.stage_can_resume("stage1_ranked", ["stage1_ranked_csv"]):
            tracker.stage_resume_hit("stage1_ranked", artifact=tracker.state["artifacts"]["stage1_ranked_csv"])
            stage1 = _records_from_csv(tracker.state["artifacts"]["stage1_ranked_csv"])
        else:
            tracker.stage_start("stage1_ranked")
            singles = generate_single_mutants(
                parent_name,
                parent_seq,
                comp_cfg["protected_positions"],
                max_candidates=int(gen_cfg["single_mutation"]["max_candidates"]),
            )
            tracker.trace("generate_single_mutants", "success", output_count=len(singles))

            passed, failed = hard_rule_filter(
                singles,
                min_length=comp_cfg["submission"]["min_length"],
                max_length=comp_cfg["submission"]["max_length"],
                must_start_with=comp_cfg["submission"]["must_start_with"],
            )
            exclusion = load_exclusion_set(processed_dir / "exclusion_sequences.txt")
            passed, exclusion_failed = exclusion_filter(passed, exclusion)
            tracker.trace("filter_stage1", "success", output_count=len(passed), failed_count=len(failed) + len(exclusion_failed))

            tracker.trace("stage1_brightness", "start", candidate_count=len(passed))
            brightness = predict_brightness_batch(
                passed,
                root / runtime_configs["model_paths"]["brightness"]["abs_model"],
                root / runtime_configs["model_paths"]["brightness"]["delta_model"],
                runtime_configs["model_paths"]["esmc"],
                progress_callback=lambda node, **extra: _stage_progress(tracker, node, **extra),
            )
            tracker.trace("stage1_brightness", "success", scored_count=len(brightness))
            tracker.trace("stage1_stability", "start", candidate_count=len(passed))
            stability = predict_retention72_proxy_batch(
                passed,
                comp_cfg,
                runtime_configs["model_paths"]["stability"],
                parent_sequence=parent_seq,
                esmc_cfg=runtime_configs["model_paths"]["esmc"],
                progress_callback=lambda node, **extra: _stage_progress(tracker, node, **extra),
            )
            tracker.trace("stage1_stability", "success", scored_count=len(stability))
            stage1 = rank_variants(
                passed,
                brightness,
                stability,
                _risk_scores(passed, comp_cfg),
                threshold=comp_cfg["brightness_elimination"]["relative_brightness_threshold"],
                scoring_cfg=runtime_configs["scoring"],
                stage="stage1",
            )
            state["stage1_ranked"] = stage1
            stage1_path = _write_records(tracker.artifact_path("stage1_ranked.csv"), stage1)
            tracker.stage_success("stage1_ranked", {"stage1_ranked_csv": str(stage1_path)}, output_count=len(stage1))
        state["stage1_ranked"] = stage1

        if tracker.stage_can_resume("stage2_candidates", ["stage2_candidates_csv"]):
            tracker.stage_resume_hit("stage2_candidates", artifact=tracker.state["artifacts"]["stage2_candidates_csv"])
            passed2 = _records_from_csv(tracker.state["artifacts"]["stage2_candidates_csv"])
        else:
            tracker.stage_start("stage2_candidates")
            combo_cfg = gen_cfg["combinatorial_mutation"]
            combos = generate_combinatorial_mutants(
                parent_name,
                parent_seq,
                stage1[: int(combo_cfg["top_single_seeds"])],
                max_order=int(combo_cfg["max_mutations_per_variant"]),
                max_candidates=int(combo_cfg["max_candidates_per_order"]),
            )
            tracker.trace("generate_combinatorial_mutants", "success", output_count=len(combos))
            inverse_result = run_proteinmpnn_adapter(
                parent_name=parent_name,
                parent_sequence=parent_seq,
                parent_pdb=comp_cfg.get("parent_pdb"),
                inverse_folding_cfg=runtime_configs["model_paths"]["inverse_folding"],
                generation_cfg=gen_cfg["inverse_folding"],
            )
            state["inverse_folding"] = inverse_result
            tracker.trace(
                "inverse_folding_adapter",
                inverse_result["status"],
                model=inverse_result.get("model"),
                generated_count=inverse_result.get("generated_count", 0),
                warnings=inverse_result.get("warnings", []),
            )
            for warning in inverse_result.get("warnings", []):
                tracker.add_warning("inverse_folding_adapter", str(warning))
            stage2_candidates = _current_stage2_candidates(
                stage1,
                combos,
                inverse_result.get("variants", []),
                gen_cfg.get("stage2", {}),
            )
            exclusion = load_exclusion_set(processed_dir / "exclusion_sequences.txt")
            passed2, failed2 = hard_rule_filter(stage2_candidates)
            passed2, exclusion_failed2 = exclusion_filter(passed2, exclusion)
            tracker.trace("filter_stage2", "success", output_count=len(passed2), failed_count=len(failed2) + len(exclusion_failed2))
            stage2_candidates_path = _write_records(tracker.artifact_path("stage2_candidates.csv"), passed2)
            tracker.stage_success("stage2_candidates", {"stage2_candidates_csv": str(stage2_candidates_path)}, output_count=len(passed2))
        state["stage2_candidates"] = passed2

        if tracker.stage_can_resume("stage2_brightness", ["stage2_brightness_csv"]):
            tracker.stage_resume_hit("stage2_brightness", artifact=tracker.state["artifacts"]["stage2_brightness_csv"])
            brightness2 = _records_from_csv(tracker.state["artifacts"]["stage2_brightness_csv"])
        else:
            tracker.stage_start("stage2_brightness", candidate_count=len(passed2))
            brightness2 = predict_brightness_batch(
                passed2,
                root / runtime_configs["model_paths"]["brightness"]["abs_model"],
                root / runtime_configs["model_paths"]["brightness"]["delta_model"],
                runtime_configs["model_paths"]["esmc"],
                progress_callback=lambda node, **extra: _stage_progress(tracker, node, **extra),
            )
            stage2_brightness_path = _write_records(tracker.artifact_path("stage2_brightness.csv"), brightness2)
            tracker.stage_success("stage2_brightness", {"stage2_brightness_csv": str(stage2_brightness_path)}, scored_count=len(brightness2))

        if tracker.stage_can_resume("stage2_stability", ["stage2_stability_csv"]):
            tracker.stage_resume_hit("stage2_stability", artifact=tracker.state["artifacts"]["stage2_stability_csv"])
            stability2 = _records_from_csv(tracker.state["artifacts"]["stage2_stability_csv"])
        else:
            tracker.stage_start("stage2_stability", candidate_count=len(passed2))
            stability2 = predict_retention72_proxy_batch(
                passed2,
                comp_cfg,
                runtime_configs["model_paths"]["stability"],
                parent_sequence=parent_seq,
                esmc_cfg=runtime_configs["model_paths"]["esmc"],
                progress_callback=lambda node, **extra: _stage_progress(tracker, node, **extra),
            )
            stage2_stability_path = _write_records(tracker.artifact_path("stage2_stability.csv"), stability2)
            tracker.stage_success("stage2_stability", {"stage2_stability_csv": str(stage2_stability_path)}, scored_count=len(stability2))
            for line in _report_warning_summary(stability2):
                tracker.add_warning("stage2_stability", line)

        if tracker.stage_can_resume("final_rank", ["ranked_variants_csv"]):
            tracker.stage_resume_hit("final_rank", artifact=tracker.state["artifacts"]["ranked_variants_csv"])
            final_ranked = _records_from_csv(tracker.state["artifacts"]["ranked_variants_csv"])
        else:
            tracker.stage_start("final_rank", candidate_count=len(passed2))
            final_ranked = rank_variants(
                passed2,
                brightness2,
                stability2,
                _risk_scores(passed2, comp_cfg),
                threshold=comp_cfg["brightness_elimination"]["relative_brightness_threshold"],
                scoring_cfg=runtime_configs["scoring"],
                stage="final",
            )
            ranked_path = _write_records(tracker.artifact_path("ranked_variants.csv"), final_ranked)
            tracker.stage_success("final_rank", {"ranked_variants_csv": str(ranked_path)}, ranked_count=len(final_ranked))
        state["final_ranked"] = final_ranked

        if tracker.stage_can_resume("select_top6", ["selected_top6_csv"]):
            tracker.stage_resume_hit("select_top6", artifact=tracker.state["artifacts"]["selected_top6_csv"])
            selected = _records_from_csv(tracker.state["artifacts"]["selected_top6_csv"])
        else:
            tracker.stage_start("select_top6", ranked_count=len(final_ranked))
            max_n = comp_cfg["submission"]["max_sequences"]
            # Classify: small (<=2 mut) vs large (>=4 mut or proteinmpnn)
            n_small = 4
            n_large = max_n - n_small  # 2
            small_variants = [v for v in final_ranked if v.get("source") != "proteinmpnn" and int(v.get("num_mutations", 0) or 0) <= 2]
            large_variants = [v for v in final_ranked if v.get("source") == "proteinmpnn" or int(v.get("num_mutations", 0) or 0) >= 4]
            selected_small = select_top_with_diversity(small_variants, max_n=n_small, min_hamming_distance=2)
            selected_large = select_top_with_diversity(large_variants, max_n=n_large, min_hamming_distance=4)
            selected = selected_small + selected_large
            # Fill gaps if not enough in each pool
            if len(selected) < max_n:
                remaining = [v for v in final_ranked if v not in selected]
                fill = select_top_with_diversity(remaining, max_n=max_n - len(selected), min_hamming_distance=2)
                selected.extend(fill)
            selected_path = _write_records(tracker.artifact_path("selected_top6.csv"), selected)
            tracker.stage_success("select_top6", {"selected_top6_csv": str(selected_path)}, selected_count=len(selected))
        state["selected_top6"] = selected

        tracker.stage_start("export_outputs", selected_count=len(selected))
        submission = export_submission_csv(selected, effective_team_name, tracker.artifact_path("submission.csv"))
        report = llm.write_report({"selected_top6": selected, "strategy": state["llm_strategy"]})
        warning_lines = _report_warning_summary(stability2)
        if tracker.state.get("warnings"):
            warning_lines.extend([f"- {item['source']}: {item['message']}" for item in tracker.state["warnings"]])
        if warning_lines:
            report = report.rstrip() + "\n\n## Runtime Warnings\n" + "\n".join(warning_lines) + "\n"
        report_path = tracker.artifact_path("final_report.md")
        report_path.write_text(report, encoding="utf-8")
        tracker.register_artifact("final_report_md", report_path)
        tracker.register_artifact("submission_csv", submission)
        tracker.register_artifact("run_state_json", tracker.state_path)
        tracker.register_artifact("trace_jsonl", tracker.trace_path)
        state["outputs"] = {
            "run_dir": str(run_dir),
            "latest_dir": str(latest_dir),
            "run_state": str(tracker.state_path),
            "trace": str(tracker.trace_path),
            "stage1_ranked": tracker.state["artifacts"].get("stage1_ranked_csv", ""),
            "stage2_candidates": tracker.state["artifacts"].get("stage2_candidates_csv", ""),
            "stage2_brightness": tracker.state["artifacts"].get("stage2_brightness_csv", ""),
            "stage2_stability": tracker.state["artifacts"].get("stage2_stability_csv", ""),
            "ranked_variants": tracker.state["artifacts"].get("ranked_variants_csv", ""),
            "selected_top6": tracker.state["artifacts"].get("selected_top6_csv", ""),
            "submission": str(submission),
            "final_report": str(report_path),
        }
        tracker.state["outputs"] = state["outputs"]
        tracker.save()
        tracker.stage_success(
            "export_outputs",
            {
                "final_report_md": str(report_path),
                "submission_csv": str(submission),
                "run_state_json": str(tracker.state_path),
                "trace_jsonl": str(tracker.trace_path),
            },
            outputs_dir=str(run_dir),
        )
        tracker.set_status("success")
        tracker.publish_latest(
            [
                "stage1_ranked_csv",
                "stage2_candidates_csv",
                "stage2_brightness_csv",
                "stage2_stability_csv",
                "ranked_variants_csv",
                "selected_top6_csv",
                "submission_csv",
                "final_report_md",
                "run_state_json",
                "trace_jsonl",
            ]
        )
        record_agent_run(conn, effective_run_id, runtime_configs, "success", state["outputs"]["final_report"], state["outputs"]["submission"])
        # ── agent-friendly top-level keys ───────────────────────────────
        state["ok"] = True
        state["status"] = "success"
        state["run_dir"] = str(run_dir)
        state["latest_dir"] = str(latest_dir)
        state["artifacts"] = dict(state["outputs"])
        state["warnings"] = tracker.state.get("warnings", [])
        state["failure_reason"] = ""
        return state
    except Exception as exc:
        tracker.fail_running_stage(str(exc))
        tracker.set_status("failed", failure_reason=str(exc))
        record_agent_run(conn, effective_run_id, runtime_configs, "failed")
        # ── return structured error instead of raising ──────────────────
        state["ok"] = False
        state["status"] = "failed"
        state["run_dir"] = str(run_dir)
        state["latest_dir"] = str(latest_dir)
        state["artifacts"] = state.get("outputs", {})
        state["warnings"] = tracker.state.get("warnings", [])
        state["failure_reason"] = str(exc)
        return state
