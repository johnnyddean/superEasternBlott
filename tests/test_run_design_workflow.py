import json
from pathlib import Path

from spebt_agent import config as config_module
from spebt_agent import graph


class DummyLLM:
    def summarize_strategy(self, context):
        return f"strategy_for_{context['team_name']}"

    def write_report(self, context):
        return f"selected={len(context.get('selected_top6', []))}"


def _runtime_configs():
    return {
        "competition": {
            "team_name": "spEBT",
            "parent_name": "sfGFP",
            "parent_pdb": "2B3P",
            "raw_data": {"refs": "refs.txt", "gfp_xlsx": "gfp.xlsx", "exclusion_csv": "exclusion.csv"},
            "protected_positions": [],
            "pocket_positions": [],
            "brightness_elimination": {"relative_brightness_threshold": 0.0},
            "submission": {"min_length": 220, "max_length": 250, "must_start_with": "M", "max_sequences": 6},
        },
        "generation": {
            "single_mutation": {"max_candidates": 3},
            "stage2": {"stage1_seed_limit": 2, "max_total_candidates": 4},
            "combinatorial_mutation": {"top_single_seeds": 2, "max_mutations_per_variant": 2, "max_candidates_per_order": 2},
            "inverse_folding": {"enabled": False, "num_seq_per_target": 1, "sampling_temps": [0.1]},
        },
        "model_paths": {
            "llm": {},
            "esmc": {"model_dir": "external/weights/esmc/ESMC-600M", "model_name": "biohub/ESMC-600M"},
            "brightness": {"abs_model": "artifacts/models/brightness/abs.joblib", "delta_model": "artifacts/models/brightness/delta.joblib"},
            "stability": {"esmc_pll_enabled": False, "esmfold_enabled": False},
            "inverse_folding": {"proteinmpnn_out_dir": "artifacts/proteinmpnn_outputs"},
        },
        "scoring": {},
    }


def _variant(variant_id, n_mut=1):
    return {
        "variant_id": variant_id,
        "parent": "sfGFP",
        "sequence": "M" + "A" * 219,
        "mutations": [f"A{idx+2}G" for idx in range(n_mut)],
        "num_mutations": n_mut,
    }


def test_build_runtime_configs_profiles():
    stable = config_module.build_runtime_configs(profile="stable")
    full = config_module.build_runtime_configs(profile="full_search")
    assert stable["generation"]["stage2"]["max_total_candidates"] == 900
    assert stable["model_paths"]["stability"]["esmfold_top_k"] == 8
    assert full["generation"]["stage2"]["max_total_candidates"] == 5200
    assert full["model_paths"]["stability"]["esmfold_top_k"] == 24


def test_run_design_writes_run_outputs_and_latest(monkeypatch, tmp_path):
    root = tmp_path
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed" / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    monkeypatch.setattr(graph, "project_root", lambda: root)
    monkeypatch.setattr(graph, "build_runtime_configs", lambda **kwargs: _runtime_configs())
    monkeypatch.setattr(graph, "load_parent_sequence", lambda *_args, **_kwargs: "M" + "A" * 219)
    monkeypatch.setattr(graph, "build_llm_client", lambda *_args, **_kwargs: DummyLLM())
    monkeypatch.setattr(graph, "load_exclusion_set", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(graph, "hard_rule_filter", lambda variants, **kwargs: (variants, []))
    monkeypatch.setattr(graph, "exclusion_filter", lambda variants, exclusion: (variants, []))

    captured = {}

    def fake_generate_single_mutants(parent_name, parent_seq, protected_positions, max_candidates=0):
        captured["max_candidates"] = max_candidates
        return [_variant("single_1"), _variant("single_2")]

    def fake_predict_brightness_batch(variants, *args, progress_callback=None, **kwargs):
        if progress_callback is not None:
            progress_callback("brightness_embeddings_start", variant_count=len(variants))
            progress_callback("brightness_embeddings_done", embedding_count=len(variants), feature_dim=4)
        return [
            {
                "variant_id": variant["variant_id"],
                "brightness_abs_pred": 0.5,
                "brightness_delta_pred": 0.5,
                "predicted_relative_brightness": 0.6,
                "brightness_score": 0.6,
                "model": "ESMC_tree_abs_delta_ensemble",
            }
            for variant in variants
        ]

    def fake_predict_retention72_proxy_batch(variants, *args, progress_callback=None, **kwargs):
        if progress_callback is not None:
            progress_callback("stability_esmc_zero_shot_start", variant_count=len(variants))
            progress_callback("stability_esmc_zero_shot_done", scored_count=len(variants))
        return [
            {
                "variant_id": variant["variant_id"],
                "predicted_retention72": 0.55,
                "thermostability_score": 0.55,
                "ddg_stability_score": None,
                "thermompnn_ddg_pred": None,
                "model": "zero_shot_stability_dual_signal_v1",
                "warnings": [],
            }
            for variant in variants
        ]

    def fake_rank_variants(variants, brightness, stability, *_args, **kwargs):
        brightness_map = {item["variant_id"]: item for item in brightness}
        stability_map = {item["variant_id"]: item for item in stability}
        ranked = []
        for idx, variant in enumerate(variants):
            row = dict(variant)
            row.update(brightness_map[variant["variant_id"]])
            row.update(stability_map[variant["variant_id"]])
            row["score"] = 1.0 - idx * 0.1
            ranked.append(row)
        return ranked

    monkeypatch.setattr(graph, "generate_single_mutants", fake_generate_single_mutants)
    monkeypatch.setattr(graph, "predict_brightness_batch", fake_predict_brightness_batch)
    monkeypatch.setattr(graph, "predict_retention72_proxy_batch", fake_predict_retention72_proxy_batch)
    monkeypatch.setattr(graph, "rank_variants", fake_rank_variants)
    monkeypatch.setattr(graph, "generate_combinatorial_mutants", lambda *_args, **_kwargs: [_variant("combo_1", n_mut=2)])
    monkeypatch.setattr(
        graph,
        "run_proteinmpnn_adapter",
        lambda **_kwargs: {"status": "skipped", "model": "proteinmpnn_disabled", "warnings": [], "generated_count": 0, "variants": []},
    )
    monkeypatch.setattr(graph, "select_top_with_diversity", lambda ranked, **_kwargs: ranked[:2])

    state = graph.run_design(team_name="TeamStable", run_id="run-stable-1", resume=True, output_dir=root / "outputs")

    assert captured["max_candidates"] == 3
    assert Path(state["outputs"]["submission"]).exists()
    assert Path(state["outputs"]["final_report"]).exists()
    assert Path(state["outputs"]["run_state"]).exists()
    assert Path(state["outputs"]["stage2_brightness"]).exists()
    assert (root / "outputs" / "latest" / "submission.csv").exists()

    run_state = json.loads(Path(state["outputs"]["run_state"]).read_text(encoding="utf-8"))
    assert run_state["status"] == "success"
    assert run_state["stage_status"]["stage2_brightness"]["status"] == "success"
    assert run_state["stage_status"]["stage2_stability"]["status"] == "success"


def test_run_design_resume_skips_completed_heavy_stages(monkeypatch, tmp_path):
    root = tmp_path
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed" / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    monkeypatch.setattr(graph, "project_root", lambda: root)
    monkeypatch.setattr(graph, "build_runtime_configs", lambda **kwargs: _runtime_configs())
    monkeypatch.setattr(graph, "load_parent_sequence", lambda *_args, **_kwargs: "M" + "A" * 219)
    monkeypatch.setattr(graph, "build_llm_client", lambda *_args, **_kwargs: DummyLLM())
    monkeypatch.setattr(graph, "load_exclusion_set", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(graph, "hard_rule_filter", lambda variants, **kwargs: (variants, []))
    monkeypatch.setattr(graph, "exclusion_filter", lambda variants, exclusion: (variants, []))
    monkeypatch.setattr(graph, "generate_combinatorial_mutants", lambda *_args, **_kwargs: [_variant("combo_1", n_mut=2)])
    monkeypatch.setattr(
        graph,
        "run_proteinmpnn_adapter",
        lambda **_kwargs: {"status": "skipped", "model": "proteinmpnn_disabled", "warnings": [], "generated_count": 0, "variants": []},
    )
    monkeypatch.setattr(graph, "select_top_with_diversity", lambda ranked, **_kwargs: ranked[:1])

    counters = {"generate_single_mutants": 0, "predict_brightness_batch": 0, "predict_retention72_proxy_batch": 0}

    def fake_generate_single_mutants(*_args, **_kwargs):
        counters["generate_single_mutants"] += 1
        return [_variant("single_1"), _variant("single_2")]

    def fake_predict_brightness_batch(variants, *args, progress_callback=None, **kwargs):
        counters["predict_brightness_batch"] += 1
        return [
            {
                "variant_id": variant["variant_id"],
                "brightness_abs_pred": 0.5,
                "brightness_delta_pred": 0.5,
                "predicted_relative_brightness": 0.6,
                "brightness_score": 0.6,
                "model": "ESMC_tree_abs_delta_ensemble",
            }
            for variant in variants
        ]

    def fake_predict_retention72_proxy_batch(variants, *args, progress_callback=None, **kwargs):
        counters["predict_retention72_proxy_batch"] += 1
        return [
            {
                "variant_id": variant["variant_id"],
                "predicted_retention72": 0.55,
                "thermostability_score": 0.55,
                "ddg_stability_score": None,
                "thermompnn_ddg_pred": None,
                "model": "zero_shot_stability_dual_signal_v1",
                "warnings": [],
            }
            for variant in variants
        ]

    def fake_rank_variants(variants, brightness, stability, *_args, **kwargs):
        brightness_map = {item["variant_id"]: item for item in brightness}
        stability_map = {item["variant_id"]: item for item in stability}
        return [{**variant, **brightness_map[variant["variant_id"]], **stability_map[variant["variant_id"]], "score": 1.0} for variant in variants]

    monkeypatch.setattr(graph, "generate_single_mutants", fake_generate_single_mutants)
    monkeypatch.setattr(graph, "predict_brightness_batch", fake_predict_brightness_batch)
    monkeypatch.setattr(graph, "predict_retention72_proxy_batch", fake_predict_retention72_proxy_batch)
    monkeypatch.setattr(graph, "rank_variants", fake_rank_variants)

    graph.run_design(team_name="ResumeTeam", run_id="resume-run-1", resume=True, output_dir=root / "outputs")
    graph.run_design(team_name="ResumeTeam", run_id="resume-run-1", resume=True, output_dir=root / "outputs")

    assert counters["generate_single_mutants"] == 1
    assert counters["predict_brightness_batch"] == 2
    assert counters["predict_retention72_proxy_batch"] == 2
