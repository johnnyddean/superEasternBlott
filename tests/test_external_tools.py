from spebt_agent.tools.external_support import diff_mutations
from spebt_agent.tools.inverse_folding import build_proteinmpnn_command, run_proteinmpnn_adapter
import pandas as pd

from spebt_agent.tools.stability import _thermompnn_parent_offset, predict_retention72_proxy_batch


def test_diff_mutations_same_length():
    muts = diff_mutations("MAAA", "MGAT")
    assert muts == ["A2G", "A4T"]


def test_build_proteinmpnn_command_contains_core_flags():
    cmd = build_proteinmpnn_command("repo", "input.pdb", "out", "weights")
    assert "--pdb_path" in cmd
    assert "--path_to_model_weights" in cmd
    assert "--num_seq_per_target" in cmd


def test_run_proteinmpnn_adapter_skips_when_disabled():
    result = run_proteinmpnn_adapter(
        parent_name="sfGFP",
        parent_sequence="MAAA",
        parent_pdb="2B3P",
        inverse_folding_cfg={
            "proteinmpnn_script": "external/repositories/ProteinMPNN/protein_mpnn_run.py",
            "proteinmpnn_weights_dir": "external/weights/ProteinMPNN",
            "proteinmpnn_out_dir": "artifacts/proteinmpnn_outputs",
            "structure_search_dirs": ["data/raw/structures"],
        },
        generation_cfg={"enabled": False, "num_seq_per_target": 1, "sampling_temps": [0.1]},
    )
    assert result["status"] == "skipped"
    assert result["model"] == "proteinmpnn_disabled"


def test_predict_retention72_proxy_batch_runs_without_optional_configs():
    out = predict_retention72_proxy_batch([{"variant_id": "v1", "sequence": "M" + "A" * 219, "mutations": ["A2G"]}])
    assert out[0]["model"] == "rule_based_retention72_proxy_v1"
    assert 0.0 <= out[0]["predicted_retention72"] <= 1.0


def test_thermompnn_parent_offset_handles_missing_start_methionine():
    df = pd.DataFrame(
        [
            {"position": 0, "wildtype": "S", "mutation": "A", "ddG_pred": 0.1},
            {"position": 1, "wildtype": "K", "mutation": "A", "ddG_pred": 0.2},
            {"position": 2, "wildtype": "G", "mutation": "A", "ddG_pred": 0.3},
        ]
    )
    offset, warning = _thermompnn_parent_offset(df, "MSKG")
    assert offset == 1
    assert warning == ""


def test_predict_retention72_proxy_batch_applies_thermompnn_ddg(monkeypatch):
    def fake_run_thermompnn_adapter(variants, parent_pdb, stability_cfg, parent_sequence=None):
        return {
            "status": "success",
            "model": "thermompnn_custom_inference",
            "warnings": [],
            "scores": [{"variant_id": "v1", "ddg_pred": -0.5}],
        }

    def fake_run_rasp_adapter(parent_pdb, stability_cfg):
        return {"status": "skipped", "model": "rasp_parser_ready", "warnings": []}

    monkeypatch.setattr("spebt_agent.tools.stability.run_thermompnn_adapter", fake_run_thermompnn_adapter)
    monkeypatch.setattr("spebt_agent.tools.stability.run_rasp_adapter", fake_run_rasp_adapter)

    out = predict_retention72_proxy_batch(
        [{"variant_id": "v1", "sequence": "M" + "A" * 219, "mutations": ["A2G"]}],
        competition_cfg={"parent_pdb": "2B3P"},
        stability_cfg={"thermompnn_repo": "repo", "thermompnn_script": "script"},
        parent_sequence="MAA",
    )
    assert out[0]["model"] == "rule_based_retention72_plus_thermompnn"
    assert out[0]["thermompnn_ddg_pred"] == -0.5


def test_predict_retention72_proxy_batch_uses_zero_shot_dual_signal(monkeypatch):
    def fake_score_sequences_zero_shot(**kwargs):
        return [
            {
                "sequence": kwargs["sequences"][0],
                "esmc_pll_abs": -1.0,
                "esmc_pll_delta_vs_parent": 0.2,
                "esmc_zero_shot_score": 0.8,
            }
        ]

    def fake_score_sequences_esmfold_zero_shot(**kwargs):
        return [
            {
                "sequence": kwargs["sequences"][0],
                "esmfold_plddt_mean": 82.0,
                "esmfold_plddt_delta_vs_parent": 1.5,
                "esmfold_zero_shot_score": 0.7,
                "esmfold_low_confidence_fraction": 0.1,
                "esmfold_status": "success",
                "esmfold_warning": "",
            }
        ]

    def fake_run_thermompnn_adapter(variants, parent_pdb, stability_cfg, parent_sequence=None):
        return {"status": "skipped", "model": "thermompnn_missing", "warnings": [], "scores": []}

    def fake_run_rasp_adapter(parent_pdb, stability_cfg):
        return {"status": "skipped", "model": "rasp_parser_ready", "warnings": []}

    monkeypatch.setattr("spebt_agent.tools.stability.score_sequences_zero_shot", fake_score_sequences_zero_shot)
    monkeypatch.setattr("spebt_agent.tools.stability.score_sequences_esmfold_zero_shot", fake_score_sequences_esmfold_zero_shot)
    monkeypatch.setattr("spebt_agent.tools.stability.run_thermompnn_adapter", fake_run_thermompnn_adapter)
    monkeypatch.setattr("spebt_agent.tools.stability.run_rasp_adapter", fake_run_rasp_adapter)

    out = predict_retention72_proxy_batch(
        [{"variant_id": "v1", "sequence": "MAAA", "mutations": ["A2G"]}],
        competition_cfg={"parent_pdb": "2B3P"},
        stability_cfg={
            "esmc_pll_enabled": True,
            "esmfold_enabled": True,
            "esmfold_top_k": 1,
            "zero_shot_weights": {"esmc": 0.45, "esmfold": 0.25, "thermompnn": 0.30},
        },
        parent_sequence="MAAA",
        esmc_cfg={"model_dir": "x", "model_name": "m"},
    )
    assert out[0]["model"] == "zero_shot_stability_dual_signal_v1"
    assert out[0]["esmc_zero_shot_score"] == 0.8
    assert out[0]["esmfold_zero_shot_score"] == 0.7
