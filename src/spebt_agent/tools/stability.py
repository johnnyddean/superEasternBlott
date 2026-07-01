from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from spebt_agent.paths import resolve_path
from spebt_agent.tools.esmc import embed_sequences_real, score_sequences_zero_shot
from spebt_agent.tools.external_support import resolve_structure_path, resolve_tool_python
from spebt_agent.tools.structure import score_sequences_esmfold_zero_shot


def _base_proxy(variant: dict) -> float:
    seq = variant["sequence"]
    muts = variant.get("mutations", [])
    length_score = 1.0 if 220 <= len(seq) <= 250 else 0.0
    m_start_score = 1.0 if seq.startswith("M") else 0.0
    mutation_burden_score = float(np.exp(-max(0, len(muts) - 4) / 4.0))
    cys_score = 1.0 if seq.count("C") <= 4 else 0.8
    charged = sum(seq.count(x) for x in "DEKR")
    charged_ratio = charged / len(seq)
    charge_score = 1.0 if 0.12 <= charged_ratio <= 0.35 else 0.8
    return float(
        0.25 * length_score
        + 0.20 * m_start_score
        + 0.25 * mutation_burden_score
        + 0.15 * cys_score
        + 0.15 * charge_score
    )


def build_stability_feature_matrix(records: list[dict[str, Any]], embeddings: np.ndarray, parent_categories: list[str]) -> np.ndarray:
    numeric = np.array(
        [
            [
                float(record.get("num_mutations", len(record.get("mutations", [])))),
                float(len(record["sequence"])),
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    parent_one_hot = np.zeros((len(records), len(parent_categories)), dtype=np.float32)
    parent_index = {name: idx for idx, name in enumerate(parent_categories)}
    for row_idx, record in enumerate(records):
        parent_name = str(record.get("parent") or record.get("parent_id") or "unknown")
        if parent_name in parent_index:
            parent_one_hot[row_idx, parent_index[parent_name]] = 1.0
    return np.hstack([embeddings.astype(np.float32), numeric, parent_one_hot])


def _load_retention72_bundle(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    bundle_path = Path(path)
    if not bundle_path.exists():
        return None
    loaded = joblib.load(bundle_path)
    if isinstance(loaded, dict) and "estimator" in loaded:
        return loaded
    return {"estimator": loaded, "parent_categories": [], "esmc": {}, "target": "retention72"}


def _predict_retention72_model_batch(
    variants: list[dict[str, Any]],
    bundle: dict[str, Any],
    esmc_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    merged_cfg = {**bundle.get("esmc", {}), **(esmc_cfg or {})}
    embeddings = embed_sequences_real(
        sequences=[variant["sequence"] for variant in variants],
        model_dir=merged_cfg["model_dir"],
        model_name=merged_cfg["model_name"],
        cache_dir=merged_cfg.get("embedding_cache_dir"),
        pooling=merged_cfg.get("pooling", "mean"),
        batch_size=int(merged_cfg.get("batch_size", 8)),
        device=merged_cfg.get("device", "auto"),
        dtype=merged_cfg.get("dtype", "float32"),
    )
    features = build_stability_feature_matrix(variants, embeddings, list(bundle.get("parent_categories", [])))
    preds = np.asarray(bundle["estimator"].predict(features), dtype=np.float32)
    preds = np.clip(preds, 0.0, 1.0)
    return [
        {
            "variant_id": variant["variant_id"],
            "predicted_retention72": float(pred),
            "thermostability_score": float(pred),
            "model": "ESMC_retention72_regressor_v1",
        }
        for variant, pred in zip(variants, preds)
    ]


def _ddg_to_score(ddg: float | None) -> float | None:
    if ddg is None:
        return None
    return float(1.0 / (1.0 + math.exp(ddg)))


def _combine_weighted_scores(components: list[tuple[float | None, float]]) -> float:
    usable = [(value, weight) for value, weight in components if value is not None]
    if not usable:
        return 0.5
    total_weight = sum(weight for _, weight in usable)
    if total_weight <= 0:
        return 0.5
    return float(sum(float(value) * weight for value, weight in usable) / total_weight)


def _thermompnn_assets_present(repo_dir: Path) -> dict[str, bool]:
    custom_script = repo_dir / "analysis" / "custom_inference.py"
    models_dir = repo_dir / "models"
    has_ckpt = False
    if models_dir.exists():
        has_ckpt = any(models_dir.glob("*.ckpt")) or any(models_dir.glob("*.pt"))
    return {
        "repo": repo_dir.exists(),
        "custom_inference": custom_script.exists(),
        "models_dir": models_dir.exists(),
        "checkpoint": has_ckpt,
    }


def _rasp_assets_present(repo_dir: Path, reduce_exe: Path) -> dict[str, bool]:
    return {
        "repo": repo_dir.exists(),
        "parser_script": (repo_dir / "src" / "pdb_parser_scripts" / "parse_pdbs_pred.sh").exists(),
        "reduce_exe": reduce_exe.exists(),
    }


def run_foldx_ddg_adapter(*args, **kwargs):
    return {"ddg_pred": None, "model": "foldx_skipped", "warnings": ["FoldX adapter is reserved but not configured."]}


def _thermompnn_sequence_from_df(df) -> tuple[str, int]:
    if df.empty:
        return "", 0
    first_rows = df.sort_values("position").drop_duplicates(subset=["position"], keep="first")
    sequence = "".join(str(x) for x in first_rows["wildtype"].tolist())
    min_position = int(first_rows["position"].min())
    return sequence, min_position


def _thermompnn_parent_offset(df, parent_sequence: str) -> tuple[int | None, str]:
    structure_sequence, min_position = _thermompnn_sequence_from_df(df)
    if not structure_sequence:
        return None, "ThermoMPNN output CSV was empty."
    first_rows = df.sort_values("position").drop_duplicates(subset=["position"], keep="first")
    best_offset = None
    best_matches = -1
    for offset in range(-len(parent_sequence), len(parent_sequence)):
        matches = 0
        for _, row in first_rows.iterrows():
            parent_idx = int(row["position"]) + offset
            if 0 <= parent_idx < len(parent_sequence) and parent_sequence[parent_idx] == str(row["wildtype"]):
                matches += 1
        if matches > best_matches:
            best_matches = matches
            best_offset = offset

    if best_offset is None or best_matches <= 0:
        return None, "Could not align ThermoMPNN structure sequence to parent sequence."

    match_ratio = best_matches / max(len(first_rows), 1)
    if match_ratio < 0.8:
        return None, f"ThermoMPNN alignment to parent sequence was too weak ({best_matches}/{len(first_rows)} positions matched)."
    return best_offset, ""


def run_thermompnn_adapter(variants: list[dict], parent_pdb: str | None, stability_cfg: dict, parent_sequence: str | None = None) -> dict:
    repo_dir = resolve_path(stability_cfg["thermompnn_repo"])
    script_path = resolve_path(stability_cfg["thermompnn_script"])
    out_dir = resolve_path(stability_cfg["thermompnn_output_dir"])
    structure_path = resolve_structure_path(parent_pdb, stability_cfg.get("structure_search_dirs"))
    env_python = resolve_tool_python("stability")
    assets = _thermompnn_assets_present(repo_dir)

    if env_python is None:
        return {"status": "skipped", "model": "thermompnn_env_missing", "warnings": ["spebt_stability environment python was not found."], "assets": assets}
    if structure_path is None:
        return {"status": "skipped", "model": "thermompnn_missing_pdb", "warnings": ["No structure PDB was found in configured search directories."], "assets": assets}
    if not assets["repo"] or not assets["custom_inference"]:
        return {"status": "skipped", "model": "thermompnn_assets_missing", "warnings": ["ThermoMPNN repository or inference script is incomplete."], "assets": assets}
    if not assets["checkpoint"] and not stability_cfg.get("thermompnn_checkpoint"):
        return {"status": "skipped", "model": "thermompnn_missing_checkpoint", "warnings": ["No ThermoMPNN checkpoint was found; add one under external/repositories/ThermoMPNN/models or config."], "assets": assets}

    checkpoint = stability_cfg.get("thermompnn_checkpoint") or ""
    command = [
        str(env_python),
        str(script_path),
        "--pdb",
        str(structure_path),
        "--chain",
        stability_cfg.get("thermompnn_chain", "A"),
        "--out_dir",
        str(out_dir),
    ]
    if checkpoint:
        command.extend(["--model_path", str(resolve_path(checkpoint))])

    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    csv_path = out_dir / f"ThermoMPNN_inference_{Path(structure_path).stem}.csv"
    predictions: dict[str, float] = {}
    alignment_warning = ""
    parent_offset = None
    if proc.returncode == 0 and csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        if parent_sequence:
            parent_offset, alignment_warning = _thermompnn_parent_offset(df, parent_sequence)
        for _, row in df.iterrows():
            reported_position = int(row["position"])
            full_position = reported_position
            if parent_offset is not None:
                full_position = reported_position + parent_offset + 1
            key = f"{row['wildtype']}{full_position}{row['mutation']}"
            predictions[key] = float(row["ddG_pred"])

    scored = []
    for variant in variants:
        muts = variant.get("mutations", [])
        ddgs = [predictions[m] for m in muts if m in predictions]
        if ddgs:
            avg_ddg = float(np.mean(ddgs))
            scored.append({"variant_id": variant["variant_id"], "ddg_pred": avg_ddg})
    status = "success" if proc.returncode == 0 else "failed"
    warnings = ([w for w in [alignment_warning] if w] if proc.returncode == 0 else [proc.stderr.strip() or "ThermoMPNN execution failed."])
    if proc.returncode == 0 and len(predictions) == 0:
        status = "partial"
        warnings = warnings + ["ThermoMPNN finished but did not map any ddG predictions onto candidate mutations."]
    return {
        "status": status,
        "model": "thermompnn_custom_inference",
        "warnings": warnings,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
        "pdb_path": str(structure_path),
        "command": command,
        "assets": assets,
        "mapped_prediction_count": len(predictions),
        "parent_offset": parent_offset,
        "scores": scored,
    }


def run_rasp_adapter(parent_pdb: str | None, stability_cfg: dict) -> dict:
    repo_dir = resolve_path(stability_cfg["rasp_repo"])
    parser_script = resolve_path(stability_cfg["rasp_parser_script"])
    reduce_exe = resolve_path(stability_cfg["rasp_reduce_exe"])
    structure_path = resolve_structure_path(parent_pdb, stability_cfg.get("structure_search_dirs"))
    assets = _rasp_assets_present(repo_dir, reduce_exe)
    if structure_path is None:
        return {"status": "skipped", "model": "rasp_missing_pdb", "warnings": ["No structure PDB was found in configured search directories."], "assets": assets}
    if not assets["repo"] or not assets["parser_script"]:
        return {"status": "skipped", "model": "rasp_assets_missing", "warnings": ["RaSP repository or parser script is incomplete."], "assets": assets}
    if not assets["reduce_exe"]:
        return {"status": "skipped", "model": "rasp_missing_reduce", "warnings": ["RaSP reduce executable is missing; parser cannot run."], "assets": assets}
    if subprocess.run(["where", "bash"], capture_output=True, text=True).returncode != 0:
        return {"status": "skipped", "model": "rasp_bash_missing", "warnings": ["RaSP parser script requires bash on Windows."], "assets": assets}
    return {
        "status": "skipped",
        "model": "rasp_parser_ready",
        "warnings": ["RaSP pipeline is wired for preparation, but direct ddG scoring is not yet implemented in spEBT."],
        "assets": assets,
        "pdb_path": str(structure_path),
    }


def predict_zero_shot_stability_batch(
    variants: list[dict],
    competition_cfg: dict | None = None,
    stability_cfg: dict | None = None,
    parent_sequence: str | None = None,
    esmc_cfg: dict | None = None,
    progress_callback=None,
):
    outputs = []
    thermompnn_result = None
    rasp_result = None
    thermompnn_scores: dict[str, float] = {}
    esmc_scores: dict[str, dict[str, Any]] = {}
    esmfold_scores: dict[str, dict[str, Any]] = {}
    zero_shot_cfg = (stability_cfg or {}).get("zero_shot_weights", {})
    esmc_weight = float(zero_shot_cfg.get("esmc", 0.45))
    esmfold_weight = float(zero_shot_cfg.get("esmfold", 0.25))
    thermompnn_weight = float(zero_shot_cfg.get("thermompnn", 0.30))
    if variants and esmc_cfg and (stability_cfg or {}).get("esmc_pll_enabled", True):
        if progress_callback is not None:
            progress_callback("stability_esmc_zero_shot_start", variant_count=len(variants))
        seq_scores = score_sequences_zero_shot(
            sequences=[variant["sequence"] for variant in variants],
            model_dir=resolve_path(esmc_cfg["model_dir"]),
            model_name=esmc_cfg["model_name"],
            cache_dir=resolve_path(esmc_cfg.get("embedding_cache_dir", "artifacts/embeddings/esmc")),
            batch_size=int(esmc_cfg.get("batch_size", 8)),
            device=esmc_cfg.get("device", "auto"),
            dtype=esmc_cfg.get("dtype", "float32"),
            parent_sequence=parent_sequence,
        )
        for variant, score in zip(variants, seq_scores):
            esmc_scores[variant["variant_id"]] = score
        if progress_callback is not None:
            progress_callback("stability_esmc_zero_shot_done", scored_count=len(seq_scores))
    if variants and stability_cfg and stability_cfg.get("esmfold_enabled", True):
        structure_candidates = sorted(
            variants,
            key=lambda variant: esmc_scores.get(variant["variant_id"], {}).get("esmc_zero_shot_score", 0.0),
            reverse=True,
        )
        top_k = int(stability_cfg.get("esmfold_top_k", 24))
        selected = structure_candidates[:top_k]
        if progress_callback is not None:
            progress_callback("stability_esmfold_start", selected_count=len(selected), total_count=len(variants))
        try:
            structure_scores = score_sequences_esmfold_zero_shot(
                sequences=[variant["sequence"] for variant in selected],
                model_dir=resolve_path(stability_cfg.get("esmfold_model_dir", "external/weights/esmfold/esmfold_v1")),
                cache_dir=resolve_path(stability_cfg.get("esmfold_cache_dir", "artifacts/structures/esmfold")),
                parent_sequence=parent_sequence,
                device=stability_cfg.get("esmfold_device", "auto"),
                batch_size=int(stability_cfg.get("esmfold_batch_size", 1)),
                top_k=None,
            )
            for variant, score in zip(selected, structure_scores):
                esmfold_scores[variant["variant_id"]] = score
            if progress_callback is not None:
                progress_callback("stability_esmfold_done", scored_count=len(structure_scores), selected_count=len(selected))
        except Exception as exc:
            if progress_callback is not None:
                progress_callback("stability_esmfold_failed", selected_count=len(selected), error=str(exc))
            for variant in selected:
                esmfold_scores[variant["variant_id"]] = {
                    "sequence": variant["sequence"],
                    "esmfold_plddt_mean": None,
                    "esmfold_plddt_delta_vs_parent": None,
                    "esmfold_zero_shot_score": None,
                    "esmfold_low_confidence_fraction": None,
                    "esmfold_status": "failed",
                    "esmfold_warning": str(exc),
                }
    if variants and competition_cfg and stability_cfg:
        if progress_callback is not None:
            progress_callback("stability_thermompnn_start", variant_count=len(variants))
        thermompnn_result = run_thermompnn_adapter(variants, competition_cfg.get("parent_pdb"), stability_cfg, parent_sequence=parent_sequence)
        if progress_callback is not None:
            progress_callback(
                "stability_thermompnn_done",
                status=thermompnn_result.get("status"),
                mapped_prediction_count=int(thermompnn_result.get("mapped_prediction_count", 0)),
            )
        if progress_callback is not None:
            progress_callback("stability_rasp_start")
        rasp_result = run_rasp_adapter(competition_cfg.get("parent_pdb"), stability_cfg)
        if progress_callback is not None:
            progress_callback("stability_rasp_done", status=rasp_result.get("status"), model=rasp_result.get("model"))
        for item in thermompnn_result.get("scores", []):
            thermompnn_scores[item["variant_id"]] = float(item["ddg_pred"])

    for v in variants:
        vid = v["variant_id"]
        esmc_item = esmc_scores.get(vid, {})
        esmfold_item = esmfold_scores.get(vid, {})
        sequence_base = float(esmc_item.get("esmc_zero_shot_score", _base_proxy(v)))
        ddg = thermompnn_scores.get(v["variant_id"])
        ddg_score = _ddg_to_score(ddg)
        structure_score = esmfold_item.get("esmfold_zero_shot_score")
        structure_score = float(structure_score) if structure_score is not None else None
        base_score = _combine_weighted_scores(
            [
                (sequence_base, esmc_weight),
                (structure_score, esmfold_weight),
            ]
        )
        adjusted = _combine_weighted_scores(
            [
                (sequence_base, esmc_weight),
                (structure_score, esmfold_weight),
                (ddg_score, thermompnn_weight),
            ]
        )
        warnings = []
        has_esmc = bool(esmc_item)
        has_structure = structure_score is not None
        if not has_esmc:
            warnings.append("ESMC zero-shot score was unavailable; fallback proxy was used.")
        if structure_score is None:
            warnings.append("ESMFold zero-shot signal was unavailable for this variant.")
            if esmfold_item.get("esmfold_warning"):
                warnings.append(str(esmfold_item["esmfold_warning"]))
        if ddg is None:
            warnings.append("ThermoMPNN ddG was unavailable for this variant.")
        model = "rule_based_retention72_proxy_v1"
        if has_esmc or has_structure:
            model = "zero_shot_stability_dual_signal_v1"
        if ddg is not None and (has_esmc or has_structure):
            model = "zero_shot_stability_dual_signal_plus_thermompnn_v1"
        elif ddg is not None:
            model = "rule_based_retention72_plus_thermompnn"
        outputs.append(
            {
                "variant_id": vid,
                "stability_base_score": base_score,
                "retention72_base_pred": base_score,
                "esmc_pll_abs": esmc_item.get("esmc_pll_abs"),
                "esmc_pll_delta_vs_parent": esmc_item.get("esmc_pll_delta_vs_parent"),
                "esmc_zero_shot_score": esmc_item.get("esmc_zero_shot_score"),
                "esmfold_plddt_mean": esmfold_item.get("esmfold_plddt_mean"),
                "esmfold_plddt_delta_vs_parent": esmfold_item.get("esmfold_plddt_delta_vs_parent"),
                "esmfold_zero_shot_score": structure_score,
                "esmfold_low_confidence_fraction": esmfold_item.get("esmfold_low_confidence_fraction"),
                "predicted_retention72": adjusted,
                "thermostability_score": adjusted,
                "thermompnn_ddg_pred": ddg,
                "ddg_stability_score": ddg_score,
                "model": model,
                "warnings": warnings,
            }
        )

    if thermompnn_result is not None:
        for item in outputs:
            item["thermompnn_status"] = thermompnn_result["status"]
            item["thermompnn_model"] = thermompnn_result["model"]
    if rasp_result is not None:
        for item in outputs:
            item["rasp_status"] = rasp_result["status"]
            item["rasp_model"] = rasp_result["model"]
    return outputs


def predict_retention72_proxy_batch(
    variants: list[dict],
    competition_cfg: dict | None = None,
    stability_cfg: dict | None = None,
    parent_sequence: str | None = None,
    esmc_cfg: dict | None = None,
    progress_callback=None,
):
    return predict_zero_shot_stability_batch(
        variants=variants,
        competition_cfg=competition_cfg,
        stability_cfg=stability_cfg,
        parent_sequence=parent_sequence,
        esmc_cfg=esmc_cfg,
        progress_callback=progress_callback,
    )
