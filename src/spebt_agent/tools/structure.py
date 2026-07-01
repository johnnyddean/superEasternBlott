from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def esmfold_assets_present(model_dir: str | Path) -> dict[str, bool]:
    root = Path(model_dir)
    return {
        "root": root.exists(),
        "config": (root / "config.json").exists(),
        "weights": (root / "pytorch_model.bin").exists() or (root / "model.safetensors").exists(),
        "vocab": (root / "vocab.txt").exists() or (root / "tokenizer.json").exists(),
    }


def build_esmfold_status(model_dir: str | Path) -> dict:
    present = esmfold_assets_present(model_dir)
    return {
        "model": "esmfold_v1",
        "ready": all(present.values()),
        "assets": present,
        "warnings": [] if all(present.values()) else ["ESMFold assets are incomplete."],
    }


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.strip().upper().encode("utf-8")).hexdigest()[:16]


def _metrics_cache_path(cache_dir: Path, sequence: str) -> Path:
    return cache_dir / f"{_sequence_hash(sequence)}.metrics.json"


def _pdb_cache_path(cache_dir: Path, sequence: str) -> Path:
    return cache_dir / f"{_sequence_hash(sequence)}.pdb"


def robust_minmax(values, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = np.quantile(arr, q_low)
    hi = np.quantile(arr, q_high)
    if hi - lo < 1e-8:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_esmfold_model(model_dir: str | Path, device: str = "auto"):
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if resolved_device == "cuda":
        # On this RTX 4060 setup, fp16 was numerically unstable while fp32 exceeded VRAM.
        # Prefer bf16 on GPU when available as the most stable memory-feasible path.
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32
    model = EsmForProteinFolding.from_pretrained(
        str(model_dir),
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    ).eval().to(resolved_device)
    return tokenizer, model, resolved_device


def _residue_plddt(outputs) -> np.ndarray:
    atom_exists = outputs.atom37_atom_exists.detach().cpu().numpy().astype(np.float32)
    plddt = outputs.plddt.detach().to(dtype=__import__("torch").float32).cpu().numpy().astype(np.float32)
    atom_counts = np.clip(atom_exists.sum(axis=-1), 1.0, None)
    residue_plddt = (plddt * atom_exists).sum(axis=-1) / atom_counts
    residue_plddt *= 100.0
    return residue_plddt


def _infer_metrics_for_sequences(
    sequences: list[str],
    model_dir: str | Path,
    cache_dir: str | Path,
    device: str = "auto",
    batch_size: int = 1,
    allow_cpu_fallback: bool = True,
) -> dict[str, dict[str, Any]]:
    import torch

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    metrics_by_sequence: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for sequence in sequences:
        metrics_path = _metrics_cache_path(cache_root, sequence)
        if metrics_path.exists():
            metrics_by_sequence[sequence] = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            pending.append(sequence)

    if not pending:
        return metrics_by_sequence

    tokenizer, model, resolved_device = _load_esmfold_model(model_dir, device=device)
    try:
        with torch.inference_mode():
            for start in range(0, len(pending), batch_size):
                batch_sequences = pending[start : start + batch_size]
                inputs = tokenizer(batch_sequences, return_tensors="pt", add_special_tokens=False, padding=True)
                inputs = {k: v.to(resolved_device) for k, v in inputs.items()}
                outputs = model(**inputs)
                residue_plddt = _residue_plddt(outputs)
                pdb_ready_outputs = {}
                for key, value in outputs.items():
                    if torch.is_tensor(value) and value.is_floating_point():
                        pdb_ready_outputs[key] = value.to(dtype=torch.float32)
                    else:
                        pdb_ready_outputs[key] = value
                pdbs = model.output_to_pdb(pdb_ready_outputs)
                for idx, sequence in enumerate(batch_sequences):
                    seq_len = len(sequence)
                    residue_scores = residue_plddt[idx][:seq_len]
                    metrics = {
                        "sequence_hash": _sequence_hash(sequence),
                        "sequence_length": seq_len,
                        "esmfold_plddt_mean": float(np.mean(residue_scores)),
                        "esmfold_low_confidence_fraction": float(np.mean(residue_scores < 70.0)),
                    }
                    metrics_by_sequence[sequence] = metrics
                    _metrics_cache_path(cache_root, sequence).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                    _pdb_cache_path(cache_root, sequence).write_text(pdbs[idx], encoding="utf-8")
    except RuntimeError as exc:
        oom_markers = ("out of memory", "cuda", "cublas")
        if allow_cpu_fallback and resolved_device != "cpu" and any(marker in str(exc).lower() for marker in oom_markers):
            return _infer_metrics_for_sequences(
                sequences=sequences,
                model_dir=model_dir,
                cache_dir=cache_dir,
                device="cpu",
                batch_size=1,
                allow_cpu_fallback=False,
            )
        raise
    return metrics_by_sequence


def score_sequences_esmfold_zero_shot(
    sequences: list[str],
    model_dir: str | Path,
    cache_dir: str | Path,
    parent_sequence: str | None = None,
    device: str = "auto",
    batch_size: int = 1,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    status = build_esmfold_status(model_dir)
    if not sequences:
        return []
    selected_sequences = list(sequences[:top_k]) if top_k is not None else list(sequences)
    if not status["ready"]:
        return [
            {
                "sequence": sequence,
                "esmfold_plddt_mean": None,
                "esmfold_plddt_delta_vs_parent": None,
                "esmfold_zero_shot_score": None,
                "esmfold_low_confidence_fraction": None,
                "esmfold_status": "skipped",
                "esmfold_warning": "ESMFold assets are incomplete.",
            }
            for sequence in selected_sequences
        ]

    unique_sequences = list(dict.fromkeys(selected_sequences + ([parent_sequence] if parent_sequence else [])))
    metrics_by_sequence = _infer_metrics_for_sequences(
        unique_sequences,
        model_dir=model_dir,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
    )
    parent_metrics = metrics_by_sequence.get(parent_sequence) if parent_sequence else None
    parent_plddt = float(parent_metrics["esmfold_plddt_mean"]) if parent_metrics is not None else None

    plddt_values = np.asarray([float(metrics_by_sequence[sequence]["esmfold_plddt_mean"]) for sequence in selected_sequences], dtype=np.float32)
    plddt_norm = robust_minmax(plddt_values)

    outputs: list[dict[str, Any]] = []
    for sequence, plddt_mean, plddt_score_norm in zip(selected_sequences, plddt_values, plddt_norm):
        metrics = metrics_by_sequence[sequence]
        delta = float(plddt_mean - parent_plddt) if parent_plddt is not None else float(plddt_mean)
        outputs.append(
            {
                "sequence": sequence,
                "esmfold_plddt_mean": float(plddt_mean),
                "esmfold_plddt_delta_vs_parent": delta,
                "esmfold_zero_shot_score": float(plddt_score_norm),
                "esmfold_low_confidence_fraction": float(metrics["esmfold_low_confidence_fraction"]),
                "esmfold_status": "success",
                "esmfold_warning": "",
            }
        )
    return outputs


def run_structure_prediction_adapter(*args, **kwargs):
    return {
        "status": "skipped",
        "model": "esmfold_reserved_adapter",
        "warnings": ["Structure prediction execution is reserved for the dedicated structure environment."],
    }
