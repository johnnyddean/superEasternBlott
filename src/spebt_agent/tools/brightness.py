from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from spebt_agent.tools.esmc import embed_sequences_real


def _simple_features(seq: str, n_mut: int) -> list[float]:
    length = len(seq)
    aromatic = sum(seq.count(x) for x in "FWY") / length
    charged = sum(seq.count(x) for x in "DEKR") / length
    gly_pro = sum(seq.count(x) for x in "GP") / length
    cys = seq.count("C") / length
    return [length, n_mut, aromatic, charged, gly_pro, cys]


def baseline_brightness_score(variant: dict) -> float:
    seq = variant["sequence"]
    n_mut = len(variant.get("mutations", []))
    burden = float(np.exp(-max(0, n_mut - 3) / 5.0))
    invalid_cys_penalty = 0.05 * max(0, seq.count("C") - 4)
    return float(np.clip(0.72 * burden - invalid_cys_penalty + 0.18, 0.0, 1.0))


def robust_minmax(values, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = np.quantile(arr, q_low)
    hi = np.quantile(arr, q_high)
    if hi - lo < 1e-8:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _get_parent_name(record: dict[str, Any]) -> str:
    return str(record.get("parent") or record.get("parent_id") or "unknown")


def build_feature_matrix(records: list[dict[str, Any]], embeddings: np.ndarray, parent_categories: list[str]) -> np.ndarray:
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
    index = {name: i for i, name in enumerate(parent_categories)}
    for row_idx, record in enumerate(records):
        parent_name = _get_parent_name(record)
        if parent_name in index:
            parent_one_hot[row_idx, index[parent_name]] = 1.0
    return np.hstack([embeddings.astype(np.float32), numeric, parent_one_hot])


def _load_bundle(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    bundle_path = Path(path)
    if not bundle_path.exists():
        return None
    loaded = joblib.load(bundle_path)
    if isinstance(loaded, dict) and "estimator" in loaded:
        return loaded
    return {"estimator": loaded, "parent_categories": [], "esmc": {}, "target": "unknown"}


def predict_brightness_batch(
    variants: list[dict[str, Any]],
    abs_model_path: str | Path | None = None,
    delta_model_path: str | Path | None = None,
    esmc_cfg: dict[str, Any] | None = None,
    progress_callback=None,
):
    abs_bundle = _load_bundle(abs_model_path)
    delta_bundle = _load_bundle(delta_model_path)
    if not variants:
        return []

    if abs_bundle is None and delta_bundle is None:
        return [
            {
                "variant_id": v["variant_id"],
                "predicted_relative_brightness": baseline_brightness_score(v),
                "brightness_score": baseline_brightness_score(v),
                "model": "baseline_brightness_proxy_v1",
            }
            for v in variants
        ]

    if progress_callback is not None:
        progress_callback("brightness_embeddings_start", variant_count=len(variants))
    shared_bundle = abs_bundle or delta_bundle or {}
    parent_categories = list(shared_bundle.get("parent_categories", []))
    model_cfg = {**shared_bundle.get("esmc", {}), **(esmc_cfg or {})}
    sequences = [v["sequence"] for v in variants]
    embeddings = embed_sequences_real(
        sequences=sequences,
        model_dir=model_cfg["model_dir"],
        model_name=model_cfg["model_name"],
        cache_dir=model_cfg.get("embedding_cache_dir"),
        pooling=model_cfg.get("pooling", "mean"),
        batch_size=int(model_cfg.get("batch_size", 8)),
        device=model_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "float32"),
    )
    if progress_callback is not None:
        progress_callback("brightness_embeddings_done", embedding_count=len(sequences), feature_dim=int(embeddings.shape[1]))
    features = build_feature_matrix(variants, embeddings, parent_categories)

    abs_pred = None
    delta_pred = None
    if abs_bundle is not None:
        if progress_callback is not None:
            progress_callback("brightness_abs_predict_start", variant_count=len(variants))
        abs_pred = np.asarray(abs_bundle["estimator"].predict(features), dtype=np.float32)
        if progress_callback is not None:
            progress_callback("brightness_abs_predict_done", variant_count=len(variants))
    if delta_bundle is not None:
        if progress_callback is not None:
            progress_callback("brightness_delta_predict_start", variant_count=len(variants))
        delta_pred = np.asarray(delta_bundle["estimator"].predict(features), dtype=np.float32)
        if progress_callback is not None:
            progress_callback("brightness_delta_predict_done", variant_count=len(variants))

    if abs_pred is None:
        abs_pred = delta_pred.copy()
    if delta_pred is None:
        delta_pred = abs_pred.copy()

    abs_norm = robust_minmax(abs_pred)
    delta_norm = robust_minmax(delta_pred)

    outputs = []
    for v, a, d, an, dn in zip(variants, abs_pred, delta_pred, abs_norm, delta_norm):
        score = float(0.4 * an + 0.6 * dn)
        outputs.append(
            {
                "variant_id": v["variant_id"],
                "brightness_abs_pred": float(a),
                "brightness_delta_pred": float(d),
                "predicted_relative_brightness": score,
                "brightness_score": score,
                "model": "ESMC_tree_abs_delta_ensemble",
            }
        )
    return outputs
