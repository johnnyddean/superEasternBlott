from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from spebt_agent.config import load_default_configs
from spebt_agent.paths import project_root, resolve_path
from spebt_agent.tools.brightness import build_feature_matrix
from spebt_agent.tools.esmc import embed_sequences_real


def _unique_sequence_embeddings(train: pd.DataFrame, valid: pd.DataFrame, esmc_cfg: dict) -> dict[str, np.ndarray]:
    seqs = pd.concat([train["sequence"], valid["sequence"]], ignore_index=True).astype(str).drop_duplicates().tolist()
    embeddings = embed_sequences_real(
        sequences=seqs,
        model_dir=resolve_path(esmc_cfg["model_dir"]),
        model_name=esmc_cfg["model_name"],
        cache_dir=resolve_path(esmc_cfg["embedding_cache_dir"]),
        pooling=esmc_cfg.get("pooling", "mean"),
        batch_size=int(esmc_cfg.get("batch_size", 8)),
        device=esmc_cfg.get("device", "auto"),
        dtype=esmc_cfg.get("dtype", "float32"),
    )
    return {seq: emb for seq, emb in zip(seqs, embeddings)}


def _rows_to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "sequence": str(row["sequence"]),
            "num_mutations": int(row["num_mutations"]),
            "parent": str(row["parent"]),
        }
        for _, row in df.iterrows()
    ]


def _features_for_dataframe(df: pd.DataFrame, seq_to_embedding: dict[str, np.ndarray], parent_categories: list[str]) -> np.ndarray:
    records = _rows_to_records(df)
    embeddings = np.vstack([seq_to_embedding[str(row["sequence"])] for _, row in df.iterrows()]).astype(np.float32)
    return build_feature_matrix(records, embeddings, parent_categories)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/brightness_train.csv")
    parser.add_argument("--valid", default="data/processed/brightness_valid.csv")
    parser.add_argument("--target", default="brightness")
    parser.add_argument("--out", default=None)
    parser.add_argument("--metrics-out", default="artifacts/reports/brightness_training_metrics.json")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    args = parser.parse_args()

    configs = load_default_configs()
    esmc_cfg = configs["model_paths"]["esmc"]
    brightness_cfg = configs["model_paths"]["brightness"]
    default_out = brightness_cfg["abs_model"] if args.target == "brightness" else brightness_cfg["delta_model"]
    out = Path(resolve_path(args.out or default_out))
    metrics_out = Path(resolve_path(args.metrics_out))

    print(f"[train_brightness] loading train={args.train} valid={args.valid} target={args.target}")
    train = pd.read_csv(resolve_path(args.train))
    valid = pd.read_csv(resolve_path(args.valid))
    if args.max_train_samples:
        train = train.head(args.max_train_samples).copy()
    if args.max_valid_samples:
        valid = valid.head(args.max_valid_samples).copy()

    print(f"[train_brightness] train_rows={len(train)} valid_rows={len(valid)}")
    parent_categories = sorted(train["parent"].astype(str).unique().tolist())
    print(f"[train_brightness] computing ESMC embeddings for {pd.concat([train['sequence'], valid['sequence']], ignore_index=True).nunique()} unique sequences")
    seq_to_embedding = _unique_sequence_embeddings(train, valid, esmc_cfg)
    print("[train_brightness] building feature matrices")
    x_train = _features_for_dataframe(train, seq_to_embedding, parent_categories)
    x_valid = _features_for_dataframe(valid, seq_to_embedding, parent_categories)
    y_train = train[args.target].astype(float).to_numpy()
    y_valid = valid[args.target].astype(float).to_numpy()

    estimator = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=42,
    )
    print(f"[train_brightness] fitting model feature_dim={x_train.shape[1]}")
    estimator.fit(x_train, y_train)
    pred = estimator.predict(x_valid)

    bundle = {
        "estimator": estimator,
        "target": args.target,
        "parent_categories": parent_categories,
        "feature_dim": int(x_train.shape[1]),
        "esmc": {
            "model_name": esmc_cfg["model_name"],
            "model_dir": str(resolve_path(esmc_cfg["model_dir"])),
            "embedding_cache_dir": str(resolve_path(esmc_cfg["embedding_cache_dir"])),
            "pooling": esmc_cfg.get("pooling", "mean"),
            "batch_size": int(esmc_cfg.get("batch_size", 8)),
            "device": esmc_cfg.get("device", "auto"),
            "dtype": esmc_cfg.get("dtype", "float32"),
            "embedding_dim": int(esmc_cfg.get("embedding_dim", x_train.shape[1] - 2 - len(parent_categories))),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"[train_brightness] saved model to {out}")

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if metrics_out.exists():
        metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    metrics[args.target] = {
        "out": str(out),
        "r2": float(r2_score(y_valid, pred)),
        "mae": float(mean_absolute_error(y_valid, pred)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "feature_dim": int(x_train.shape[1]),
        "parent_categories": parent_categories,
    }
    metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[train_brightness] wrote metrics to {metrics_out}")
    print(metrics[args.target])


if __name__ == "__main__":
    main()
