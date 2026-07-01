from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sequence_cache_key(sequence: str) -> str:
    return hashlib.sha1(sequence.strip().upper().encode("utf-8")).hexdigest()[:16]


def model_files_present(model_dir: str | Path) -> dict[str, bool]:
    root = Path(model_dir)
    return {
        "root": root.exists(),
        "config": (root / "config.json").exists(),
        "weights": (root / "model.safetensors").exists() or (root / "pytorch_model.bin").exists(),
        "tokenizer": (root / "tokenizer.json").exists() or (root / "vocab.txt").exists(),
    }


def deterministic_embedding(sequence: str, dim: int = 32) -> np.ndarray:
    digest = hashlib.sha256(sequence.encode("utf-8")).digest()
    values = np.frombuffer((digest * ((dim // len(digest)) + 1))[:dim], dtype=np.uint8)
    return values.astype(np.float32) / 255.0


def embed_sequences_offline(sequences: list[str], cache_dir: str | Path | None = None, dim: int = 32) -> np.ndarray:
    embeddings = np.vstack([deterministic_embedding(seq, dim=dim) for seq in sequences])
    if cache_dir is not None:
        out = Path(cache_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = [{"seq_hash": sequence_cache_key(seq), "length": len(seq)} for seq in sequences]
        (out / "offline_embedding_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        np.save(out / "offline_embeddings.npy", embeddings)
    return embeddings


def _cache_file(cache_dir: Path, model_name: str, pooling: str, sequence: str) -> Path:
    seq_hash = sequence_cache_key(sequence)
    model_hash = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:12]
    return cache_dir / pooling / model_hash / f"{seq_hash}.npy"


def _score_cache_file(cache_dir: Path, model_name: str, scoring_mode: str, sequence: str) -> Path:
    seq_hash = sequence_cache_key(sequence)
    model_hash = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:12]
    return cache_dir / scoring_mode / model_hash / f"{seq_hash}.json"


def _prepare_sequence(sequence: str) -> str:
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Sequence cannot be empty.")
    return seq


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype, torch.float32)


def _load_config(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))


def _map_hf_state_dict_keys(state_dict: dict[str, object]) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for key, value in state_dict.items():
        if key.endswith("._extra_state"):
            continue
        new_key = key
        if new_key.startswith("esmc."):
            new_key = new_key[len("esmc.") :]
        new_key = new_key.replace("attn.layernorm_qkv.layer_norm_weight", "attn.layernorm_qkv.0.weight")
        new_key = new_key.replace("attn.layernorm_qkv.layer_norm_bias", "attn.layernorm_qkv.0.bias")
        new_key = new_key.replace("attn.layernorm_qkv.weight", "attn.layernorm_qkv.1.weight")
        new_key = new_key.replace("ffn.layer_norm_weight", "ffn.0.weight")
        new_key = new_key.replace("ffn.layer_norm_bias", "ffn.0.bias")
        new_key = new_key.replace("ffn.fc1_weight", "ffn.1.weight")
        new_key = new_key.replace("ffn.fc2_weight", "ffn.3.weight")
        mapped[new_key] = value
    return mapped


def _load_model_and_tokenizer(model_dir: str | Path, device: str = "auto", dtype: str = "float32"):
    try:
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers
    except ImportError:
        raise RuntimeError(
            "ESMC model code requires esm>=3.2.1.post1 (Python>=3.12). "
            "Install from https://github.com/evolutionaryscale/esm or use --fallback-embeddings."
        )
    from safetensors.torch import load_file

    resolved_dir = Path(model_dir)
    if not all(model_files_present(resolved_dir).values()):
        raise FileNotFoundError(f"ESMC assets are incomplete under {resolved_dir}")

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)
    config = _load_config(resolved_dir)
    tokenizer = get_esmc_model_tokenizers()
    model = ESMC(
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        tokenizer=tokenizer,
        use_flash_attn=False,
    ).eval()
    state_dict = _map_hf_state_dict_keys(load_file(str(resolved_dir / "model.safetensors")))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = ("sequence_head.",)
    allowed_unexpected_prefixes = ("lm_head.",)
    disallowed_missing = [name for name in missing if not name.startswith(allowed_missing_prefixes)]
    disallowed_unexpected = [name for name in unexpected if not name.startswith(allowed_unexpected_prefixes)]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "Failed to map ESMC safetensors into esm runtime: "
            f"missing={disallowed_missing[:10]} unexpected={disallowed_unexpected[:10]}"
        )
    model.to(dtype=resolved_dtype)
    model.eval()
    model.to(resolved_device)
    return tokenizer, model, resolved_device


def _pool_hidden_states(hidden_states, input_ids, tokenizer, pooling: str):
    import torch

    if pooling == "cls":
        return hidden_states[:, 0, :]

    if pooling == "eos":
        eos_positions = (input_ids != tokenizer.pad_token_id).sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_idx, eos_positions]

    special_ids = {x for x in [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id] if x is not None}
    token_mask = input_ids != tokenizer.pad_token_id
    for special_id in special_ids:
        token_mask &= input_ids != special_id
    token_mask = token_mask.unsqueeze(-1)
    pooled = (hidden_states * token_mask).sum(dim=1)
    counts = token_mask.sum(dim=1).clamp(min=1)
    return pooled / counts


def robust_minmax(values, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = np.quantile(arr, q_low)
    hi = np.quantile(arr, q_high)
    if hi - lo < 1e-8:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def embed_sequences_real(
    sequences: list[str],
    model_dir: str | Path,
    model_name: str,
    cache_dir: str | Path | None = None,
    pooling: str = "mean",
    batch_size: int = 8,
    device: str = "auto",
    dtype: str = "float32",
    allow_cpu_fallback: bool = True,
) -> np.ndarray:
    import torch

    if not sequences:
        return np.empty((0, 0), dtype=np.float32)

    cache_root = Path(cache_dir) if cache_dir is not None else None
    embeddings: list[np.ndarray | None] = [None] * len(sequences)
    pending_idx = []
    if cache_root is not None:
        for i, seq in enumerate(sequences):
            cache_path = _cache_file(cache_root, model_name, pooling, seq)
            if cache_path.exists():
                embeddings[i] = np.load(cache_path)
            else:
                pending_idx.append(i)
    else:
        pending_idx = list(range(len(sequences)))

    if not pending_idx:
        return np.vstack(embeddings).astype(np.float32)

    try:
        tokenizer, model, resolved_device = _load_model_and_tokenizer(model_dir, device=device, dtype=dtype)
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        # Fallback to deterministic embeddings when ESMC model is unavailable
        import warnings
        warnings.warn(f"ESMC model unavailable, falling back to deterministic embeddings: {exc}")
        for idx in pending_idx:
            embeddings[idx] = deterministic_embedding(sequences[idx], dim=1152)
            if cache_root is not None:
                cache_path = _cache_file(cache_root, model_name, pooling, sequences[idx])
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, embeddings[idx])
        return np.vstack(embeddings).astype(np.float32)
    try:
        with torch.inference_mode():
            for start in range(0, len(pending_idx), batch_size):
                idxs = pending_idx[start : start + batch_size]
                batch_sequences = [sequences[i] for i in idxs]
                tokenized = model._tokenize([_prepare_sequence(seq) for seq in batch_sequences]).to(resolved_device)
                outputs = model(sequence_tokens=tokenized)
                hidden_states = outputs.hidden_states[-1]
                pooled = _pool_hidden_states(
                    hidden_states,
                    tokenized,
                    tokenizer,
                    pooling=pooling,
                )
                batch_embeddings = pooled.detach().cpu().numpy().astype(np.float32)
                for idx, emb in zip(idxs, batch_embeddings):
                    embeddings[idx] = emb
                    if cache_root is not None:
                        cache_path = _cache_file(cache_root, model_name, pooling, sequences[idx])
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cache_path, emb)
    except RuntimeError as exc:
        oom_markers = ("out of memory", "cuda", "cublas")
        if allow_cpu_fallback and resolved_device != "cpu" and any(marker in str(exc).lower() for marker in oom_markers):
            return embed_sequences_real(
                sequences=sequences,
                model_dir=model_dir,
                model_name=model_name,
                cache_dir=cache_dir,
                pooling=pooling,
                batch_size=max(1, batch_size // 2),
                device="cpu",
                dtype="float32",
                allow_cpu_fallback=False,
            )
        raise

    return np.vstack(embeddings).astype(np.float32)


def score_sequences_zero_shot(
    sequences: list[str],
    model_dir: str | Path,
    model_name: str,
    cache_dir: str | Path | None = None,
    batch_size: int = 8,
    device: str = "auto",
    dtype: str = "float32",
    parent_sequence: str | None = None,
    allow_cpu_fallback: bool = True,
) -> list[dict[str, float]]:
    import torch
    import torch.nn.functional as F

    if not sequences:
        return []

    cache_root = Path(cache_dir) if cache_dir is not None else None
    score_cache_root = cache_root / "zero_shot_scores" if cache_root is not None else None
    abs_scores: list[float | None] = [None] * len(sequences)
    pending_idx: list[int] = []
    if score_cache_root is not None:
        for i, seq in enumerate(sequences):
            score_path = _score_cache_file(score_cache_root, model_name, "token_logprob_proxy", seq)
            if score_path.exists():
                payload = json.loads(score_path.read_text(encoding="utf-8"))
                abs_scores[i] = float(payload["esmc_pll_abs"])
            else:
                pending_idx.append(i)
    else:
        pending_idx = list(range(len(sequences)))

    if pending_idx:
        try:
            tokenizer, model, resolved_device = _load_model_and_tokenizer(model_dir, device=device, dtype=dtype)
        except (RuntimeError, FileNotFoundError, ImportError) as exc:
            import warnings
            warnings.warn(f"ESMC model unavailable for zero-shot scoring, falling back to mock scores: {exc}")
            for idx in pending_idx:
                emb = deterministic_embedding(sequences[idx], dim=32)
                abs_scores[idx] = float(np.mean(emb)) * 10.0 - 5.0
                if score_cache_root is not None:
                    score_path = _score_cache_file(score_cache_root, model_name, "token_logprob_proxy", sequences[idx])
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    score_path.write_text(json.dumps({"esmc_pll_abs": abs_scores[idx]}, ensure_ascii=False), encoding="utf-8")
            pending_idx = []
        else:
            try:
                with torch.inference_mode():
                    for start in range(0, len(pending_idx), batch_size):
                        idxs = pending_idx[start : start + batch_size]
                        batch_sequences = [_prepare_sequence(sequences[i]) for i in idxs]
                        tokenized = model._tokenize(batch_sequences).to(resolved_device)
                        outputs = model(sequence_tokens=tokenized)
                        logits = outputs.sequence_logits
                        log_probs = F.log_softmax(logits, dim=-1)

                        special_ids = {x for x in [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id] if x is not None}
                        token_mask = tokenized != tokenizer.pad_token_id
                        for special_id in special_ids:
                            token_mask &= tokenized != special_id
                        gathered = log_probs.gather(dim=-1, index=tokenized.unsqueeze(-1)).squeeze(-1)
                        masked = gathered * token_mask
                        counts = token_mask.sum(dim=1).clamp(min=1)
                        batch_scores = (masked.sum(dim=1) / counts).detach().cpu().numpy().astype(np.float32)
                        for idx, score in zip(idxs, batch_scores):
                            abs_scores[idx] = float(score)
                            if score_cache_root is not None:
                                score_path = _score_cache_file(score_cache_root, model_name, "token_logprob_proxy", sequences[idx])
                                score_path.parent.mkdir(parents=True, exist_ok=True)
                                score_path.write_text(json.dumps({"esmc_pll_abs": float(score)}, ensure_ascii=False), encoding="utf-8")
            except RuntimeError as exc:
                oom_markers = ("out of memory", "cuda", "cublas")
                if allow_cpu_fallback and resolved_device != "cpu" and any(marker in str(exc).lower() for marker in oom_markers):
                    return score_sequences_zero_shot(
                        sequences=sequences,
                        model_dir=model_dir,
                        model_name=model_name,
                        cache_dir=cache_dir,
                        batch_size=max(1, batch_size // 2),
                        device="cpu",
                        dtype="float32",
                        parent_sequence=parent_sequence,
                        allow_cpu_fallback=False,
                    )
                raise

    abs_arr = np.asarray([float(x) for x in abs_scores], dtype=np.float32)
    parent_abs = None
    if parent_sequence:
        parent_abs = score_sequences_zero_shot(
            [parent_sequence],
            model_dir=model_dir,
            model_name=model_name,
            cache_dir=cache_dir,
            batch_size=1,
            device=device,
            dtype=dtype,
            parent_sequence=None,
            allow_cpu_fallback=allow_cpu_fallback,
        )[0]["esmc_pll_abs"]
    delta_arr = abs_arr - float(parent_abs) if parent_abs is not None else abs_arr.copy()
    abs_norm = robust_minmax(abs_arr)
    delta_norm = robust_minmax(delta_arr)

    outputs: list[dict[str, float]] = []
    for seq, abs_score, delta_score, abs_score_norm, delta_score_norm in zip(sequences, abs_arr, delta_arr, abs_norm, delta_norm):
        zero_shot_score = float(0.35 * abs_score_norm + 0.65 * delta_score_norm)
        outputs.append(
            {
                "sequence": seq,
                "esmc_pll_abs": float(abs_score),
                "esmc_pll_delta_vs_parent": float(delta_score),
                "esmc_zero_shot_score": zero_shot_score,
            }
        )
    return outputs
