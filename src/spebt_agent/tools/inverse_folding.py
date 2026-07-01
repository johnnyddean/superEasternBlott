from __future__ import annotations

import subprocess
from pathlib import Path

from spebt_agent.paths import resolve_path
from spebt_agent.tools.external_support import diff_mutations, resolve_structure_path, resolve_tool_python


def _cli_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix()


def proteinmpnn_assets_present(repo_dir: str | Path, weights_dir: str | Path) -> dict[str, bool]:
    repo = Path(repo_dir)
    weights = Path(weights_dir)
    direct_ckpt = weights / "v_48_020.pt"
    nested_ckpt = weights / "vanilla_model_weights" / "v_48_020.pt"
    return {
        "repo": repo.exists(),
        "runner": (repo / "protein_mpnn_run.py").exists(),
        "weights_dir": weights.exists(),
        "v_48_020": direct_ckpt.exists() or nested_ckpt.exists(),
    }


def build_proteinmpnn_command(
    repo_dir: str | Path,
    pdb_path: str | Path,
    out_dir: str | Path,
    weights_dir: str | Path,
    model_name: str = "v_48_020",
    num_seq_per_target: int = 1,
    sampling_temp: str = "0.1",
) -> list[str]:
    runner = Path(repo_dir) / "protein_mpnn_run.py"
    return [
        "python",
        str(runner),
        "--pdb_path",
        _cli_path(pdb_path),
        "--out_folder",
        _cli_path(out_dir),
        "--path_to_model_weights",
        _cli_path(weights_dir),
        "--model_name",
        model_name,
        "--num_seq_per_target",
        str(num_seq_per_target),
        "--sampling_temp",
        sampling_temp,
        "--batch_size",
        "1",
        "--suppress_print",
        "1",
    ]


def _parse_proteinmpnn_fastas(out_dir: Path, parent_name: str, parent_sequence: str) -> list[dict]:
    """Parse ProteinMPNN FASTA outputs, repairing structural gaps in generated sequences.

    ProteinMPNN may produce sequences that:
    - Lack the initial M (the structure input may exclude flexible N-terminus)
    - Contain 'X' at positions where the backbone has no structural constraints (disordered loops)

    Both issues are repaired by falling back to the parent residue at those positions.
    """
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    parent_upper = parent_sequence.strip().upper()

    def _repair(seq: str) -> str:
        # Ensure uppercase
        seq = seq.upper()
        # Repair invalid positions (X or other unknown) from parent
        if len(seq) != len(parent_upper):
            # Try aligning: if seq is shorter, pad with parent suffix
            if len(seq) < len(parent_upper):
                seq = seq + parent_upper[len(seq):]
            else:
                seq = seq[:len(parent_upper)]
        chars = list(seq)
        for i in range(len(chars)):
            if chars[i] not in VALID_AA:
                if i < len(parent_upper):
                    chars[i] = parent_upper[i]
                else:
                    chars[i] = 'A'  # fallback
        seq = ''.join(chars)
        # Ensure starts with M
        if not seq.startswith('M'):
            if len(seq) >= 2:
                seq = 'M' + seq[1:]
            else:
                seq = 'M'
        return seq

    variants: list[dict] = []
    seq_dir = out_dir / "seqs"
    if not seq_dir.exists():
        return variants

    seen: set[str] = set()
    for fasta in sorted(seq_dir.glob("*.fa")):
        lines = [line.strip() for line in fasta.read_text(encoding="utf-8").splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if line.startswith(">"):
                continue
            raw = line.replace("/", "").replace(":", "").upper()
            sequence = _repair(raw)
            if sequence in seen or sequence == parent_upper:
                continue
            seen.add(sequence)
            mutations = diff_mutations(parent_upper, sequence)
            variants.append(
                {
                    "variant_id": f"proteinmpnn_{len(variants)+1}",
                    "parent": parent_name,
                    "sequence": sequence,
                    "mutations": mutations,
                    "num_mutations": len(mutations),
                    "source": "proteinmpnn",
                }
            )
    return variants


def run_proteinmpnn_adapter(
    parent_name: str,
    parent_sequence: str,
    parent_pdb: str | None,
    inverse_folding_cfg: dict,
    generation_cfg: dict,
) -> dict:
    repo_dir = resolve_path(inverse_folding_cfg["proteinmpnn_script"]).parent
    weights_dir = resolve_path(inverse_folding_cfg["proteinmpnn_weights_dir"])
    out_dir = resolve_path(inverse_folding_cfg["proteinmpnn_out_dir"])
    structure_path = resolve_structure_path(parent_pdb, inverse_folding_cfg.get("structure_search_dirs"))
    assets = proteinmpnn_assets_present(repo_dir, weights_dir)
    env_python = resolve_tool_python("inverse_folding")

    if not generation_cfg.get("enabled", False):
        return {
            "status": "skipped",
            "model": "proteinmpnn_disabled",
            "warnings": ["Inverse folding is disabled in generation.yaml."],
            "assets": assets,
        }
    if env_python is None:
        return {
            "status": "skipped",
            "model": "proteinmpnn_env_missing",
            "warnings": ["spebt_inverse_folding environment python was not found."],
            "assets": assets,
        }
    if structure_path is None:
        return {
            "status": "skipped",
            "model": "proteinmpnn_missing_pdb",
            "warnings": ["No structure PDB was found in configured search directories."],
            "assets": assets,
        }
    if not all(assets.values()):
        return {
            "status": "skipped",
            "model": "proteinmpnn_assets_missing",
            "warnings": ["ProteinMPNN repository or weights are incomplete."],
            "assets": assets,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_proteinmpnn_command(
        repo_dir=repo_dir,
        pdb_path=structure_path,
        out_dir=out_dir,
        weights_dir=weights_dir,
        num_seq_per_target=int(generation_cfg.get("num_seq_per_target", 1)),
        sampling_temp=" ".join(str(x) for x in generation_cfg.get("sampling_temps", [0.1])),
    )
    command[0] = str(env_python)
    proc = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    variants = _parse_proteinmpnn_fastas(out_dir, parent_name, parent_sequence) if proc.returncode == 0 else []
    status = "success" if proc.returncode == 0 else "failed"
    warnings = [] if proc.returncode == 0 else [proc.stderr.strip() or "ProteinMPNN execution failed."]
    if proc.returncode == 0 and not variants:
        status = "partial"
        warnings = ["ProteinMPNN finished but produced no usable variants."]
    return {
        "status": status,
        "model": "proteinmpnn_v_48_020",
        "warnings": warnings,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
        "returncode": proc.returncode,
        "pdb_path": str(structure_path),
        "command": command,
        "generated_count": len(variants),
        "variants": variants,
        "assets": assets,
    }
