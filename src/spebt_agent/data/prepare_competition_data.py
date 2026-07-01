from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

AA = set("ACDEFGHIKLMNPQRSTVWY")
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]


def read_reference_sequences(path: str | Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    current: str | None = None
    seq: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                refs[current] = "".join(seq)
            current = line[1:].strip().split()[0]
            seq = []
        elif not line.startswith("#"):
            seq.append(line)
    if current is not None:
        refs[current] = "".join(seq)
    return refs


def parse_mutations(mutation_string: str):
    if pd.isna(mutation_string) or str(mutation_string).strip() == "WT":
        return []
    mutations = []
    for part in str(mutation_string).split(":"):
        m = MUT_RE.match(part)
        if m is None:
            return None
        old_aa, raw_pos, new_aa = m.group(1), int(m.group(2)), m.group(3)
        if old_aa not in AA or new_aa not in AA:
            return None
        mutations.append((old_aa, raw_pos, new_aa))
    return mutations


def apply_mutations_with_offset(parent_seq: str, mutations):
    seq = list(parent_seq)
    for old_aa, raw_pos, new_aa in mutations:
        real_pos = raw_pos + 1
        idx = real_pos - 1
        if idx < 0 or idx >= len(seq):
            return None
        if seq[idx] != old_aa:
            return None
        seq[idx] = new_aa
    return "".join(seq)


def prepare_data(refs_path: str | Path, gfp_xlsx_path: str | Path, exclusion_csv_path: str | Path, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    refs = read_reference_sequences(refs_path)
    pd.DataFrame(
        [{"name": k, "sequence": v, "length": len(v), "seq_hash": seq_hash(v)} for k, v in refs.items()]
    ).to_csv(out / "reference_parents.csv", index=False)

    brightness = pd.read_excel(gfp_xlsx_path, sheet_name="brightness")
    beforetop = pd.read_excel(gfp_xlsx_path, sheet_name="beforetopseqs")
    wt_brightness = (
        brightness[brightness["aaMutations"].astype(str) == "WT"]
        .set_index("GFP type")["Brightness"]
        .to_dict()
    )

    rows, dropped = [], []
    for i, row in brightness.iterrows():
        parent = row["GFP type"]
        if parent not in refs:
            dropped.append({"row": i, "reason": "unknown_parent", "aaMutations": row["aaMutations"]})
            continue
        muts = parse_mutations(row["aaMutations"])
        if muts is None:
            dropped.append({"row": i, "reason": "parse_fail", "aaMutations": row["aaMutations"]})
            continue
        seq = apply_mutations_with_offset(refs[parent], muts)
        if seq is None:
            dropped.append({"row": i, "reason": "mismatch", "aaMutations": row["aaMutations"]})
            continue
        b = float(row["Brightness"])
        wt_b = float(wt_brightness[parent])
        rows.append(
            {
                "variant_id": f"{parent}_{i}",
                "parent": parent,
                "aaMutations_raw": row["aaMutations"],
                "sequence": seq,
                "length": len(seq),
                "num_mutations": len(muts),
                "brightness": b,
                "parent_wt_brightness": wt_b,
                "delta_brightness_vs_parent": b - wt_b,
                "seq_hash": seq_hash(seq),
            }
        )

    recon = pd.DataFrame(rows)
    recon.to_csv(out / "brightness_reconstructed.csv", index=False)
    pd.DataFrame(dropped).to_csv(out / "brightness_dropped_rows.csv", index=False)
    if len(recon) and recon["parent"].nunique() > 1:
        train, valid = train_test_split(recon, test_size=0.15, random_state=42, stratify=recon["parent"])
    else:
        train, valid = train_test_split(recon, test_size=0.15, random_state=42)
    train.to_csv(out / "brightness_train.csv", index=False)
    valid.to_csv(out / "brightness_valid.csv", index=False)

    if "sequence" in beforetop.columns:
        beforetop["length"] = beforetop["sequence"].astype(str).str.len()
        beforetop["seq_hash"] = beforetop["sequence"].astype(str).apply(seq_hash)
    beforetop.to_csv(out / "beforetopseqs.csv", index=False)

    exclusion = pd.read_csv(exclusion_csv_path)
    exclusion["Sequence"].astype(str).str.strip().drop_duplicates().to_csv(
        out / "exclusion_sequences.txt", index=False, header=False
    )

    summary = {
        "n_reference_parents": len(refs),
        "reference_lengths": {k: len(v) for k, v in refs.items()},
        "n_brightness_raw": int(len(brightness)),
        "n_brightness_reconstructed": int(len(recon)),
        "n_brightness_dropped": int(len(dropped)),
        "n_beforetop": int(len(beforetop)),
        "n_exclusion": int(len(exclusion)),
        "wt_brightness": {k: float(v) for k, v in wt_brightness.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
