AA = set("ACDEFGHIKLMNPQRSTVWY")


def hard_rule_filter(variants, min_length=220, max_length=250, must_start_with="M"):
    passed, failed = [], []
    for v in variants:
        seq = v["sequence"]
        reasons = []
        if not seq.startswith(must_start_with):
            reasons.append("not_start_with_required_residue")
        if not (min_length <= len(seq) <= max_length):
            reasons.append("invalid_length")
        if set(seq) - AA:
            reasons.append("invalid_amino_acid")
        if reasons:
            item = dict(v)
            item["filter_reasons"] = reasons
            failed.append(item)
        else:
            passed.append(v)
    return passed, failed


def load_exclusion_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and line.strip() != "Sequence"}


def exclusion_filter(variants, exclusion_set):
    passed, failed = [], []
    for v in variants:
        if v["sequence"] in exclusion_set:
            item = dict(v)
            item["filter_reasons"] = ["in_exclusion_list"]
            failed.append(item)
        else:
            passed.append(v)
    return passed, failed


def chromophore_risk_score(mutations, chromophore_positions, pocket_positions):
    mutated_positions = [int(m[1:-1]) for m in mutations]
    core_hits = sorted(set(mutated_positions) & set(chromophore_positions))
    pocket_hits = sorted(set(mutated_positions) & set(pocket_positions))
    risk = 0.0
    if core_hits:
        risk += 1.0
    if pocket_hits:
        risk += 0.3
    return {"chromophore_risk": min(risk, 1.0), "core_hits": core_hits, "pocket_hits": pocket_hits}
