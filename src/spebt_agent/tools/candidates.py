from __future__ import annotations

import itertools

AA = "ACDEFGHIKLMNPQRSTVWY"


def generate_single_mutants(parent_name: str, parent_seq: str, protected_positions: list[int], max_candidates: int = 5000):
    protected = set(protected_positions)
    variants = []
    for idx, old in enumerate(parent_seq, start=1):
        if idx in protected:
            continue
        for new in AA:
            if new == old:
                continue
            seq = parent_seq[: idx - 1] + new + parent_seq[idx:]
            mut = f"{old}{idx}{new}"
            variants.append(
                {
                    "variant_id": f"{parent_name}_{mut}",
                    "parent_id": parent_name,
                    "source": "single_mutation",
                    "sequence": seq,
                    "mutations": [mut],
                }
            )
            if len(variants) >= max_candidates:
                return variants
    return variants


def generate_combinatorial_mutants(parent_name: str, parent_seq: str, seed_variants: list[dict], max_order: int = 3, max_candidates: int = 5000):
    seeds = []
    seen_positions = set()
    for item in seed_variants:
        muts = item.get("mutations", [])
        if len(muts) != 1:
            continue
        mut = muts[0]
        pos = int(mut[1:-1])
        if pos in seen_positions:
            continue
        seeds.append(mut)
        seen_positions.add(pos)
    import random, math
    rng = random.Random(42)
    variants = []
    per_order = max(200, max_candidates // max_order)
    for order in range(2, max_order + 1):
        budget = min(per_order, max_candidates - len(variants))
        if budget <= 0:
            break
        # Use fewer seeds for higher orders to keep combos tractable
        max_seeds_for_order = {2: len(seeds), 3: len(seeds), 4: 50, 5: 40}.get(order, 30)
        order_seeds = seeds[:max_seeds_for_order] if len(seeds) > max_seeds_for_order else seeds
        total = math.comb(len(order_seeds), order)
        if total == 0:
            continue
        if total <= budget:
            combos = list(itertools.combinations(order_seeds, order))
        else:
            indices = rng.sample(range(total), budget)
            all_combos = itertools.combinations(order_seeds, order)
            combos = []
            targets = set(indices)
            max_idx = max(indices)
            for i, c in enumerate(all_combos):
                if i in targets:
                    combos.append(c)
                if i >= max_idx:
                    break
        for combo in combos:
            seq = list(parent_seq)
            valid = True
            for mut in combo:
                old, pos, new = mut[0], int(mut[1:-1]), mut[-1]
                if seq[pos - 1] != old:
                    valid = False
                    break
                seq[pos - 1] = new
            if not valid:
                continue
            mut_id = "_".join(combo)
            variants.append({
                "variant_id": f"{parent_name}_{mut_id}",
                "parent_id": parent_name,
                "source": "combinatorial_mutation",
                "sequence": "".join(seq),
                "mutations": list(combo),
                "num_mutations": len(combo),
            })
            if len(variants) >= max_candidates:
                return variants
    return variants
