def by_id(items):
    return {x["variant_id"]: x for x in items}


def predicted_competition_score(relative_brightness, retention72, threshold=0.30):
    if relative_brightness < threshold:
        return 0.0
    return relative_brightness * retention72


def _mutation_diversity_proxy(variant):
    num_mut = int(variant.get("num_mutations", len(variant.get("mutations", [])) or 0))
    return min(1.0, num_mut / 3.0)


def _stage_weights(scoring_cfg, stage):
    if not scoring_cfg:
        return {}
    return scoring_cfg.get(stage, {})


def rank_variants(variants, brightness_scores, stability_scores, risk_scores, threshold=0.30, scoring_cfg=None, stage="final"):
    bmap = by_id(brightness_scores)
    smap = by_id(stability_scores)
    rmap = by_id(risk_scores)
    stage_cfg = _stage_weights(scoring_cfg, stage)
    final_cfg = _stage_weights(scoring_cfg, "final")
    use_product = bool((scoring_cfg or {}).get("competition_score", {}).get("use_product_score", True))
    ranked = []
    for v in variants:
        vid = v["variant_id"]
        rb = float(bmap.get(vid, {}).get("predicted_relative_brightness", 0.0))
        rt = float(smap.get(vid, {}).get("predicted_retention72", 0.0))
        risk = float(rmap.get(vid, {}).get("chromophore_risk", 0.0))
        ddg_score = smap.get(vid, {}).get("ddg_stability_score")
        ddg_score = float(ddg_score) if ddg_score is not None else 0.5
        diversity_score = _mutation_diversity_proxy(v)
        product = predicted_competition_score(rb, rt, threshold=threshold)
        if stage == "stage1":
            weighted_score = (
                float(stage_cfg.get("brightness", 0.65)) * rb
                + float(stage_cfg.get("stability", 0.25)) * rt
            )
            final_score = weighted_score + 0.25 * product - float(stage_cfg.get("chromophore_risk_penalty", 0.40)) * risk
        else:
            weighted_score = (
                float(final_cfg.get("predicted_relative_brightness", 0.45)) * rb
                + float(final_cfg.get("predicted_retention72", 0.30)) * rt
                + float(final_cfg.get("ddg_stability", 0.10)) * ddg_score
                + float(final_cfg.get("diversity", 0.05)) * diversity_score
            )
            final_score = weighted_score - float(final_cfg.get("chromophore_risk_penalty", 0.45)) * risk
            if use_product:
                final_score += float(final_cfg.get("product_score", 0.30)) * product
        item = dict(v)
        item.update({k: val for k, val in bmap.get(vid, {}).items() if k != "variant_id"})
        stability_meta = {}
        for key, val in smap.get(vid, {}).items():
            if key == "variant_id":
                continue
            if key == "model":
                stability_meta["stability_model"] = val
                continue
            stability_meta[key] = val
        item.update(stability_meta)
        item.update({k: val for k, val in rmap.get(vid, {}).items() if k != "variant_id"})
        item.update(
            {
                "predicted_relative_brightness": rb,
                "predicted_retention72": rt,
                "ddg_stability_score": ddg_score,
                "diversity_score": diversity_score,
                "weighted_score": weighted_score,
                "predicted_competition_score": product,
                "chromophore_risk": risk,
                "final_score": final_score,
                "overall_score": final_score,
                "score": final_score,
            }
        )
        ranked.append(item)
    return sorted(ranked, key=lambda x: x["final_score"], reverse=True)


def hamming_distance(a, b):
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def select_top_with_diversity(ranked, max_n=6, min_hamming_distance=2):
    selected = []
    for item in ranked:
        seq = item["sequence"]
        if any(hamming_distance(seq, s["sequence"]) < min_hamming_distance for s in selected):
            continue
        selected.append(item)
        if len(selected) >= max_n:
            break
    return selected
