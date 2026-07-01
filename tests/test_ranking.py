from spebt_agent.tools.ranking import rank_variants, select_top_with_diversity


def test_rank_and_diversity():
    variants = [
        {"variant_id": "a", "sequence": "MAAA"},
        {"variant_id": "b", "sequence": "MAAT"},
        {"variant_id": "c", "sequence": "MTTT"},
    ]
    brightness = [{"variant_id": v["variant_id"], "predicted_relative_brightness": 0.8} for v in variants]
    stability = [{"variant_id": v["variant_id"], "predicted_retention72": 0.9} for v in variants]
    risk = [{"variant_id": v["variant_id"], "chromophore_risk": 0.0} for v in variants]
    ranked = rank_variants(variants, brightness, stability, risk)
    selected = select_top_with_diversity(ranked, max_n=2, min_hamming_distance=2)
    assert len(selected) == 2


def test_rank_variants_preserves_brightness_metadata():
    variants = [{"variant_id": "a", "sequence": "MAAA"}]
    brightness = [
        {
            "variant_id": "a",
            "predicted_relative_brightness": 0.8,
            "brightness_abs_pred": 1.2,
            "brightness_delta_pred": 0.3,
            "model": "ESMC_tree_abs_delta_ensemble",
        }
    ]
    stability = [{"variant_id": "a", "predicted_retention72": 0.9, "model": "rule_based_retention72_proxy_v1"}]
    risk = [{"variant_id": "a", "chromophore_risk": 0.0}]
    ranked = rank_variants(variants, brightness, stability, risk)
    assert ranked[0]["model"] == "ESMC_tree_abs_delta_ensemble"
    assert ranked[0]["stability_model"] == "rule_based_retention72_proxy_v1"
    assert ranked[0]["brightness_abs_pred"] == 1.2
    assert ranked[0]["brightness_delta_pred"] == 0.3


def test_rank_variants_emits_overall_score_and_ddg_score():
    variants = [{"variant_id": "a", "sequence": "MAAA", "num_mutations": 2}]
    brightness = [{"variant_id": "a", "predicted_relative_brightness": 0.8}]
    stability = [{"variant_id": "a", "predicted_retention72": 0.9, "ddg_stability_score": 0.7, "model": "m"}]
    risk = [{"variant_id": "a", "chromophore_risk": 0.1}]
    ranked = rank_variants(variants, brightness, stability, risk)
    assert "overall_score" in ranked[0]
    assert ranked[0]["ddg_stability_score"] == 0.7
    assert ranked[0]["overall_score"] == ranked[0]["final_score"]
