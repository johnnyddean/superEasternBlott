from spebt_agent.tools.filters import exclusion_filter, hard_rule_filter


def test_hard_rule_filter():
    variants = [
        {"variant_id": "ok", "sequence": "M" + "A" * 219},
        {"variant_id": "bad", "sequence": "A" * 220},
    ]
    passed, failed = hard_rule_filter(variants)
    assert [x["variant_id"] for x in passed] == ["ok"]
    assert failed[0]["filter_reasons"]


def test_exclusion_filter():
    variants = [{"variant_id": "x", "sequence": "MAAA"}, {"variant_id": "y", "sequence": "MBBB"}]
    passed, failed = exclusion_filter(variants, {"MAAA"})
    assert [x["variant_id"] for x in passed] == ["y"]
    assert [x["variant_id"] for x in failed] == ["x"]
