from spebt_agent.data.prepare_competition_data import apply_mutations_with_offset, parse_mutations


def test_parse_mutations_and_offset():
    muts = parse_mutations("A1C:G3D")
    assert muts == [("A", 1, "C"), ("G", 3, "D")]
    assert apply_mutations_with_offset("MAGG", muts) == "MCGD"


def test_parse_rejects_stop():
    assert parse_mutations("*238G") is None
