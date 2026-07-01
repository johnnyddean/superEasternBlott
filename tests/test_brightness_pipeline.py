import numpy as np

from spebt_agent.tools.brightness import build_feature_matrix, robust_minmax
from spebt_agent.tools.esmc import sequence_cache_key


def test_sequence_cache_key_is_stable():
    assert sequence_cache_key("MAAA") == sequence_cache_key("maaa")


def test_robust_minmax_constant():
    out = robust_minmax([5.0, 5.0, 5.0])
    assert np.allclose(out, 0.5)


def test_build_feature_matrix_shape():
    records = [
        {"sequence": "MAAA", "num_mutations": 1, "parent": "avGFP"},
        {"sequence": "MTTT", "num_mutations": 2, "parent": "cgreGFP"},
    ]
    embeddings = np.ones((2, 8), dtype=np.float32)
    mat = build_feature_matrix(records, embeddings, ["avGFP", "cgreGFP"])
    assert mat.shape == (2, 12)
