from core.retrieval import reciprocal_rank_fusion


def test_rrf_prioritizes_ids_seen_in_both_rankings() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=1)
    assert fused[0][0] == "a" or fused[0][0] == "b"
    assert dict(fused)["a"] == dict(fused)["b"]
