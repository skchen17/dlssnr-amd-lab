from scripts.summarize_native_candidate_abba import p95


def test_p95_uses_nearest_rank():
    assert p95(list(range(1, 12))) == 11
    assert p95([3, 1, 2]) == 3
