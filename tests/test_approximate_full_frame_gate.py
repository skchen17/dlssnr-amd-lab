import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_approximate_track_is_explicit_deterministic_and_measured():
    source = (ROOT / 'scripts' / 'benchmark_nr_resolution.py').read_text(encoding='utf-8')
    assert "p.add_argument('--allow-approximate-output',action='store_true'" in source
    assert "digest!=candidate_hash" in source
    assert "'max_absolute_error':float(delta.abs().max())" in source
    assert "'nrmse':rmse/denominator" in source
    assert "'accuracy_track':'approximate_deterministic'" in source


def test_strict_hash_gate_remains_the_default():
    source = (ROOT / 'scripts' / 'benchmark_nr_resolution.py').read_text(encoding='utf-8')
    assert "if not a.allow_approximate_output and digest!=reference" in source
    assert "'strict_hash'" in source
