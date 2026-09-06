import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_matrix_abba import evaluate


def rows():
    return {k:{'frame_ms':100,'module_ms':50 if k.startswith('a') else 40,'output_sha256':'same'} for k in ('a1','b1','b2','a2','a3','b3','b4','a4')}


def test_two_series_gate_does_not_promote_game():
    r=evaluate(rows());assert r['two_series_performance_gate'];assert not r['game_promoted']


def test_second_series_regression_blocks_gate():
    r=rows();r['b3']['frame_ms']=110;r['b4']['frame_ms']=110
    assert not evaluate(r)['two_series_performance_gate']


def test_hash_and_missing_run_rejected():
    r=rows();r['b4']['output_sha256']='changed'
    with pytest.raises(ValueError):evaluate(r)
    r.pop('b4')
    with pytest.raises(ValueError):evaluate(r)
