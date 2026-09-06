import sys
from pathlib import Path
from dataclasses import FrozenInstanceError
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_inference_schedule import select_schedule


def test_schedule_preserves_explicit_baseline_and_opt_in():
    baseline=select_schedule('baseline12');fast=select_schedule('native_opt2')
    assert (baseline.window_batch,baseline.query_chunk,baseline.compact_layout)==(12,32,False)
    assert (fast.window_batch,fast.query_chunk,fast.compact_layout)==(384,128,True)
    memory=select_schedule('native_opt3')
    assert (memory.window_batch,memory.query_chunk,memory.compact_layout,memory.allocator_budget_bytes)==(768,1024,True,5000000000)
    assert baseline.allocator_budget_bytes is fast.allocator_budget_bytes is None
    with pytest.raises(FrozenInstanceError):fast.window_batch=999
    with pytest.raises(ValueError):select_schedule('whatever')


def test_invalid_schedule_fails_before_any_gpu_initialization():
    pytest.importorskip('torch')
    from native_frame_runtime import ResidentNativeFrame
    with pytest.raises(ValueError,match='optimization profile'):
        ResidentNativeFrame('missing','missing',optimization_profile='unknown')
