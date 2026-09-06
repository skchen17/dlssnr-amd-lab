import pytest
import torch

from native_grid_policy import grid_policy,run_windows,whole_grid_enabled


class Recorder:
    def __init__(self):self.shapes=[]
    def __call__(self,value):
        self.shapes.append(tuple(value.shape))
        return value+1


def test_default_preserves_batched_schedule_and_values():
    source=torch.arange(30).reshape(10,3)
    op=Recorder();actual=run_windows(op,source,4,'c32')
    assert torch.equal(actual,source+1)
    assert op.shapes==[(4,3),(4,3),(2,3)]


def test_family_scope_runs_one_whole_grid_call_and_restores():
    source=torch.arange(30).reshape(10,3);op=Recorder()
    assert not whole_grid_enabled('c32')
    with grid_policy(('c32',)):
        assert whole_grid_enabled('c32')
        actual=run_windows(op,source,4,'c32')
    assert torch.equal(actual,source+1)
    assert op.shapes==[(10,3)]
    assert not whole_grid_enabled('c32')


def test_policy_rejects_unknown_family_and_invalid_batch():
    with pytest.raises(ValueError,match='unreviewed'):
        with grid_policy(('unknown',)):pass
    with pytest.raises(ValueError,match='positive'):
        run_windows(Recorder(),torch.zeros(1,1),0,'c32')
