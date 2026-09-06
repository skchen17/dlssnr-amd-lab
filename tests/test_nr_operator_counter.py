import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nr_operator_counter import OperatorCounter


def test_counter_is_not_gpu_dispatch_and_preserves_math():
    x=torch.arange(16).reshape(4,4).float()
    c=OperatorCounter()
    with c:
        c.block=0;y=x+x
        c.block=1;z=y@y
    assert torch.equal(z,(x+x)@(x+x))
    report=c.report()
    assert report['gpu_dispatch_count'] is None and report['total']>=2
    assert set(report['blocks'])=={'0','1'}
