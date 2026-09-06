import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_head_torch import RecoveredHead32


def test_cached_indices_are_exact_and_do_not_cache_features():
    torch.manual_seed(2)
    m=RecoveredHead32(bytes(21808)).eval()
    # Observe actual fused features, avoiding all-zero model hiding index mistakes.
    m.project=lambda fused:fused
    with torch.no_grad():
        m.main_scale.fill_(.5);m.skip_scale.fill_(.25)
        a=torch.randn(16*24*8).half();b=torch.randn(16*24*32).half()
        ids=torch.arange(1,7)
        reference=m(a,b,ids,24,16)
        m.cache_layout_enabled=True
        actual=m.forward_range(a,b,1,6,24,16)
        assert torch.equal(reference,actual)
        parts=m._layout_cache[(1,6)]
        assert parts[0].dtype==parts[2].dtype==torch.int32
        assert m._layout_bytes==6*2048*10
        assert torch.equal(m(a*2,b,ids,24,16),m.forward_range(a*2,b,1,6,24,16))
        assert not torch.equal(actual,m.forward_range(a*2,b,1,6,24,16))
        assert m._layout_cache[(1,6)] is parts
        m.forward_range(torch.zeros(32*16*8).half(),torch.zeros(32*16*32).half(),0,2,32,16)
        assert set(m._layout_cache)=={(0,2)}
        m.cpu()
        assert not m._layout_cache and m._layout_bytes==0


def test_gradients_bypass_cache_and_ranges_reject():
    m=RecoveredHead32(bytes(21808)).eval();m.cache_layout_enabled=True
    m.project=lambda fused:fused
    with torch.no_grad():m.main_scale.fill_(1);m.skip_scale.fill_(1)
    a=torch.ones(8*8*8,dtype=torch.float16,requires_grad=True)
    b=torch.ones(8*8*32,dtype=torch.float16,requires_grad=True)
    m.forward_range(a,b,0,4,8,8).float().sum().backward()
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    assert not m._layout_cache
    for start,count in [(-1,1),(0,0),(0,99),(True,1)]:
        with pytest.raises(ValueError,match='CTA range'):m.forward_range(a,b,start,count,8,8)
