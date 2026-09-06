import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_swin_torch import RecoveredSwin32,quantize_e4


def model():
    m=RecoveredSwin32(bytes(20672)).eval()
    m.inference_cache_enabled=True
    return m


def test_cache_reuses_exact_frozen_transforms_and_invalidates_mutation():
    m=model()
    with torch.no_grad():
        first=m._weight('contract',part=1,fp32=True)
        assert first is m._weight('contract',part=1,fp32=True)
        m.contract[1].fill_(.3)
        second=m._weight('contract',part=1,fp32=True)
        assert second is not first
        assert torch.equal(second,quantize_e4(m.contract[1]).float())
        assert not any('inference' in k for k in m.state_dict())
        m.contract=torch.nn.Parameter(torch.ones_like(m.contract),requires_grad=False)
        assert torch.equal(m._weight('contract',part=1,fp32=True),torch.ones_like(second))


def test_trainable_path_keeps_ste_and_ignores_cached_tensor():
    m=model()
    with torch.no_grad():m._weight('expand')
    m.expand.requires_grad_(True)
    value=m._weight('expand')
    value.float().sum().backward()
    assert torch.equal(m.expand.grad,torch.ones_like(m.expand))
    m.train()
    assert not m._inference_weights


def test_cache_cleared_for_device_dtype_application_and_state_load():
    m=model()
    with torch.no_grad():
        first=m._weight('expand')
        m.cpu()
        assert not m._inference_weights
        first=m._weight('expand')
        state=m.state_dict();state['expand']=torch.ones_like(m.expand)
        m.load_state_dict(state)
        assert m._weight('expand') is not first
        assert torch.equal(m._weight('expand'),torch.ones_like(m.expand))


def test_frozen_full_operator_cache_exact_and_disabled_by_default():
    m=RecoveredSwin32(bytes(20672)).eval()
    assert not m.inference_cache_enabled
    x=torch.randn(3,2048).half()
    with torch.no_grad():
        a=m(x);m.inference_cache_enabled=True;b=m(x)
        assert torch.equal(a,b)
        assert len(m._inference_weights)==8
