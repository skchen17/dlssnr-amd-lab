import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_fusion_policy import active_fusions,fusion_policy
from native_grouped_ffn import cubic_silu,quantized_cubic_silu
from native_swin_torch import quantize_e4


def test_default_unfused_math_and_gradients_unchanged():
    assert active_fusions() is None
    x=torch.linspace(-8,8,1024).half().requires_grad_()
    a=quantized_cubic_silu(x);b=quantize_e4(cubic_silu(x))
    assert torch.equal(a,b)
    a.float().sum().backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(RuntimeError):
        with fusion_policy():raise RuntimeError('test')
    assert active_fusions() is None
