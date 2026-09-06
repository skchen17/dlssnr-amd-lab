import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_split_image512 import pool_to_bottleneck


def test_pool_preserves_whole_feature_then_explicit_zero_padding():
    x=torch.full((12,20,512),.5).half().requires_grad_()
    result=pool_to_bottleneck(x)
    assert result.shape==(8,12,512)
    assert torch.all(result[:6,:10]==.5)
    assert torch.count_nonzero(result[6:])==0
    assert torch.count_nonzero(result[:,10:])==0
    result.float().sum().backward()
    assert torch.all(x.grad==.25)
