import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_upsample_swin import RecoveredUpsampleSwin,RECORD_SIZES,plain_record,sections
from native_window_attention import LAYOUTS
from native_multiscale import pack_image,pack_outview


@pytest.mark.parametrize('kind',list(RECORD_SIZES))
def test_sections_and_true_skip_fusion(kind):
    raw=bytes(RECORD_SIZES[kind])
    assert len(plain_record(raw,kind))==LAYOUTS[kind].record_bytes
    model=RecoveredUpsampleSwin(raw,record_kind=kind)
    c=model.channels
    with torch.no_grad():
        model.skip_scale.fill_(1)
    low=pack_outview(torch.zeros(4,4,2*c).half())
    skip=torch.randn(8,8,c).half().requires_grad_()
    fused=model.fuse(low,pack_image(skip),8,8)
    assert torch.equal(fused,skip)
    fused.float().sum().backward()
    assert torch.equal(skip.grad,torch.ones_like(skip))
    with pytest.raises(ValueError):
        model.fuse(low,pack_image(skip)[:-1],8,8)


def test_shape_and_record_gaps_cannot_be_silently_filled():
    with pytest.raises(ValueError):
        plain_record(bytes(22783),'swin32')
    with pytest.raises(ValueError):
        sections('unknown')
