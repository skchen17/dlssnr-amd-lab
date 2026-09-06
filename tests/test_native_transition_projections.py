import sys
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_transition_projections import EncoderFinalProjection, DecoderInputProjection


@pytest.mark.parametrize('cls,size,inputs,outputs', [(EncoderFinalProjection, 524304, 512, 1024),
                                                    (DecoderInputProjection, 525312, 1024, 512)])
def test_original_projection_contract(cls, size, inputs, outputs):
    model = cls(bytes(size))
    result = model(torch.zeros(1, 3, inputs).half())
    assert result.shape == (1, 3, outputs) and torch.count_nonzero(result) == 0
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 4).half())
    with pytest.raises(ValueError):
        cls(bytes(size-16))


def test_spatial_adapter_requires_actual_skip_and_explicit_geometry():
    model = DecoderInputProjection(bytes(525312))
    with pytest.raises(ValueError):
        model.fuse(torch.zeros(1), torch.zeros(1),20,12)
    with torch.no_grad():
        model.residual_scale.fill_(1)
    skip=torch.full((12,20,512),.5).half().requires_grad_()
    actual=model.fuse(torch.zeros(8,12,1024).half(),skip,20,12)
    assert torch.equal(actual,skip)
    actual.float().sum().backward()
    assert torch.equal(skip.grad,torch.ones_like(skip))
    with pytest.raises(ValueError):
        model.fuse(torch.zeros(8,8,1024).half(),skip,20,12)
