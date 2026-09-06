import sys
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_head_torch import RecoveredHead32, compose_legacy_sdr_debug


@pytest.mark.parametrize('width,height,pw,ph', [(8,8,8,8), (9,7,16,8), (641,361,648,368)])
def test_head_surface_coverage_with_odd_sizes(width, height, pw, ph):
    windows = ((pw + 11) // 8) * ((ph + 11) // 8)
    base = torch.rand(height, width, 4, dtype=torch.float16)
    residual = torch.zeros(windows, 64, 4, dtype=torch.float16)
    image = compose_legacy_sdr_debug(residual, base, pw, ph)
    assert torch.equal(image[..., :3], base[..., :3])
    assert (image[..., 3] == 1).all()


def test_head_requires_upstream_not_rgb_substitution():
    model = RecoveredHead32(bytes(21808))
    assert not model.native_graph_complete
    with pytest.raises(ValueError):
        model.fuse(torch.zeros(8 * 8 * 4), torch.zeros(8 * 8 * 4), torch.tensor([0]), 8, 8)
    fused = model.fuse(torch.zeros(8 * 8 * 8), torch.zeros(8 * 8 * 32), torch.tensor([0]), 8, 8)
    assert fused.shape == (1, 2048)
    assert torch.count_nonzero(fused) == 0


def test_head_rejects_nan_scale():
    raw = bytearray(21808)
    raw[8272:8274] = b'\x00\x7e'
    with pytest.raises(ValueError, match='nonfinite'):
        RecoveredHead32(raw)
