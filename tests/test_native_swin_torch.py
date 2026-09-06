import sys
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_swin_torch import (RecoveredSwin32, decode_e4, encode_e4,
                               gather_windows, scatter_windows, packed_a, packed_b,
                               permute32, quantize_e4)


def test_fragment_mappings_are_permutations():
    g, r, k = torch.arange(4)[:, None, None], torch.arange(16)[None, :, None], torch.arange(32)[None, None, :]
    assert sorted(packed_a(g, r, k).flatten().tolist()) == list(range(2048))
    assert sorted(packed_b(0, torch.arange(32)[:, None], torch.arange(16)[None]).flatten().tolist()) == list(range(512))
    assert sorted(permute32(torch.arange(32)).tolist()) == list(range(32))


@pytest.mark.parametrize('ox,oy', [(0, 0), (-4, 0), (0, -4), (-4, -4)])
@pytest.mark.parametrize('width,height', [(8, 8), (16, 24), (648, 368)])
def test_gather_scatter_identity_and_boundary_coverage(ox, oy, width, height):
    # Fill projected logical values by inverting scatter (not assuming A layout).
    packed = torch.arange(width * height * 32, dtype=torch.float32)
    raw, mapping = gather_windows(packed, width, height, ox, oy)
    lane, element = torch.arange(512) // 16, torch.arange(512) % 16
    row = lane // 4 + element % 8 // 4 * 8
    col = lane % 4 * 2 + element % 4 // 2 * 8 + element // 8 * 16 + (element & 1)
    logical = torch.empty(raw.shape[0], 4, 16, 32)
    logical[:, :, row, col] = raw.reshape(-1, 4, 512)
    assert torch.equal(scatter_windows(logical, mapping, width, height), packed)
    indices, valid = mapping
    assert sorted(indices[valid].tolist()) == list(range(width // 4 * (height // 4)))


def test_e4_finite_codes_roundtrip_and_saturation():
    raw = bytes(i for i in range(256) if (i & 127) != 127)
    assert bytes(encode_e4(decode_e4(raw)).tolist()) == raw
    x = torch.tensor([-1000., 1000.], dtype=torch.float16)
    assert quantize_e4(x).tolist() == [-448, 448]


def test_quantization_ste_has_gradient():
    x = torch.tensor([.11, -.13, .19], requires_grad=True)
    quantize_e4(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_malformed_geometry_rejected():
    for args in [(0, 8, 0, 0), (9, 8, 0, 0), (8, 8, 4, 0)]:
        with pytest.raises(ValueError):
            gather_windows(torch.zeros(2048), *args)
    with pytest.raises(ValueError):
        gather_windows(torch.zeros(1), 8, 8, 0, 0)
    with pytest.raises(ValueError):
        RecoveredSwin32(bytes(32))


def test_zero_record_no_missing_stage_claim():
    model = RecoveredSwin32(bytes(20672))
    assert not model.native_graph_complete
    assert not any(p.requires_grad for p in model.parameters())
    assert torch.equal(model(torch.zeros(1, 2048, dtype=torch.float16)), torch.zeros(1, 64, 32, dtype=torch.float16))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 64, 32))
