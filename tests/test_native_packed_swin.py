import sys
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_packed_swin import gather_packed, scatter_packed, logical_indices, RecoveredPackedSwin
from native_swin_torch import RecoveredSwin32, gather_windows, scatter_windows


@pytest.mark.parametrize('c', [32, 64, 128, 256])
@pytest.mark.parametrize('ox,oy', [(0, 0), (-4, -4), (-4, 0), (0, -4)])
def test_layout_bijections_and_window_coverage(c, ox, oy):
    a, residual = logical_indices(c)
    assert sorted(a.flatten().tolist()) == list(range(64*c))
    assert sorted(residual.flatten().tolist()) == list(range(64*c))
    width, height = 24, 16
    packed = torch.arange(width*height*c).float()
    windows, mapping = gather_packed(packed, width, height, c, ox, oy)
    index, valid = mapping
    assert sorted(index[valid].tolist()) == list(range(width*height//16))
    physical = torch.arange(16*c)
    head, lane, element = physical // 512, physical % 512 // 16, physical % 16
    row = lane // 4 + element % 8 // 4 * 8
    col = head*32 + lane%4*2 + element%4//2*8 + element//8*16 + (element&1)
    logical = torch.empty(len(windows), 4, 16, c)
    logical[:, :, row, col] = windows.reshape(-1, 4, 16*c)
    assert torch.equal(scatter_packed(logical, mapping, width, height), packed)


def test_existing_32_packed_contract_preserved():
    torch.manual_seed(1201)
    weights = bytes([24] * 20672)
    original, generic = RecoveredSwin32(weights), RecoveredPackedSwin(weights, record_kind='swin32')
    x = (torch.randn(2, 2048) * .125).half()
    assert torch.equal(original.a_index, generic.a_index)
    assert torch.equal(original.residual_index, generic.residual_index)
    assert torch.equal(original(x), generic(x))
    packed = torch.arange(16*24*32).float()
    old, old_mapping = gather_windows(packed, 16, 24, -4, 0)
    new, new_mapping = gather_packed(packed, 16, 24, 32, -4, 0)
    assert torch.equal(old, new)
    value = torch.randn(len(new), 64, 32)
    assert torch.equal(scatter_windows(value, old_mapping, 16, 24), scatter_packed(value, new_mapping, 16, 24))
