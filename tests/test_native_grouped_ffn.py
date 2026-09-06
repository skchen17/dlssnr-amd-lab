import sys
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_grouped_ffn import RecoveredGroupedFFN, RecoveredLogicalSwin, expansion_indices, grouped_contract_indices
from native_swin_torch import RecoveredSwin32
from native_window_attention import LAYOUTS


@pytest.mark.parametrize('c', [32, 64, 128, 256])
def test_weight_sections_cover_full_ffn_without_overlap(c):
    expansion = expansion_indices(c).flatten().tolist()
    contract = grouped_contract_indices(c).flatten().tolist()
    assert sorted(expansion) == list(range(4*c*c))
    assert sorted(contract) == list(range(4*c*c, 4*c*c + 128*c))
    prefix = 4*c*c + 128*c + (c*c if c > 32 else 0)
    assert prefix == {32: 8192, 64: 28672, 128: 98304, 256: 360448}[c]


@pytest.mark.parametrize('kind', list(LAYOUTS))
def test_ffn_and_block_zero_weights(kind):
    layout = LAYOUTS[kind]
    model = RecoveredLogicalSwin(bytes(layout.record_bytes), record_kind=kind)
    x = torch.ones(1, 64, layout.channels).half()
    assert torch.count_nonzero(model(x, x)) == 0
    assert not model.native_graph_complete and not model.tensor_view_adapter_verified
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('kind', ['swin32', 'head32'])
def test_32_ffn_boundary_nonzero_reference(kind):
    torch.manual_seed(1201)
    layout = LAYOUTS[kind]
    raw = bytes([24] * layout.record_bytes)
    old = RecoveredSwin32(raw, record_kind=kind)
    new = RecoveredGroupedFFN(raw, record_kind=kind)
    x = (torch.randn(2, 2048) * .125).half()
    assert torch.equal(old.forward_ffn(x), new(x[:, old.a_index], x[:, old.residual_index]))


def test_explicit_input_views_required():
    model = RecoveredGroupedFFN(bytes(61760), record_kind='swin64')
    with pytest.raises(TypeError):
        model(torch.zeros(1, 64, 64).half())
    with pytest.raises(ValueError):
        model(torch.zeros(1, 64, 64), torch.zeros(1, 64, 64))
