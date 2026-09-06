import sys
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_window_attention import (LAYOUTS, RecoveredWindowAttention,
                                     projection_indices, window_attention)
from native_swin_torch import RecoveredSwin32


@pytest.mark.parametrize('kind', list(LAYOUTS))
def test_sections_are_bounded_nonoverlapping_and_bijective(kind):
    layout = LAYOUTS[kind]
    c = layout.channels
    for start, parts in [(layout.qkv, 3), (layout.project, 1)]:
        indices = projection_indices(c, start, qkv=parts == 3)
        assert sorted(indices.flatten().tolist()) == list(range(start, start + parts * c * c))
    assert layout.qkv + 3 * c * c == layout.bias
    assert layout.bias + (c // 32) * 8192 == layout.q_scale
    assert layout.q_scale + c // 32 * 4 <= layout.project
    assert layout.project + c * c == layout.residual_scale
    assert layout.residual_scale + c * 2 <= layout.record_bytes


@pytest.mark.parametrize('kind', list(LAYOUTS))
def test_original_record_constructor_and_zero_forward(kind):
    model = RecoveredWindowAttention(bytes(LAYOUTS[kind].record_bytes), record_kind=kind)
    assert not model.native_graph_complete and not model.rtx_quality_verified
    assert not any(p.requires_grad for p in model.parameters())
    result = model(torch.zeros(1, 64, model.channels, dtype=torch.float16))
    assert result.shape == (1, 64, model.channels)
    assert torch.count_nonzero(result) == 0


def test_head_independence_and_gradient():
    torch.manual_seed(1201)
    q = (torch.randn(1, 4, 64, 32) * .1).half().requires_grad_()
    k, v = torch.randn_like(q), torch.randn_like(q)
    scale, bias = torch.ones(4).half(), torch.zeros(4, 64, 64).half()
    result = window_attention(q, k, v, scale, bias)
    split = torch.cat([window_attention(q[:, h:h+1], k[:, h:h+1], v[:, h:h+1], scale[h:h+1], bias[h:h+1]) for h in range(4)], dim=1)
    assert torch.equal(result, split)
    result.float().square().mean().backward()
    assert torch.isfinite(q.grad).all() and torch.count_nonzero(q.grad) > 0


@pytest.mark.parametrize('kind', ['swin32', 'head32'])
def test_32_boundary_matches_existing_prototype(kind):
    # Synthetic nonzero E4 weights; scale/bias region is initialized explicitly.
    layout = LAYOUTS[kind]
    raw = bytearray([24] * layout.record_bytes)
    raw[layout.bias:] = bytes(layout.record_bytes - layout.bias)
    model = RecoveredSwin32(bytes(raw), record_kind=kind)
    attention = RecoveredWindowAttention(bytes(raw), record_kind=kind)
    torch.manual_seed(1201)
    value = (torch.randn(2, 2048) * .2).half()
    assert torch.equal(attention(model.forward_ffn(value)), model(value))


def test_invalid_records_and_inputs_rejected():
    with pytest.raises(ValueError):
        RecoveredWindowAttention(bytes(1), record_kind='swin64')
    with pytest.raises(ValueError):
        RecoveredWindowAttention(bytes(20672), record_kind='unknown')
    model = RecoveredWindowAttention(bytes(61760), record_kind='swin64')
    for shape, dtype in [((1, 64, 32), torch.float16), ((1, 4096), torch.float16), ((1, 64, 64), torch.float32)]:
        with pytest.raises(ValueError):
            model(torch.zeros(shape, dtype=dtype))
    raw = bytearray(61760)
    raw[57504:57508] = bytes.fromhex('0000807f')  # invalid infinite query scale
    with pytest.raises(ValueError, match='nonfinite'):
        RecoveredWindowAttention(bytes(raw), record_kind='swin64')
