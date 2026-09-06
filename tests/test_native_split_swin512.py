import sys
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_split_swin512 import (RECORD_SIZES, RecoveredSplitSwin512, grouped_indices,
                                   SplitProjection512, SplitAttention512, chunked_linear)
from native_window_attention import projection_indices
from native_packed_swin import gather_packed, logical_indices


@pytest.mark.parametrize('expand,start,end', [(True, 262144, 393216), (False, 393216, 524288)])
def test_grouped_sections_bijective(expand, start, end):
    assert sorted(grouped_indices(expand).flatten().tolist()) == list(range(start, end))


@pytest.mark.parametrize('qkv', [False, True])
def test_512_projection_bijective(qkv):
    n = 512*512*(3 if qkv else 1)
    assert sorted(projection_indices(512, 0, qkv=qkv).flatten().tolist()) == list(range(n))


def test_split_records_are_not_merged_or_guessed():
    with pytest.raises(ValueError):
        RecoveredSplitSwin512(**{key: bytes(size-1) for key, size in RECORD_SIZES.items()})
    with pytest.raises(ValueError):
        SplitProjection512(bytes(263168), kind='unknown')
    with pytest.raises(ValueError):
        SplitAttention512(bytes(917568-1))


def test_zero_projection_preserves_scaled_residual():
    raw = bytearray(263168)
    raw[262144:] = bytes.fromhex('003c') * 512
    model = SplitProjection512(bytes(raw), kind='ffwd_projection')
    assert torch.equal(model(torch.zeros(1, 64, 512).half(), torch.ones(1, 64, 512).half()), torch.ones(1, 64, 512).half())
    with pytest.raises(ValueError):
        model(torch.zeros(1, 64, 512).half(), torch.zeros(2, 64, 512).half())


def test_four_aligned_feature_size_20_by_12():
    packed = torch.zeros(20*12*512).half()
    windows, (index, valid) = gather_packed(packed, 20, 12, 512, -4, -4)
    assert windows.shape == (6, 64*512)
    assert sorted(index[valid].tolist()) == list(range(15))
    a, residual = logical_indices(512)
    assert sorted(a.flatten().tolist()) == sorted(residual.flatten().tolist()) == list(range(64*512))


def test_chunked_linear_differentiable():
    x = torch.ones(1, 2, 64).half().requires_grad_()
    w = torch.full((64, 2), .125).half().requires_grad_()
    output = chunked_linear(x, w)
    output.float().sum().backward()
    assert torch.equal(output, torch.full((1, 2, 2), 8.).half())
    assert torch.count_nonzero(x.grad) and torch.count_nonzero(w.grad)
