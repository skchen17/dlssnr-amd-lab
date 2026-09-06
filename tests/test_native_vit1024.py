import sys
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_vit1024 import matrix_indices, qkv_indices, global_attention, exp_candidate, RecoveredVit1024
from native_sequence_layout import pack_sequence, unpack_sequence, sequence_indices


@pytest.mark.parametrize('tokens,channels', [(1, 32), (17, 64), (96, 1024), (129, 32)])
def test_sequence_roundtrip_and_padding(tokens, channels):
    x = torch.arange(tokens*channels).reshape(1, tokens, channels).float()
    packed = pack_sequence(x)
    assert torch.equal(unpack_sequence(packed, tokens, channels), x)
    index = sequence_indices(tokens, channels).flatten()
    assert torch.unique(index).numel() == tokens*channels
    mask = torch.ones_like(packed, dtype=torch.bool); mask[index] = False
    assert torch.count_nonzero(packed[mask]) == 0


def test_rectangular_weight_map_covers_all_bytes():
    assert sorted(matrix_indices(64, 128, 16).flatten().tolist()) == list(range(16, 16+8192))
    index = qkv_indices().flatten()
    assert torch.unique(index).numel() == 3145728
    assert int(index.min()) == 128 and int(index.max()) == 3145855


def test_exponent_finite_monotonic_and_trainable():
    x = torch.linspace(-8, 8, 65).half().requires_grad_()
    out = exp_candidate(x)
    assert torch.isfinite(out).all() and (out > 0).all()
    assert (out[1:] >= out[:-1]).all()
    out.float().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)


def test_global_attention_chunk_invariance_and_distant_context():
    torch.manual_seed(1201)
    q, k, v = [(torch.randn(1, 2, 97, 32)*.125).half() for _ in range(3)]
    first = global_attention(q, k, v, query_chunk=13, key_chunk=32)
    second = global_attention(q, k, v, query_chunk=32, key_chunk=128)
    assert torch.equal(first, second)
    v[..., -1, :] += 4
    changed = global_attention(q, k, v)
    assert not torch.equal(changed[..., 0, :], first[..., 0, :])


def test_global_backward_no_optimizer():
    q = torch.ones(1, 1, 3, 32).half().requires_grad_()
    k = torch.ones_like(q)*.1
    v = torch.randn_like(q).requires_grad_()
    global_attention(q, k, v).float().square().mean().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(v.grad).all()
    assert torch.count_nonzero(v.grad)


def test_explicit_larger_query_workspace_preserves_key_reduction():
    torch.manual_seed(32)
    q,k,v=[(torch.randn(1,2,257,32)*.125).half() for _ in range(3)]
    reference=global_attention(q,k,v,query_chunk=128,key_chunk=128)
    assert torch.equal(reference,global_attention(q,k,v,query_chunk=1024,key_chunk=128,max_score_elements=4194304))
    with pytest.raises(ValueError,match='workspace budget'):
        global_attention(q,k,v,max_score_elements=999999999)


def test_bad_geometry_and_records_rejected():
    with pytest.raises(ValueError):
        RecoveredVit1024()
    with pytest.raises(ValueError):
        matrix_indices(33, 32)
    with pytest.raises(ValueError):
        unpack_sequence(torch.zeros(32), 2, 32)
    q = torch.zeros(1, 1, 2, 32).half()
    with pytest.raises(ValueError):
        global_attention(q, q, q, key_chunk=7)
