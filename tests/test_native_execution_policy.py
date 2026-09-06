import sys
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from native_execution_policy import execution_policy, current_profile, native_gemm, profile_description
from native_split_swin512 import chunked_linear
from native_swin_torch import quantize_e4
from native_grouped_ffn import RecoveredGroupedFFN
from native_window_attention import LAYOUTS, window_attention
from native_vit1024 import global_attention


def test_profile_is_explicit_nested_and_exception_safe():
    assert current_profile() == 'recovered_k32'
    with execution_policy('native_fp16'):
        assert not profile_description()['original_weights_modified']
        with pytest.raises(RuntimeError):
            with execution_policy('recovered_k32'):
                raise RuntimeError('test')
        assert current_profile() == 'native_fp16'
    assert current_profile() == 'recovered_k32'
    with pytest.raises(ValueError):
        with execution_policy('unknown'):
            pass


def test_default_chunked_reference_is_unchanged():
    torch.manual_seed(42)
    x, w = [torch.randn(*shape).half() for shape in [(2, 3, 64), (64, 32)]]
    a, b = quantize_e4(x), quantize_e4(w)
    expected = ((a[..., :32].float() @ b[:32].float()).half().float()
                + a[..., 32:].float() @ b[32:].float()).half()
    assert torch.equal(chunked_linear(x, w), expected)
    before = w.clone()
    with execution_policy('native_fp16'):
        assert torch.equal(chunked_linear(x, w), a @ b)
    assert torch.equal(before, w)


def test_grouped_native_gemm_has_gradients_and_rejects_wrong_types():
    x = torch.randn(2, 3, 7, 64).half().requires_grad_()
    w = torch.randn(3, 64, 32).half().requires_grad_()
    seed = torch.ones(2, 3, 7, 32).half()
    result = native_gemm(x, w, seed)
    assert result.shape == seed.shape
    result.float().square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    with pytest.raises(ValueError):
        native_gemm(x.float(), w)


@pytest.mark.parametrize('kind', ['swin32', 'swin64', 'swin128', 'swin256'])
def test_all_grouped_ffn_shapes_native_profile(kind):
    model = RecoveredGroupedFFN(bytes(LAYOUTS[kind].record_bytes), record_kind=kind)
    x = torch.ones(1, 64, model.channels).half().requires_grad_()
    with execution_policy('native_fp16'):
        result = model(x, x)
        assert result.shape == x.shape and torch.count_nonzero(result) == 0
        result.float().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_fast_attention_keeps_global_context_and_finite_gradients():
    torch.manual_seed(1201)
    q, k, v = [(torch.randn(1, 2, 97, 32)*.1).half() for _ in range(3)]
    with execution_policy('native_fp16'):
        first = global_attention(q, k, v)
        v[..., -1, :] += 4
        second = global_attention(q, k, v)
        assert not torch.equal(first[..., 0, :], second[..., 0, :])
        q.requires_grad_()
        global_attention(q, k, v).float().square().mean().backward()
    assert torch.isfinite(q.grad).all()


@pytest.mark.parametrize('kind', ['swin32', 'swin64', 'swin128', 'swin256'])
def test_grouped_fast_path_changes_rounding_not_channel_order(kind):
    torch.manual_seed(73)
    model = RecoveredGroupedFFN(bytes(LAYOUTS[kind].record_bytes), record_kind=kind)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_((torch.randn_like(parameter)*.05).half())
        model.residual_scale.fill_(1)
    a = (torch.randn(2, 64, model.channels)*.1).half()
    residual = (torch.randn_like(a)*.1).half()
    reference = model(a, residual)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with execution_policy('native_fp16'):
        actual = model(a, residual)
    error = (actual.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt()
    assert error < .02
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
