"""Static-recovered grouped FFN candidate for C32/64/128/256 Swin.

The wider FFN is C -> 4C -> grouped(C/32,128->32) -> C, not a
dense two-layer C -> 4C -> C MLP. Tensor-view adapters remain unverified.
Arguments explicitly separate matrix-A order from residual order: pretending
these are interchangeable was not established by the captured packed layouts.
"""
import torch
from torch import nn
from native_swin_torch import decode_e4, packed_b, permute32, quantize_e4
from native_window_attention import LAYOUTS, RecoveredWindowAttention, projection_indices
from native_execution_policy import use_native_fp16, native_gemm


def expansion_indices(channels):
    if channels not in (32, 64, 128, 256):
        raise ValueError('unrecovered channel count')
    k = torch.arange(channels)[:, None]
    n = torch.arange(4 * channels)[None, :]
    # One output group contains 128 hidden channels. K32 stride=4096.
    tile = n // 128 * (128 * channels) + k // 32 * 4096 + n % 128 // 16 * 512
    return packed_b(tile, k % 32, n % 16).long()


def grouped_contract_indices(channels):
    if channels not in (32, 64, 128, 256):
        raise ValueError('unrecovered channel count')
    head = torch.arange(channels // 32)[:, None, None]
    k = torch.arange(128)[None, :, None]
    n = torch.arange(32)[None, None, :]
    tile = 4 * channels * channels + head * 4096 + k // 32 * 1024 + n // 16 * 512
    return packed_b(tile, k % 32, n % 16).long()


def cubic_silu(x):
    clamped = x.clamp(-4, 4)
    first = (-.055908203125 * clamped.abs().float() + .447265625).half()
    second = (clamped.float() * first.float() + .89453125).half()
    return (x.float() * second.float()).half()


def quantized_cubic_silu(x):
    from native_fusion_policy import active_fusions
    fusion=active_fusions()
    return fusion.cubic(x) if fusion is not None else quantize_e4(cubic_silu(x))


class RecoveredGroupedFFN(nn.Module):
    native_graph_complete = False
    rtx_quality_verified = False
    tensor_view_adapter_verified = False

    def __init__(self, raw: bytes, *, record_kind: str):
        super().__init__()
        if record_kind not in LAYOUTS or len(raw) != LAYOUTS[record_kind].record_bytes:
            raise ValueError('wrong FFN record kind/size')
        self.channels = c = LAYOUTS[record_kind].channels
        self.heads = c // 32
        e4 = decode_e4(raw)

        def parameter(name, value):
            value = value.clone().half()
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'nonfinite FFN weight {name}')
            self.register_parameter(name, nn.Parameter(value, requires_grad=False))

        parameter('expand', e4[expansion_indices(c)])
        parameter('contract', e4[grouped_contract_indices(c)])
        prefix = 4 * c * c + 128 * c
        if c > 32:
            parameter('mix', e4[projection_indices(c, prefix)])
            prefix += c * c
        else:
            self.register_parameter('mix', None)
        halves = torch.frombuffer(bytearray(raw), dtype=torch.float16)
        parameter('residual_scale', halves[(prefix + 16) // 2:(prefix + 16) // 2 + c])
        self.register_buffer('permutation', permute32(torch.arange(32)).long())

    def forward(self, matrix_a_features, residual_features):
        c = self.channels
        if (matrix_a_features.ndim != 3 or matrix_a_features.shape[1:] != (64, c)
                or residual_features.shape != matrix_a_features.shape
                or matrix_a_features.dtype != torch.float16 or residual_features.dtype != torch.float16):
            raise ValueError('expected separate matrix-A and residual FP16 [windows,64,C] views')
        if any(t.device != self.expand.device for t in (matrix_a_features, residual_features)):
            raise ValueError('FFN inputs and parameters must share a device')
        x = quantize_e4(matrix_a_features)
        weight = quantize_e4(self.expand)
        if use_native_fp16():
            hidden = native_gemm(x, weight)
        else:
            hidden = None
            for k in range(0, c, 32):
                value = x[..., k:k+32].float() @ weight[k:k+32].float()
                hidden = value.half() if hidden is None else (hidden.float() + value).half()
        hidden = quantized_cubic_silu(hidden).reshape(-1, 64, self.heads, 4, 32)
        hidden = hidden.permute(0, 2, 1, 3, 4)  # B,H,T,4,K32
        seed = (residual_features * self.residual_scale).half()
        contracted = seed[:, None] if c == 32 else None
        if use_native_fp16():
            contracted = native_gemm(hidden[..., self.permutation].flatten(-2), quantize_e4(self.contract), contracted)
        else:
            for p in range(4):
                value = hidden[..., p, self.permutation].float() @ quantize_e4(self.contract[:, p*32:(p+1)*32]).float()
                contracted = value.half() if contracted is None else (contracted.float() + value).half()
        if self.mix is None:
            return contracted[:, 0]
        grouped = quantize_e4(contracted[..., self.permutation]).transpose(1, 2).reshape(-1, 64, c)
        weight = quantize_e4(self.mix)
        if use_native_fp16():
            return native_gemm(grouped, weight, seed)
        result = seed
        for k in range(0, c, 32):
            result = (result.float() + grouped[..., k:k+32].float() @ weight[k:k+32].float()).half()
        return result


class RecoveredLogicalSwin(nn.Module):
    """Native FFN + attention; does not claim packed-image adapters are restored."""
    native_graph_complete = False
    tensor_view_adapter_verified = False
    rtx_quality_verified = False

    def __init__(self, raw: bytes, *, record_kind: str):
        super().__init__()
        self.ffn = RecoveredGroupedFFN(raw, record_kind=record_kind)
        self.attention = RecoveredWindowAttention(raw, record_kind=record_kind)

    def forward(self, matrix_a_features, residual_features):
        return self.attention(self.ffn(matrix_a_features, residual_features))
