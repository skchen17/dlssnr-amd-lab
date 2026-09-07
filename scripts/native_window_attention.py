"""Original-weight multi-head window attention candidate, not a complete Swin block.

Input is logical post-FFN [windows,64,C], NOT packed features or RGB.
32-channel behavior is cross-checked against the existing native prototype.
Wider-family projection layouts are static recoveries pending RTX intermediate
validation; this module cannot be promoted to a full-network runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
import torch
from torch import nn
from native_swin_torch import decode_e4, packed_b, permute32, quantize_e4
from native_execution_policy import use_native_fp16, native_gemm


@dataclass(frozen=True)
class AttentionLayout:
    channels: int
    record_bytes: int
    qkv: int
    bias: int
    q_scale: int
    project: int
    residual_scale: int


# Byte offsets in unmodified local records. Never infer record kind from length.
LAYOUTS = {
    'swin32': AttentionLayout(32, 20672, 8288, 11360, 19552, 19568, 20592),
    'head32': AttentionLayout(32, 21808, 8400, 11472, 19664, 19680, 20704),
    'swin64': AttentionLayout(64, 61760, 28832, 41120, 57504, 57520, 61616),
    'swin128': AttentionLayout(128, 197184, 98592, 147744, 180512, 180528, 196912),
    'swin256': AttentionLayout(256, 689232, 360992, 557600, 623136, 623168, 688704),
}


def projection_indices(channels, base, *, qkv=False):
    """Tile-major 32x16 E4M3 matrices; Q/K/V interleave per output head.

    Returns [C,3*C] or [C,C] byte indices in logical Q,K,V column order.
    Static evidence: C64 qkv head stride3072, K32 stride6144;
    output projection head stride1024, K32 stride2048.
    """
    if channels not in (32, 64, 128, 256, 512):
        raise ValueError('unrecovered channel count')
    parts = 3 if qkv else 1
    k = torch.arange(channels)[:, None]
    n = torch.arange(channels * parts)[None, :]
    part, col = n // channels, n % channels
    tile = base + k // 32 * (channels // 32) * parts * 1024
    tile = tile + col // 32 * parts * 1024 + part * 1024 + col % 32 // 16 * 512
    return packed_b(tile, k % 32, col % 16).long()


def normalize_qk(t):
    squares = (t * t).half()
    while squares.shape[-1] > 1:
        squares = (squares[..., 0::2] + squares[..., 1::2]).half()
    return (t * squares.float().clamp_min(6.198883056640625e-5).rsqrt().half()).half()


def window_attention(q, k, v, q_scale, position_bias):
    """[batch,heads,64,32], no cross-head mixing or global image attention.

    FP8 probability and FP16 accumulation boundaries are part of this recovered
    candidate. Generic SDPA would silently discard those boundaries.
    """
    q = quantize_e4((normalize_qk(q) * q_scale[None, :, None, None]).half())
    k = quantize_e4(normalize_qk(k))
    scores = ((native_gemm(q, k.transpose(-1, -2)) + position_bias).half() if use_native_fp16() else
              (q.float() @ k.float().transpose(-1, -2) + position_bias.float()).half())
    probability = quantize_e4(torch.softmax(scores.float(), dim=-1).half())
    v = quantize_e4(v)
    if use_native_fp16():
        return native_gemm(probability, v)
    attention = (probability[..., :32].float() @ v[..., :32, :].float()).half()
    return (attention.float() + probability[..., 32:].float() @ v[..., 32:, :].float()).half()


def window_attention_prepared(q,k,v,position_bias):
    """Attention after a native Q/K normalization + E4M3 preparation stage."""
    scores=((native_gemm(q,k.transpose(-1,-2))+position_bias).half() if use_native_fp16() else
            (q.float()@k.float().transpose(-1,-2)+position_bias.float()).half())
    probability=quantize_e4(torch.softmax(scores.float(),dim=-1).half())
    if use_native_fp16():return native_gemm(probability,v)
    attention=(probability[...,:32].float()@v[...,:32,:].float()).half()
    return (attention.float()+probability[...,32:].float()@v[...,32:,:].float()).half()


class RecoveredWindowAttention(nn.Module):
    native_graph_complete = False
    rtx_quality_verified = False

    def __init__(self, packed_weights: bytes, *, record_kind: str):
        super().__init__()
        if record_kind not in LAYOUTS:
            raise ValueError('unknown attention record kind')
        layout = LAYOUTS[record_kind]
        if len(packed_weights) != layout.record_bytes:
            raise ValueError('wrong attention record size')
        self.record_kind, self.channels = record_kind, layout.channels
        self.heads = self.channels // 32
        self.layout_status = 'NATIVE32_CONTROL' if self.heads == 1 else 'STATIC_RECOVERY_PENDING_RTX'
        e4 = decode_e4(packed_weights)
        halves = torch.frombuffer(bytearray(packed_weights), dtype=torch.float16)

        def parameter(name, value):
            value = value.clone().half()
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'nonfinite attention weight: {name}')
            self.register_parameter(name, nn.Parameter(value, requires_grad=False))

        parameter('qkv', e4[projection_indices(self.channels, layout.qkv, qkv=True)])
        parameter('project', e4[projection_indices(self.channels, layout.project)])
        parameter('q_scale', torch.tensor(struct.unpack_from(f'<{self.heads}f', packed_weights, layout.q_scale)))
        parameter('attention_scale', halves[layout.residual_scale // 2:layout.residual_scale // 2 + self.channels])
        query, key = torch.arange(64)[:, None], torch.arange(64)[None, :]
        row, kb = query % 16, key // 8
        lane = row % 8 * 4 + key % 8 // 2
        element = (row >= 8) * 2 + (key & 1)
        bias_index = query // 16 * 2048 + kb // 2 * 512 + lane * 16 + (kb & 1) * 8 + element * 2
        bias_index = layout.bias + torch.arange(self.heads)[:, None, None] * 8192 + bias_index
        parameter('position_bias', halves[bias_index // 2])
        self.register_buffer('permutation', (torch.arange(self.heads)[:, None] * 32 + permute32(torch.arange(32))).flatten().long())

    def _linear(self, x, weights, seed=None):
        x = quantize_e4(x[..., self.permutation])
        weights = quantize_e4(weights)
        if use_native_fp16():
            return native_gemm(x, weights, seed)
        result = seed
        for first in range(0, self.channels, 32):
            value = x[..., first:first + 32].float() @ weights[first:first + 32].float()
            result = value.half() if result is None else (result.float() + value).half()
        return result

    def forward(self, post_ffn):
        if post_ffn.ndim != 3 or post_ffn.shape[1:] != (64, self.channels) or post_ffn.dtype != torch.float16:
            raise ValueError(f'expected logical FP16 [windows,64,{self.channels}] post-FFN input')
        if post_ffn.device != self.qkv.device:
            raise ValueError('input and parameters must share the same device')
        projection = self._linear(post_ffn, self.qkv)
        from native_matrix_fusion import active_matrix_fusion
        matrix=active_matrix_fusion();norm_family=f'c{self.channels}_attention_norm'
        if matrix is not None and norm_family in matrix.modules:
            q,k,v=matrix.wide_attention_norm(self,projection)
            value=window_attention_prepared(q,k,v,self.position_bias)
        else:
            q, k, v = [t.reshape(-1, 64, self.heads, 32).transpose(1, 2) for t in projection.chunk(3, dim=-1)]
            value = window_attention(q, k, v, self.q_scale, self.position_bias)
        value = value.transpose(1, 2).reshape(-1, 64, self.channels)
        seed = (post_ffn * self.attention_scale).half()
        return self._linear(value, self.project, seed)
