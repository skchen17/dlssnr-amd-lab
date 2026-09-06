"""Original-weight 1024-channel ViT tensor candidate, not a full image runtime.

FFN storage is E4M3 (not the legacy capture manifest's fp16 label). Global
attention streams query/key chunks; it never constructs a whole N-by-N matrix.
All parameters stay frozen by default. CPU execution is reference-only.
"""
import hashlib
import struct
import torch
from torch import nn
from native_swin_torch import decode_e4, packed_b, quantize_e4
from native_grouped_ffn import cubic_silu,quantized_cubic_silu
from native_split_swin512 import parameter, chunked_linear, channel_permutation
from native_window_attention import normalize_qk
from native_execution_policy import use_native_fp16, native_gemm


RECORD_SIZES = {'expand': 4194320, 'contract': 4196352, 'qkv': 3145856,
                'attention_abi_placeholder': 2, 'projection': 1050624}


def matrix_indices(inputs, outputs, base=0):
    if inputs <= 0 or outputs <= 0 or inputs % 32 or outputs % 32:
        raise ValueError('matrix dimensions must be positive multiples of32')
    k, n = torch.arange(inputs)[:, None], torch.arange(outputs)[None, :]
    return packed_b(base + k // 32 * outputs * 32 + n // 16 * 512, k % 32, n % 16).long()


def qkv_indices():
    k, n = torch.arange(1024)[:, None], torch.arange(3072)[None, :]
    part, channel = n // 1024, n % 1024
    tile = 128 + k // 32 * 98304 + channel // 32 * 3072 + part * 1024 + channel % 32 // 16 * 512
    return packed_b(tile, k % 32, channel % 16).long()


def exp_candidate(scores):
    """Recovered bounded half exponent approximation with explicit STE.

    This expresses the original approximation as a tensor-level scalar function,
    not an instruction emulator. The derivative is a smooth exp surrogate for
    later training; no claim of exact NVIDIA rounding is made.
    """
    a, b = .08953857421875, 1.708984375  # half-rounded source constants
    mapped = (scores.float() * a + b).half().clamp(1.439453125, 1.9775390625)
    bits = mapped.contiguous().view(torch.int16).int()
    result = ((bits * 16 + 16384) & 65535).to(torch.int16).view(torch.float16)
    if scores.requires_grad:
        surrogate = scores.float().clamp(-3.02, 3.02).exp().half()
        return surrogate + (result - surrogate).detach()
    return result


def global_attention(q, k, v, *, query_chunk=32, key_chunk=128, max_score_elements=1048576):
    """B,H,N,32. Quantize unnormalized weights, normalize AFTER P@V.

    Learned Q scale is applied upstream. There is no implicit 1/sqrt(d) factor,
    causal mask, local-window restriction, or replacement by generic SDPA.
    Padding keys are never included in either numerator or denominator.
    """
    if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 32
            or q.shape[-2] <= 0 or query_chunk <= 0 or key_chunk <= 0 or key_chunk % 32):
        raise ValueError('invalid global attention geometry/chunk size')
    if any(t.dtype != torch.float16 or t.device != q.device for t in (q, k, v)):
        raise ValueError('Q/K/V must be FP16 on one device')
    if max_score_elements not in (1048576,4194304):
        raise ValueError('unreviewed score workspace budget')
    if q.shape[0]*q.shape[1]*min(query_chunk, q.shape[-2])*min(key_chunk, k.shape[-2]) > max_score_elements:
        raise ValueError('attention chunk exceeds bounded score workspace; reduce chunk sizes')
    chunks = []
    for start in range(0, q.shape[-2], query_chunk):
        queries = q[..., start:start+query_chunk, :]
        numerator = torch.zeros_like(queries)
        denominator = torch.zeros((*queries.shape[:-1], 1), device=q.device, dtype=torch.float32)
        for key_start in range(0, k.shape[-2], key_chunk):
            keys, values = k[..., key_start:key_start+key_chunk, :], v[..., key_start:key_start+key_chunk, :]
            score = (native_gemm(queries, keys.transpose(-1, -2)) if use_native_fp16() else
                     (queries.float() @ keys.float().transpose(-1, -2)).half())
            exponent = exp_candidate(score)
            denominator = denominator + exponent.float().sum(-1, keepdim=True)
            weighted = quantize_e4(exponent)
            if use_native_fp16():
                numerator = native_gemm(weighted, values, numerator)
            else:
                for first in range(0, keys.shape[-2], 32):
                    numerator = (numerator.float() + weighted[..., first:first+32].float() @ values[..., first:first+32, :].float()).half()
        scale = denominator.clamp_min(6.198883056640625e-5).reciprocal().half()
        chunks.append(quantize_e4((numerator * scale).half()))
    return torch.cat(chunks, dim=-2)


class RecoveredVit1024(nn.Module):
    native_graph_complete = False
    image_quality_verified = False
    gpu_verified = False
    layout_status = 'STATIC_RECOVERY_CPU_REFERENCE_PENDING_INTEGRATED_VALIDATION'

    def __init__(self, **records):
        super().__init__()
        if set(records) != set(RECORD_SIZES):
            raise ValueError('five explicitly named ViT records required')
        for name, size in RECORD_SIZES.items():
            if len(records[name]) != size:
                raise ValueError(f'wrong ViT {name} record size')
        self.record_sha256 = {name: hashlib.sha256(raw).hexdigest().upper() for name, raw in records.items()}
        parameter(self, 'expand', decode_e4(records['expand'])[matrix_indices(1024, 4096)])
        parameter(self, 'contract', decode_e4(records['contract'])[matrix_indices(4096, 1024)])
        parameter(self, 'qkv', decode_e4(records['qkv'])[qkv_indices()])
        parameter(self, 'q_scale', torch.tensor(struct.unpack_from('<32f', records['qkv'], 0)))
        parameter(self, 'project', decode_e4(records['projection'])[matrix_indices(1024, 1024)])
        for name, key, offset in [('ffn_scale', 'contract', 4194304), ('attention_scale', 'projection', 1048576)]:
            halves = torch.frombuffer(bytearray(records[key]), dtype=torch.float16)
            parameter(self, name, halves[offset // 2:])
        self.register_buffer('perm1024', channel_permutation(1024))
        self.register_buffer('perm4096', channel_permutation(4096))
        # The 2-byte layer3 record is an unused ABI placeholder: original attention
        # entry does not load parameter+32. Preserve its hash, do not invent a weight.

    def forward_ffn(self, x):
        self.require_input(x)
        hidden = quantized_cubic_silu(chunked_linear(x[..., self.perm1024], self.expand))
        return quantize_e4(chunked_linear(hidden[..., self.perm4096], self.contract, (x * self.ffn_scale).half()))

    def forward(self, x, *, query_chunk=32, key_chunk=128):
        post_ffn = self.forward_ffn(x)
        projected = chunked_linear(post_ffn[..., self.perm1024], self.qkv)
        q, k, v = [t.reshape(t.shape[0], t.shape[1], 32, 32).transpose(1, 2) for t in projected.chunk(3, -1)]
        q = quantize_e4((normalize_qk(q) * self.q_scale[None, :, None, None]).half())
        k, v = quantize_e4(normalize_qk(k)), quantize_e4(v)
        attended = global_attention(q, k, v, query_chunk=query_chunk, key_chunk=key_chunk,
                                    max_score_elements=4194304 if query_chunk in (512,1024) else 1048576)
        attended = attended.transpose(1, 2).reshape(x.shape)
        return quantize_e4(chunked_linear(attended[..., self.perm1024], self.project, (post_ffn * self.attention_scale).half()))

    def require_input(self, x):
        if x.ndim != 3 or x.shape[1] <= 0 or x.shape[-1] != 1024 or x.dtype != torch.float16:
            raise ValueError('expected logical FP16 [batch,tokens,1024]')
        if x.device != self.expand.device:
            raise ValueError('model and input must share a device')


class RecoveredVitBottleneck1024(nn.Module):
    """All eight original ViT blocks, no RTX boundary injection between blocks."""
    native_graph_complete = False
    image_quality_verified = False
    gpu_verified = False

    def __init__(self, blocks):
        super().__init__()
        if set(blocks) != set(range(31, 39)) or not all(isinstance(m, RecoveredVit1024) for m in blocks.values()):
            raise ValueError('all eight explicitly numbered original ViT blocks required')
        if len({id(m) for m in blocks.values()}) != 8:
            raise ValueError('cannot substitute one repeated model for distinct original blocks')
        self.blocks = nn.ModuleList([blocks[b] for b in range(31, 39)])

    def forward(self, x, **chunks):
        for block in self.blocks:
            x = block(x, **chunks)
        return x
