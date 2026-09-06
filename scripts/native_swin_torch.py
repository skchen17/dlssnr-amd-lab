"""Differentiable ROCm candidate for the recovered 1h/32 family ONLY.

Ports the documented DXIL mathematical prototype, not NVIDIA instructions.
The DXIL prototype failed full-network quality: isolated passes MUST NOT promote
this module to a complete model. Window tensors use the recovered packed layout.
CPU is allowed for unit-test/reference execution, never the deployment backend.
"""
from __future__ import annotations

import struct
import torch
from torch import nn
from native_fusion_policy import active_fusions


def permute32(k):
    return k // 16 * 16 + 2 * (k % 16 // 4) + (k & 1) + ((k & 2) != 0) * 8


def packed_a(group, row, k):
    return group * 512 + (row % 8 * 4 + k % 16 // 4) * 16 + k % 4 + (row >= 8) * 4 + (k >= 16) * 8


def packed_b(tile, k, col):
    return tile + (col % 8 * 4 + k % 16 // 4) * 16 + col // 8 * 8 + k % 4 + (k >= 16) * 4


def quantize_e4(x):
    """Finite saturating E4M3 rounding; STE derivative for later micro-tuning."""
    fusion=active_fusions()
    if fusion is not None:return fusion.quantize(x)
    rounded = x.clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype)
    return x + (rounded - x).detach() if x.requires_grad else rounded


def quantized_gather(x,index):
    from native_head_fusion import active_head_fusion
    fusion=active_head_fusion()
    if fusion is not None and fusion.in_block:return fusion.gather_e4(x,index)
    return quantize_e4(x[...,index])


def decode_e4(raw: bytes):
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float8_e4m3fn).half()


def encode_e4(x):
    return x.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def gather_windows(packed, width, height, ox, oy):
    if width <= 0 or height <= 0 or width % 8 or height % 8 or ox not in (0, -4) or oy not in (0, -4):
        raise ValueError('recovered family needs multiple-of-eight features and origin 0/-4')
    if packed.numel() != width * height * 32:
        raise ValueError('wrong feature storage size')
    gx, gy = (width - ox + 7) // 8, (height - oy + 7) // 8
    cta = torch.arange(gx * gy, device=packed.device)[:, None]
    cell = torch.arange(4, device=packed.device)[None, :]
    tx = (cta % gx * 8 + ox) // 4 + cell % 2
    ty = (cta // gx * 8 + oy) // 4 + cell // 2
    valid = (tx >= 0) & (tx < width // 4) & (ty >= 0) & (ty < height // 4)
    index = (ty * (width // 4) + tx).clamp(0, width // 4 * (height // 4) - 1)
    windows = packed.reshape(-1, 512)[index].masked_fill(~valid[..., None], 0).reshape(-1, 2048)
    return windows, (index, valid)


def scatter_windows(values, mapping, width, height):
    index, valid = mapping
    lane = torch.arange(512, device=values.device) // 16
    element = torch.arange(512, device=values.device) % 16
    row = lane // 4 + element % 8 // 4 * 8
    col = lane % 4 * 2 + element % 4 // 2 * 8 + element // 8 * 16 + (element & 1)
    packed = values.reshape(-1, 4, 16, 32)[:, :, row, col]
    result = torch.zeros((height // 4 * (width // 4), 512), device=values.device, dtype=values.dtype)
    result[index[valid]] = packed[valid]
    return result.reshape(-1)


class RecoveredSwin32(nn.Module):
    native_graph_complete = False

    def __init__(self, packed_weights: bytes, *, record_kind='swin32'):
        super().__init__()
        self.record_kind=record_kind
        self.inference_cache_enabled=False
        self._inference_weights={}
        sizes = {'swin32': 20672, 'head32': 21808}
        if record_kind not in sizes or len(packed_weights) != sizes[record_kind]:
            raise ValueError('wrong recovered weight record kind/size')
        shift = 112 if record_kind == 'head32' else 0
        e4 = decode_e4(packed_weights)
        def parameter(name, value):
            if not torch.isfinite(value).all():
                raise ValueError(f'nonfinite weight {name}')
            self.register_parameter(name, nn.Parameter(value.clone().half(), requires_grad=False))
        k, col = torch.arange(32)[:, None], torch.arange(32)[None, :]
        def matrix(base):
            return e4[packed_b(base + col // 16 * 512, k, col % 16)]
        parameter('expand', torch.stack([matrix(p * 1024) for p in range(4)]))
        parameter('contract', torch.stack([matrix(4096 + p * 1024) for p in range(4)]))
        parameter('qkv', torch.cat([matrix(8288 + shift + p * 1024) for p in range(3)], dim=1))
        parameter('project', matrix(19568 + shift))
        halves = torch.frombuffer(bytearray(packed_weights), dtype=torch.float16)
        column = torch.arange(32)
        # Mirrors ResidualSeed's scale selection; channels remain logical here.
        mma = column // 16 * 2 + column % 16 // 8
        scale_index = 8208 + (mma * 4 + column % 8 // 2) * 4 + (column & 1) * 2
        parameter('ffn_scale', halves[scale_index // 2])
        parameter('attention_scale', halves[(20592 + shift) // 2:(20592 + shift) // 2 + 32])
        parameter('q_scale', torch.tensor(struct.unpack_from('<f', packed_weights, 19552 + shift)[0]))
        query, key = torch.arange(64)[:, None], torch.arange(64)[None, :]
        row, kb = query % 16, key // 8
        lane = row % 8 * 4 + key % 8 // 2
        element = (row >= 8) * 2 + (key & 1)
        offset = 11360 + shift + query // 16 * 2048 + kb // 2 * 512 + lane * 16 + (kb & 1) * 8 + element * 2
        parameter('position_bias', halves[offset // 2])
        group = torch.arange(4)[:, None, None]
        row = torch.arange(16)[None, :, None]
        channel = torch.arange(32)[None, None, :]
        self.register_buffer('a_index', packed_a(group, row, channel).reshape(64, 32).long())
        mma = channel // 16 * 2 + channel % 16 // 8
        pairs = torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])
        element = (row >= 8) * 2 + (channel & 1)
        operation = pairs[mma * 2 + element // 2]
        residual = group * 512 + (row % 8 * 4 + channel % 8 // 2) * 16 + operation * 2 + (element & 1)
        self.register_buffer('residual_index', residual.reshape(64, 32).long())
        self.register_buffer('permutation', permute32(torch.arange(32)).long())

    def _apply(self,fn,recurse=True):
        self._inference_weights.clear()
        return super()._apply(fn,recurse=recurse)

    def train(self,mode=True):
        self._inference_weights.clear()
        return super().train(mode)

    def _weight(self,name,*,part=None,fp32=False,quantized=True):
        """Cache only frozen eval/no-grad transforms; never detach trainable math.

        Device/dtype/storage/version changes invalidate entries. No .data writes
        are used or supported as a way to mutate a frozen parameter behind PyTorch.
        Cache is not model state and is cleared on device moves/training changes.
        """
        parameter=getattr(self,name)
        enabled=self.inference_cache_enabled and not self.training and not torch.is_grad_enabled() and not parameter.requires_grad
        key=(name,part,fp32,quantized)
        if enabled:
            identity=(parameter.data_ptr(),parameter._version,parameter.device,parameter.dtype)
            old=self._inference_weights.get(key)
            if old is not None and old[0] is parameter and old[1]==identity:return old[2]
        value=parameter if part is None else parameter[part]
        if quantized:value=quantize_e4(value)
        if fp32:value=value.float()
        if enabled:self._inference_weights[key]=(parameter,identity,value)
        return value

    def forward_ffn(self, raw_windows):
        """Expose the logical FFN boundary for native-family cross checks."""
        from native_matrix_fusion import active_matrix_fusion
        matrix=active_matrix_fusion()
        if matrix is not None and self.record_kind=='head32' and 'head_ffn' in matrix.modules:
            return matrix.head_ffn(self,raw_windows)
        if raw_windows.ndim != 2 or raw_windows.shape[1] != 2048 or raw_windows.dtype != torch.float16:
            raise ValueError('expected [windows,2048] decoded FP16 packed features')
        a = quantized_gather(raw_windows, self.a_index)
        x = (a[:, None] @ self._weight('expand')).half()
        from native_grouped_ffn import quantized_cubic_silu
        hidden = quantized_cubic_silu(x)
        first_output = (raw_windows[:, self.residual_index] * self.ffn_scale).half()
        from native_head_fusion import active_head_fusion
        fusion=active_head_fusion()
        for p in range(4):
            # Preserve per-32-channel accumulation boundaries, not per-instruction rounding.
            product = hidden[:, p, :, self.permutation].float() @ self._weight('contract',part=p,fp32=True)
            first_output = fusion.add(first_output,product) if fusion is not None and fusion.epilogue_active else (first_output.float() + product).half()
        return first_output

    def forward(self, raw_windows):
        from native_head_fusion import active_head_fusion
        fusion=active_head_fusion()
        epilogue=fusion is not None and fusion.epilogue_active
        first_output = self.forward_ffn(raw_windows)
        from native_matrix_fusion import active_matrix_fusion
        matrix=active_matrix_fusion()
        if matrix is not None and self.record_kind=='head32' and 'head_attention_bounded' in matrix.modules:
            return matrix.head_attention_bounded(self,first_output)
        if matrix is not None and self.record_kind=='head32' and 'head_attention' in matrix.modules:
            projection=matrix.head_qkv(self,first_output)
        else:projection = (quantized_gather(first_output,self.permutation) @ self._weight('qkv')).half()
        q, k, v = projection.chunk(3, dim=-1)
        def norm(t):
            squares = (t * t).half()
            while squares.shape[-1] > 1:
                squares = (squares[..., 0::2] + squares[..., 1::2]).half()
            inv = squares.float().clamp_min(6.198883056640625e-5).rsqrt().half()
            return (t * inv).half()
        qkv_fused=fusion is not None and fusion.qkv_active
        if qkv_fused:q,k,v=fusion.qkv(projection,self.q_scale)
        else:
            q = quantize_e4((norm(q) * self.q_scale).half())
            k = quantize_e4(norm(k))
        if matrix is not None and self.record_kind=='head32' and 'head_attention' in matrix.modules:
            scores=matrix.head_scores(self,q,k)
        else:scores = (q.float() @ k.float().transpose(-1, -2) + self._weight('position_bias',fp32=True,quantized=False)).half()
        if matrix is not None and self.record_kind=='head32' and 'head_softmax' in matrix.modules:
            probability=matrix.head_probability(scores)
        else:probability = quantize_e4(torch.softmax(scores.float(), dim=-1).half())
        if not qkv_fused:v = quantize_e4(v)
        if matrix is not None and self.record_kind=='head32' and 'head_attention' in matrix.modules:
            attention=matrix.head_pv(probability,v)
        else:
            attention = (probability[..., :32].float() @ v[..., :32, :].float()).half()
            second=probability[..., 32:].float() @ v[..., 32:, :].float()
            attention = fusion.add(attention,second) if epilogue else (attention.float()+second).half()
            del second
        if matrix is not None and self.record_kind=='head32' and 'head_output' in matrix.modules:
            return matrix.head_project(self,first_output,attention)
        if not epilogue:seed=(first_output*self.attention_scale).half()
        projected = quantized_gather(attention,self.permutation).float() @ self._weight('project',fp32=True)
        if epilogue:return fusion.add(first_output,projected,self.attention_scale)
        return (seed.float() + projected).half()
