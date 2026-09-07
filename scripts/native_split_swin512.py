"""Original four-record 512-channel split-Swin mathematical candidate.

Native tensor operations only. CPU execution is for reference/tests; deployment
must explicitly require ROCm. No translated instructions or captured activations
are used by this module. Tensor adapters and image-quality acceptance are separate.
"""
import hashlib
import struct
import torch
from torch import nn
from native_grouped_ffn import cubic_silu
from native_swin_torch import decode_e4, packed_b, permute32, quantize_e4
from native_window_attention import projection_indices, window_attention, window_attention_prepared
from native_execution_policy import use_native_fp16, native_gemm


RECORD_SIZES = {'ffwd': 524288, 'ffwd_projection': 263168,
                'attention': 917568, 'attention_projection': 263168}


def parameter(module, name, value):
    value = value.clone().half()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'nonfinite original parameter {name}')
    module.register_parameter(name, nn.Parameter(value, requires_grad=False))


def check_record(raw, kind):
    if len(raw) != RECORD_SIZES[kind]:
        raise ValueError(f'incorrect {kind} record size')


def grouped_indices(expand):
    """Eight independent 64->256->64 MLPs after a full 512->512 projection.

    Static FFWD addresses: preprojection [0,262144), group expansion
    [262144,393216), contraction [393216,524288). Group stride16384.
    Expansion K32 stride8192; contraction K32 stride2048.
    """
    rows, cols, base = (64, 256, 262144) if expand else (256, 64, 393216)
    group = torch.arange(8)[:, None, None]
    k, n = torch.arange(rows)[None, :, None], torch.arange(cols)[None, None, :]
    tile = base + group * 16384 + k // 32 * (cols * 32) + n // 16 * 512
    return packed_b(tile, k % 32, n % 16).long()


def chunked_linear(x, weight, seed=None):
    """FP8 inputs/weights and explicit FP16 boundaries per K32 partial sum."""
    x, weight = quantize_e4(x), quantize_e4(weight)
    if use_native_fp16():
        return native_gemm(x, weight, seed)
    result = seed
    for k in range(0, weight.shape[-2], 32):
        value = x[..., k:k+32].float() @ weight[..., k:k+32, :].float()
        result = value.half() if result is None else (result.float() + value).half()
    return result


def channel_permutation(channels):
    return (torch.arange(channels // 32)[:, None] * 32 + permute32(torch.arange(32))).flatten().long()


class SplitFFWD512(nn.Module):
    def __init__(self, raw):
        super().__init__()
        check_record(raw, 'ffwd')
        e4 = decode_e4(raw)
        parameter(self, 'preproject', e4[projection_indices(512, 0)])
        parameter(self, 'expand', e4[grouped_indices(True)])
        parameter(self, 'contract', e4[grouped_indices(False)])
        self.register_buffer('perm64', channel_permutation(64))
        self.register_buffer('perm256', channel_permutation(256))

    def forward(self, matrix_a_features):
        require_features(matrix_a_features, self.preproject.device)
        projected = chunked_linear(matrix_a_features, self.preproject)
        from native_matrix_fusion import active_matrix_fusion
        matrix=active_matrix_fusion()
        if matrix is not None and 'c512_ffn' in matrix.modules:
            return matrix.c512_group_ffn(self,projected)
        grouped = projected.reshape(-1, 64, 8, 64).transpose(1, 2)
        hidden = chunked_linear(grouped[..., self.perm64], self.expand)
        hidden = cubic_silu(hidden)
        contracted = chunked_linear(hidden[..., self.perm256], self.contract)
        return quantize_e4(contracted.transpose(1, 2).reshape(-1, 64, 512))


def require_features(value, device):
    if value.ndim != 3 or value.shape[1:] != (64, 512) or value.dtype != torch.float16:
        raise ValueError('expected logical FP16 [windows,64,512]')
    if value.device != device:
        raise ValueError('all tensors must share the model device')


class SplitProjection512(nn.Module):
    def __init__(self, raw, *, kind):
        super().__init__()
        if kind not in ('ffwd_projection', 'attention_projection'):
            raise ValueError('projection record kind must be explicit')
        check_record(raw, kind)
        parameter(self, 'weight', decode_e4(raw)[projection_indices(512, 0)])
        halves = torch.frombuffer(bytearray(raw), dtype=torch.float16)
        parameter(self, 'residual_scale', halves[262144 // 2:])
        self.register_buffer('permutation', channel_permutation(512))

    def unquantized(self, values, residual):
        require_features(values, self.weight.device)
        require_features(residual, self.weight.device)
        if values.shape != residual.shape:
            raise ValueError('projection and residual window counts differ')
        seed = (residual * self.residual_scale).half()
        return chunked_linear(values[..., self.permutation], self.weight, seed)

    def forward(self, values, residual):
        return quantize_e4(self.unquantized(values, residual))


class SplitAttention512(nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.channels = 512
        check_record(raw, 'attention')
        parameter(self, 'qkv', decode_e4(raw)[projection_indices(512, 0, qkv=True)])
        halves = torch.frombuffer(bytearray(raw), dtype=torch.float16)
        query, key = torch.arange(64)[:, None], torch.arange(64)[None, :]
        row, kb = query % 16, key // 8
        lane = row % 8 * 4 + key % 8 // 2
        element = (row >= 8) * 2 + (key & 1)
        index = 786432 + query // 16 * 2048 + kb // 2 * 512 + lane * 16 + (kb & 1) * 8 + element * 2
        index = index + torch.arange(16)[:, None, None] * 8192
        parameter(self, 'position_bias', halves[index // 2])
        parameter(self, 'q_scale', torch.tensor(struct.unpack_from('<16f', raw, 917504)))
        self.register_buffer('permutation', channel_permutation(512))

    def forward(self, post_ffn):
        require_features(post_ffn, self.qkv.device)
        projected = chunked_linear(post_ffn[..., self.permutation], self.qkv)
        from native_matrix_fusion import active_matrix_fusion
        matrix = active_matrix_fusion()
        if matrix is not None and 'c512_attention_norm' in matrix.modules:
            q, k, v = matrix.wide_attention_norm(self, projected)
            result = window_attention_prepared(q, k, v, self.position_bias)
        else:
            q, k, v = [x.reshape(-1, 64, 16, 32).transpose(1, 2) for x in projected.chunk(3, -1)]
            result = window_attention(q, k, v, self.q_scale, self.position_bias)
        return quantize_e4(result.transpose(1, 2).reshape(-1, 64, 512))


class RecoveredSplitSwin512(nn.Module):
    """Four-stage tensor composition. Not a promoted complete-image model."""
    native_graph_complete = False
    image_quality_verified = False
    gpu_verified = False

    def __init__(self, *, ffwd, ffwd_projection, attention, attention_projection):
        super().__init__()
        self.record_sha256 = {name: hashlib.sha256(raw).hexdigest().upper() for name, raw in
                              [('ffwd', ffwd), ('ffwd_projection', ffwd_projection),
                               ('attention', attention), ('attention_projection', attention_projection)]}
        self.ffwd = SplitFFWD512(ffwd)
        self.ffwd_projection = SplitProjection512(ffwd_projection, kind='ffwd_projection')
        self.attention = SplitAttention512(attention)
        self.attention_projection = SplitProjection512(attention_projection, kind='attention_projection')

    def unquantized(self, matrix_a_features, residual_features):
        ffwd = self.ffwd(matrix_a_features)
        post_ffn = self.ffwd_projection(ffwd, residual_features)
        attended = self.attention(post_ffn)
        return self.attention_projection.unquantized(attended, post_ffn)

    def forward(self, matrix_a_features, residual_features):
        return quantize_e4(self.unquantized(matrix_a_features, residual_features))
