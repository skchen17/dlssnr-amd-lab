"""Candidate shared 4x4-cell layout adapter; RTX family tests gate its use."""
import torch
from torch import nn
from native_swin_torch import packed_a
from native_grouped_ffn import RecoveredLogicalSwin


def gather_packed(packed, width, height, channels, ox, oy):
    if (channels not in (32, 64, 128, 256, 512) or width <= 0 or height <= 0
            or width % 4 or height % 4 or ox not in (-4, 0) or oy not in (-4, 0)):
        raise ValueError('unrecovered feature geometry')
    if packed.numel() != width * height * channels:
        raise ValueError('incorrect packed feature size')
    from native_c32_layout_fusion import active_c32_layout,Mapping
    fusion=active_c32_layout()
    if fusion is not None and channels==32:
        mapping=Mapping(width,height,ox,oy)
        return fusion.apply(packed,mapping),mapping
    gx, gy = (width - ox + 7) // 8, (height - oy + 7) // 8
    cta = torch.arange(gx * gy, device=packed.device)[:, None]
    cell = torch.arange(4, device=packed.device)[None, :]
    tx, ty = (cta % gx * 8 + ox) // 4 + cell % 2, (cta // gx * 8 + oy) // 4 + cell // 2
    valid = (tx >= 0) & (tx < width // 4) & (ty >= 0) & (ty < height // 4)
    index = (ty * (width // 4) + tx).clamp(0, width // 4 * (height // 4) - 1)
    result = packed.reshape(-1, 16 * channels)[index].masked_fill(~valid[..., None], 0)
    return result.reshape(-1, 64 * channels), (index, valid)


def logical_indices(channels):
    group = torch.arange(4)[:, None, None]
    row = torch.arange(16)[None, :, None]
    channel = torch.arange(channels)[None, None, :]
    base = group * (16 * channels) + channel // 32 * 512
    a = base + packed_a(0, row, channel % 32)
    c = channel % 32
    mma = c // 16 * 2 + c % 16 // 8
    pairs = torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])
    element = (row >= 8) * 2 + (c & 1)
    operation = pairs[mma * 2 + element // 2]
    residual = base + (row % 8 * 4 + c % 8 // 2) * 16 + operation * 2 + (element & 1)
    return a.reshape(64, channels).long(), residual.reshape(64, channels).long()


def scatter_packed(values, mapping, width, height):
    channels = values.shape[-1]
    from native_c32_layout_fusion import active_c32_layout,Mapping
    if isinstance(mapping,Mapping):
        fusion=active_c32_layout()
        if fusion is None or channels!=32 or (width,height)!=(mapping.width,mapping.height):raise ValueError('C32 mapping scope mismatch')
        return fusion.apply(values,mapping,scatter=True)
    index, valid = mapping
    physical = torch.arange(16 * channels, device=values.device)
    head, lane, element = physical // 512, physical % 512 // 16, physical % 16
    row = lane // 4 + element % 8 // 4 * 8
    col = head * 32 + lane % 4 * 2 + element % 4 // 2 * 8 + element // 8 * 16 + (element & 1)
    packed = values.reshape(-1, 4, 16, channels)[:, :, row, col]
    result = torch.zeros((height // 4 * (width // 4), 16 * channels), device=values.device, dtype=values.dtype)
    result[index[valid]] = packed[valid]
    return result.flatten()


class RecoveredPackedSwin(nn.Module):
    native_graph_complete = False
    rtx_quality_verified = False

    def __init__(self, raw, *, record_kind):
        super().__init__()
        self.block = RecoveredLogicalSwin(raw, record_kind=record_kind)
        self.channels = self.block.ffn.channels
        a, residual = logical_indices(self.channels)
        self.register_buffer('a_index', a)
        self.register_buffer('residual_index', residual)

    def forward(self, windows):
        if windows.ndim != 2 or windows.shape[1] != 64 * self.channels or windows.dtype != torch.float16:
            raise ValueError('incorrect packed window shape/dtype')
        from native_matrix_fusion import active_matrix_fusion
        matrix=active_matrix_fusion()
        family=f'c{self.channels}_ffn'
        if matrix is not None and self.channels in (64,128,256) and family in matrix.modules:
            post=matrix.wide_ffn(self,windows)
            attention_family=f'c{self.channels}_attention'
            bounded_family=f'c{self.channels}_attention_bounded'
            if bounded_family in matrix.modules:return matrix.wide_attention_bounded(self.block.attention,post)
            return matrix.wide_attention(self.block.attention,post) if attention_family in matrix.modules else self.block.attention(post)
        attention_family=f'c{self.channels}_attention'
        if matrix is not None and self.channels in (64,128) and attention_family in matrix.modules:
            post=self.block.ffn(windows[:,self.a_index],windows[:,self.residual_index])
            return matrix.wide_attention(self.block.attention,post)
        bounded_family=f'c{self.channels}_attention_bounded'
        if matrix is not None and self.channels in (64,128) and bounded_family in matrix.modules:
            post=self.block.ffn(windows[:,self.a_index],windows[:,self.residual_index])
            return matrix.wide_attention_bounded(self.block.attention,post)
        if matrix is not None and self.channels==32 and 'c32_ffn' in matrix.modules:
            post=matrix.c32_ffn(self,windows)
            if 'c32_attention_staged' in matrix.modules:return matrix.c32_attention_staged(self.block.attention,post)
            if 'c32_attention_core' in matrix.modules:return matrix.c32_attention_core(self.block.attention,post)
            if 'c32_attention_bounded' in matrix.modules:return matrix.c32_attention_bounded(self.block.attention,post)
            return matrix.c32_attention(self.block.attention,post) if 'c32_attention' in matrix.modules else self.block.attention(post)
        if matrix is not None and self.channels==32 and 'c32_attention_bounded' in matrix.modules:
            return matrix.c32_attention_bounded(self.block.attention,self.block.ffn(windows[:,self.a_index],windows[:,self.residual_index]))
        if matrix is not None and self.channels==32 and 'c32_attention_staged' in matrix.modules:
            return matrix.c32_attention_staged(self.block.attention,self.block.ffn(windows[:,self.a_index],windows[:,self.residual_index]))
        if matrix is not None and self.channels==32 and 'c32_attention_core' in matrix.modules:
            return matrix.c32_attention_core(self.block.attention,self.block.ffn(windows[:,self.a_index],windows[:,self.residual_index]))
        if matrix is not None and self.channels==32 and 'c32_attention' in matrix.modules:
            return matrix.c32_attention(self.block.attention,self.block.ffn(windows[:,self.a_index],windows[:,self.residual_index]))
        return self.block(windows[:, self.a_index], windows[:, self.residual_index])
