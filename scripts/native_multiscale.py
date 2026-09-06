"""Native learned encoder downsampling and global feature layout adapters.

Downsample returns BOTH the encoder skip and pooled/projected next-scale feature.
No independent image tiles or resized image substitutes are introduced.
"""
import torch
import native_compact_layout
from torch import nn
from native_packed_swin import RecoveredPackedSwin, gather_packed, scatter_packed
from native_sequence_layout import sequence_indices
from native_vit1024 import matrix_indices
from native_window_attention import LAYOUTS
from native_split_swin512 import parameter, channel_permutation, chunked_linear
from native_swin_torch import decode_e4, quantize_e4


DOWNSAMPLE_RECORDS = {'swin32': 22720, 'swin64': 69936, 'swin128': 229936, 'swin256': 820288}


def image_indices(width, height, channels, device):
    if width <= 0 or height <= 0 or width % 4 or height % 4:
        raise ValueError('packed image needs positive four-aligned dimensions')
    y, x = torch.arange(height, device=device)[:, None], torch.arange(width, device=device)[None, :]
    cell = y // 4 * (width // 4) + x // 4
    row = y % 4 * 4 + x % 4
    local = sequence_indices(16, channels, device=device)
    return cell[..., None] * (16 * channels) + local[row]


def unpack_image(packed, width, height, channels):
    if native_compact_layout.enabled():return native_compact_layout.unpack(packed,width,height,channels)
    if packed.ndim != 1 or packed.numel() != width * height * channels:
        raise ValueError('incorrect packed image storage')
    return packed[image_indices(width, height, channels, packed.device)]


def pack_image(logical):
    if native_compact_layout.enabled():return native_compact_layout.pack(logical)
    if logical.ndim != 3:
        raise ValueError('expected HWC logical feature image')
    height, width, channels = logical.shape
    index = image_indices(width, height, channels, logical.device)
    result = torch.empty(logical.numel(), dtype=logical.dtype, device=logical.device)
    result[index] = logical
    return result


def average_pool2x2(x):
    if x.ndim != 3 or x.shape[0] % 2 or x.shape[1] % 2 or x.dtype != torch.float16:
        raise ValueError('pool requires even-sized FP16 HWC image')
    a = (x[0::2, 0::2] + x[0::2, 1::2]).half()
    b = (x[1::2, 0::2] + x[1::2, 1::2]).half()
    return ((a + b).half() * .25).half()


def pack_planar16(logical):
    if logical.ndim != 3 or logical.shape[-1] % 16:
        raise ValueError('expected HWC feature image with channels divisible by16')
    h, w, c = logical.shape
    return logical.reshape(h, w, c//16, 16).permute(2, 0, 1, 3).contiguous().flatten()


def unpack_planar16(packed, width, height, channels):
    if width <= 0 or height <= 0 or channels <= 0 or channels % 16 or packed.numel() != width*height*channels:
        raise ValueError('incorrect planar16 view')
    return packed.reshape(channels//16, height, width, 16).permute(1, 2, 0, 3).reshape(height, width, channels)


def outview_channels(device=None):
    """One store joins pairs from columns N0 and N8 of the same MMA lane."""
    return torch.tensor([0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15], device=device)


def pack_outview(logical):
    planar = pack_planar16(logical)
    return planar.reshape(-1, 16)[:, outview_channels(logical.device)].flatten()


def unpack_outview(packed, width, height, channels):
    if width <= 0 or height <= 0 or channels <= 0 or channels % 16 or packed.ndim != 1 or packed.numel() != width*height*channels:
        raise ValueError('incorrect outview storage')
    inverse = torch.argsort(outview_channels(packed.device))
    planar = packed.reshape(-1, 16)[:, inverse].flatten()
    return unpack_planar16(planar, width, height, channels)


class RecoveredOutviewSwin(nn.Module):
    """Native input-view adapter plus a plain Swin, no captured interior tensors."""
    native_graph_complete = False

    def __init__(self, raw, *, record_kind):
        super().__init__()
        self.block = RecoveredPackedSwin(raw, record_kind=record_kind)
        self.channels = self.block.channels

    def _forward_packed(self, packed, width, height, ox, oy, window_batch):
        if packed.ndim != 1 or packed.numel() != width*height*self.channels:
            raise ValueError('resident packed feature size mismatch')
        windows, mapping = gather_packed(packed, width, height, self.channels, ox, oy)
        from native_grid_policy import run_windows
        values = run_windows(self.block, windows, window_batch, f'c{self.channels}')
        return quantize_e4(scatter_packed(values, mapping, width, height))

    def forward_resident(self, source, width, height, ox, oy, *, window_batch=12):
        if window_batch <= 0 or source.dtype != torch.float16:
            raise ValueError('positive window batch and decoded FP16 resident source required')
        return self._forward_packed(source,width,height,ox,oy,window_batch)

    def forward(self, source, width, height, ox, oy, *, window_batch=12):
        if window_batch <= 0 or source.dtype != torch.float16:
            raise ValueError('positive window batch and decoded FP16 source required')
        packed = pack_image(unpack_outview(source, width, height, self.channels))
        return self._forward_packed(packed,width,height,ox,oy,window_batch)


class RecoveredDownsampleSwin(nn.Module):
    native_graph_complete = False
    gpu_verified = False
    image_quality_verified = False
    output_layout_verified = False

    def __init__(self, raw, *, record_kind):
        super().__init__()
        if record_kind not in DOWNSAMPLE_RECORDS or len(raw) != DOWNSAMPLE_RECORDS[record_kind]:
            raise ValueError('explicit downsample record kind/size required')
        layout = LAYOUTS[record_kind]
        self.channels = c = layout.channels
        # Standard block trailing alignment is unused; a boundary record places
        # its learned pool projection there. Preserve raw original byte order.
        self.block = RecoveredPackedSwin(raw[:layout.record_bytes], record_kind=record_kind)
        tail = layout.residual_scale + 2*c
        parameter(self, 'pool_project', decode_e4(raw)[matrix_indices(c, 2*c, tail)])
        self.register_buffer('permutation', channel_permutation(c))

    def forward(self, packed, width, height, ox, oy, *, window_batch=12, capture_layouts=True):
        if window_batch <= 0 or width % 8 or height % 8:
            raise ValueError('downsample needs positive batch and eight-aligned source geometry')
        if packed.device != self.pool_project.device or packed.dtype != torch.float16:
            raise ValueError('decoded FP16 features must share the model device')
        windows, mapping = gather_packed(packed, width, height, self.channels, ox, oy)
        from native_grid_policy import run_windows
        projected = run_windows(self.block, windows, window_batch, f'c{self.channels}')
        unquantized = scatter_packed(projected, mapping, width, height)
        from native_transition_fusion import active_transition_fusion
        fusion=active_transition_fusion()
        if fusion is not None and fusion.encoder_enabled:
            return fusion.encoder(unquantized,width,height,self.channels,self.permutation,self.pool_project,
                                  capture_layouts=capture_layouts)
        pooled = average_pool2x2(unpack_image(unquantized, width, height, self.channels))
        down = quantize_e4(chunked_linear(pooled[..., self.permutation], self.pool_project))
        resident=pack_image(down)
        result={'skip':quantize_e4(unquantized),'resident':resident,
                'source_size':(width,height),'target_size':(width//2,height//2)}
        if capture_layouts:
            # Compatibility/debug aliases. `downsampled` is the same tensor as
            # resident, not a second allocation; outview is explicitly materialized.
            result['downsampled']=resident
            result['captured_outview']=pack_outview(down)
        return result
