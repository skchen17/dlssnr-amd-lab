"""Original learned 2x decoder projection/skip fusion plus Swin tensor candidate.

Whole-feature coordinates, not image tiles. Expands each projected low-resolution
feature to its 2x2 children, then adds the actual encoder skip with original scale.
Static layout recovery and captured-input validation are separate from full-image
quality. No synthetic/zero skip is accepted as a substitute by the validator.
"""
import hashlib
import torch
from torch import nn
from native_multiscale import unpack_outview, unpack_image, pack_image
from native_packed_swin import RecoveredPackedSwin, gather_packed, scatter_packed
from native_split_swin512 import parameter, channel_permutation, chunked_linear
from native_swin_torch import decode_e4, quantize_e4
from native_vit1024 import matrix_indices
from native_window_attention import LAYOUTS

RECORD_SIZES = {'swin32': 22784, 'swin64': 70048, 'swin128': 230176, 'swin256': 820784}


def sections(kind):
    if kind not in RECORD_SIZES:
        raise ValueError('explicit recovered upsample family required')
    c = LAYOUTS[kind].channels
    prefix = 8192 if c == 32 else 5*c*c+128*c
    projection_end = prefix+2*c*c
    ffn_scale = projection_end+16 if c == 32 else projection_end
    skip_scale = 10336 if c == 32 else ffn_scale+2*c
    qkv = skip_scale+2*c
    return c, prefix, ffn_scale, skip_scale, qkv


def plain_record(raw, kind):
    c, prefix, ffn_scale, skip_scale, qkv = sections(kind)
    if len(raw) != RECORD_SIZES[kind]:
        raise ValueError('incorrect original upsample record size')
    # Only unused alignment fields are zeroed in this private decode adapter.
    # Every learned FFN/attention/scale value comes from the original record.
    core = raw[:prefix]+bytes(16)+raw[ffn_scale:ffn_scale+2*c]+bytes(16)+raw[qkv:]
    if len(core) != LAYOUTS[kind].record_bytes:
        raise ValueError('compacted learned record sections differ')
    return core


class RecoveredUpsampleSwin(nn.Module):
    native_graph_complete = False
    gpu_verified = False
    image_quality_verified = False

    def __init__(self, raw, *, record_kind):
        super().__init__()
        c,prefix,_,skip_scale,_ = sections(record_kind)
        self.channels = c
        self.original_record_sha256 = hashlib.sha256(raw).hexdigest().upper()
        self.block = RecoveredPackedSwin(plain_record(raw,record_kind),record_kind=record_kind)
        parameter(self,'project',decode_e4(raw)[matrix_indices(2*c,c,prefix)])
        halves = torch.frombuffer(bytearray(raw),dtype=torch.float16)
        parameter(self,'skip_scale',halves[skip_scale//2:skip_scale//2+c])
        self.register_buffer('permutation',channel_permutation(2*c))

    def fuse(self,low_outview,skip_packed,width,height):
        c=self.channels
        if width<=0 or height<=0 or width%8 or height%8:
            raise ValueError('upsample requires positive eight-aligned destination features')
        if any(t.dtype!=torch.float16 or t.device!=self.project.device for t in (low_outview,skip_packed)):
            raise ValueError('actual decoded inputs and original parameters must share FP16/device')
        low = unpack_outview(low_outview,width//2,height//2,2*c)
        skip = unpack_image(skip_packed,width,height,c)
        projected = chunked_linear(low[...,self.permutation],self.project)
        expanded = projected.repeat_interleave(2,0).repeat_interleave(2,1)
        return expanded+(skip*self.skip_scale).half()

    def forward(self,low_outview,skip_packed,width,height,ox,oy,*,window_batch=12):
        if window_batch<=0:
            raise ValueError('positive window batch required')
        fused = self.fuse(low_outview,skip_packed,width,height)
        windows,mapping=gather_packed(pack_image(fused),width,height,self.channels,ox,oy)
        logical=torch.cat([self.block(batch) for batch in windows.split(window_batch)])
        return quantize_e4(scatter_packed(logical,mapping,width,height))
