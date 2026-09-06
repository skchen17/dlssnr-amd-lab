"""Whole-feature adapters for original split512 encoder and decoder stages."""
import torch
from torch import nn
from native_split_swin512 import RecoveredSplitSwin512
from native_packed_swin import gather_packed, scatter_packed, logical_indices
from native_multiscale import pack_image, unpack_image, unpack_outview, pack_outview, average_pool2x2
from native_swin_torch import quantize_e4


def pool_to_bottleneck(logical):
    """Preserve FP16 pre-store values through pool; pad its feature grid to4."""
    if logical.ndim != 3 or logical.shape[-1] != 512:
        raise ValueError('original 512-channel logical image required')
    pooled = quantize_e4(average_pool2x2(logical))
    h,w,c = pooled.shape
    padded = torch.zeros(((h+3)//4*4, (w+3)//4*4, c), device=pooled.device,dtype=pooled.dtype)
    padded[:h,:w] = pooled
    return padded


class RecoveredSplitImage512(nn.Module):
    native_graph_complete = False

    def __init__(self, records, *, input_layout='packed', output_layout='packed'):
        super().__init__()
        if input_layout not in ('packed','outview') or output_layout not in ('packed','outview','pool'):
            raise ValueError('explicit recovered split512 boundary layout required')
        self.block=RecoveredSplitSwin512(**records)
        self.input_layout,self.output_layout=input_layout,output_layout
        a,r=logical_indices(512)
        self.register_buffer('a_index',a)
        self.register_buffer('residual_index',r)

    def forward(self,source,width,height,ox,oy,*,window_batch=12,source_layout=None,output_layout=None):
        if window_batch<=0:
            raise ValueError('positive window batch required')
        layout=self.input_layout if source_layout is None else source_layout
        target=self.output_layout if output_layout is None else output_layout
        if layout not in ('packed','outview'):
            raise ValueError('explicit packed/outview source layout required')
        if target not in ('packed','outview','pool'):
            raise ValueError('explicit packed/outview/pool target layout required')
        if layout=='outview':
            source=pack_image(unpack_outview(source,width,height,512))
        windows,mapping=gather_packed(source,width,height,512,ox,oy)
        values=torch.cat([self.block.unquantized(p[:,self.a_index],p[:,self.residual_index])
                          for p in windows.split(window_batch)])
        packed=scatter_packed(values,mapping,width,height)
        if target=='pool':
            logical=unpack_image(packed,width,height,512)
            return {'skip':quantize_e4(packed),'pooled':pool_to_bottleneck(logical)}
        if target=='outview':
            packed=pack_outview(unpack_image(packed,width,height,512))
        return quantize_e4(packed)
