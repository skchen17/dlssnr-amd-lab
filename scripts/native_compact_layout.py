"""Opt-in exact cell permutation; no image tiling or floating-point arithmetic."""
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from native_sequence_layout import sequence_indices

_enabled=ContextVar('native_compact_feature_layout',default=False)


def enabled():return _enabled.get()


@contextmanager
def compact_layout(value=True):
    if type(value) is not bool:raise ValueError('explicit layout boolean required')
    token=_enabled.set(value)
    try:yield
    finally:_enabled.reset(token)


def validate(width,height,channels):
    if min(width,height,channels)<=0 or width%4 or height%4 or channels%32:
        raise ValueError('positive four-aligned image and32-aligned channels required')


def pack(logical):
    if logical.ndim!=3:raise ValueError('HWC feature image required')
    h,w,c=logical.shape;validate(w,h,c)
    # Only16*C indices instead ofH*W*C int64 indices. The four-by-four cells
    # are layout units, not independent images or attention contexts.
    inverse=torch.argsort(sequence_indices(16,c,device=logical.device).flatten())
    cells=logical.reshape(h//4,4,w//4,4,c).permute(0,2,1,3,4).reshape(-1,16*c)
    return cells[:,inverse].flatten()


def unpack(packed,width,height,channels):
    validate(width,height,channels)
    if packed.ndim!=1 or packed.numel()!=width*height*channels:raise ValueError('incorrect packed storage')
    order=sequence_indices(16,channels,device=packed.device).flatten()
    cells=packed.reshape(height//4,width//4,16*channels)[:,:,order]
    return cells.reshape(height//4,width//4,4,4,channels).permute(0,2,1,3,4).reshape(height,width,channels)
