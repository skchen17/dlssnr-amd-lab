"""Opt-in whole-grid packed-image/window layout adapters.

The historical context name is retained for compatibility.  Wider channel
families must be explicitly enabled; loading a new DLL never changes them by
accident.
"""
import ctypes as ct
from pathlib import Path
from dataclasses import dataclass
from contextvars import ContextVar
from contextlib import contextmanager
_active=ContextVar('c32_layout_fusion',default=None)
def active_c32_layout():return _active.get()

@dataclass(frozen=True)
class Mapping:
    width:int
    height:int
    ox:int
    oy:int

class Layout:
    def __init__(self,dll,channels=(32,)):
        self.library=ct.CDLL(str(Path(dll).resolve()));self.launch=self.library.native_fusion_c32_layout
        self.launch.argtypes=[ct.c_void_p]*2+[ct.c_int]*5+[ct.c_void_p];self.launch.restype=ct.c_int
        allowed=frozenset(int(c) for c in channels)
        if not allowed or allowed-({32,64,128,256,512}):raise ValueError('unsupported packed-layout channel family')
        self.channels=allowed;self.wide_launch=None
        if allowed-{32}:
            try:self.wide_launch=self.library.native_fusion_packed_layout
            except AttributeError as error:raise ValueError('DLL lacks wide packed-layout ABI') from error
            self.wide_launch.argtypes=[ct.c_void_p]*2+[ct.c_int]*6+[ct.c_void_p];self.wide_launch.restype=ct.c_int
        self.gather_calls=0;self.scatter_calls=0;self.calls_by_channel={c:{'gather':0,'scatter':0} for c in allowed}
    def supports(self,channels):return channels in self.channels
    def apply(self,x,m,scatter=False,channels=32):
        import torch
        w,h,ox,oy=m.width,m.height,m.ox,m.oy
        if channels not in self.channels:raise ValueError('packed-layout channel family was not enabled')
        if min(w,h)<=0 or w%4 or h%4 or ox not in (0,-4) or oy not in (0,-4):raise ValueError('packed-layout geometry invalid')
        if not torch.version.hip or not x.is_cuda or x.dtype!=torch.float16 or not x.is_contiguous() or x.requires_grad:
            raise ValueError('layout fusion requires contiguous inference ROCm FP16')
        windows=((w-ox+7)//8)*((h-oy+7)//8)
        if x.numel()!=(windows*64*channels if scatter else w*h*channels):raise ValueError('packed-layout storage mismatch')
        with torch.cuda.device(x.device):
            out=torch.empty((w*h*channels,) if scatter else (windows,64*channels),device=x.device,dtype=x.dtype)
            stream=torch.cuda.current_stream(x.device).cuda_stream
            code=(self.launch(x.data_ptr(),out.data_ptr(),w,h,ox,oy,int(scatter),stream) if channels==32 else
                  self.wide_launch(x.data_ptr(),out.data_ptr(),w,h,channels,ox,oy,int(scatter),stream))
        if code:raise RuntimeError(f'C{channels} layout launch failed {code}')
        key='scatter' if scatter else 'gather';self.calls_by_channel[channels][key]+=1
        if scatter:self.scatter_calls+=1
        else:self.gather_calls+=1
        return out

@contextmanager
def c32_layout_fusion(dll=None,channels=(32,)):
    op=None if dll is None else Layout(dll,channels);token=_active.set(op)
    try:yield op
    finally:_active.reset(token)
