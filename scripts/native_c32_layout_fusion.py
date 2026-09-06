"""C32 whole-grid gather/scatter: one launch each with no tensor index map."""
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
    def __init__(self,dll):
        self.library=ct.CDLL(str(Path(dll).resolve()));self.launch=self.library.native_fusion_c32_layout
        self.launch.argtypes=[ct.c_void_p]*2+[ct.c_int]*5+[ct.c_void_p];self.launch.restype=ct.c_int
        self.gather_calls=0;self.scatter_calls=0
    def apply(self,x,m,scatter=False):
        import torch
        w,h,ox,oy=m.width,m.height,m.ox,m.oy
        if min(w,h)<=0 or w%4 or h%4 or ox not in (0,-4) or oy not in (0,-4):raise ValueError('C32 geometry invalid')
        if not torch.version.hip or not x.is_cuda or x.dtype!=torch.float16 or not x.is_contiguous() or x.requires_grad:
            raise ValueError('C32 fusion requires contiguous inference ROCm FP16')
        windows=((w-ox+7)//8)*((h-oy+7)//8)
        if x.numel()!=(windows*2048 if scatter else w*h*32):raise ValueError('C32 storage mismatch')
        with torch.cuda.device(x.device):
            out=torch.empty((w*h*32,) if scatter else (windows,2048),device=x.device,dtype=x.dtype)
            code=self.launch(x.data_ptr(),out.data_ptr(),w,h,ox,oy,int(scatter),torch.cuda.current_stream(x.device).cuda_stream)
        if code:raise RuntimeError(f'C32 layout launch failed {code}')
        if scatter:self.scatter_calls+=1
        else:self.gather_calls+=1
        return out

@contextmanager
def c32_layout_fusion(dll=None):
    op=None if dll is None else Layout(dll);token=_active.set(op)
    try:yield op
    finally:_active.reset(token)
