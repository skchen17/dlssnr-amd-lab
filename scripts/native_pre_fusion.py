"""Opt-in single-color feature construction; existing positional noise retained."""
import ctypes as ct
from pathlib import Path
from contextvars import ContextVar
from contextlib import contextmanager
_active=ContextVar('pre_features_fusion',default=None)
def active_pre_fusion():return _active.get()

class PreFeatures:
    def __init__(self,dll):
        self.library=ct.CDLL(str(Path(dll).resolve()));self.launch=self.library.native_fusion_pre_features
        self.launch.argtypes=[ct.c_void_p]*4+[ct.c_int]*4+[ct.c_float,ct.c_void_p];self.launch.restype=ct.c_int
        self.launches=0
    def __call__(self,color,noise,pw,ph,scale,conditioning):
        import torch
        h,w,c=color.shape
        if not torch.version.hip or c!=4 or pw<w or ph<h or noise.shape!=(ph,pw,3) or len(conditioning)!=5:
            raise ValueError('pre fusion geometry/conditioning invalid')
        if any(not x.is_cuda or x.device!=color.device or x.dtype!=torch.float16 or not x.is_contiguous() or x.requires_grad for x in (color,noise)):
            raise ValueError('pre fusion requires contiguous ROCm FP16 inference inputs')
        with torch.cuda.device(color.device):
            constants=torch.tensor(conditioning,device=color.device,dtype=color.dtype)
            out=torch.empty((ph,pw,16),device=color.device,dtype=color.dtype)
            code=self.launch(color.data_ptr(),noise.data_ptr(),constants.data_ptr(),out.data_ptr(),w,h,pw,ph,float(scale),torch.cuda.current_stream(color.device).cuda_stream)
        if code:raise RuntimeError(f'pre feature launch failed {code}')
        self.launches+=1;return out

@contextmanager
def pre_features_fusion(dll=None):
    op=None if dll is None else PreFeatures(dll);token=_active.set(op)
    try:yield op
    finally:_active.reset(token)
