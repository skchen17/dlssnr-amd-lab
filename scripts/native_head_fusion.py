"""Explicit head-input fusion; geometry is computed on GPU, no index arena."""
import ctypes as ct
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_active=ContextVar('head_input_fusion',default=None)
def active_head_fusion():return _active.get()

class HeadInput:
    def __init__(self,dll,*,gather=False,epilogue=False,qkv=False):
        self.library=ct.CDLL(str(Path(dll).resolve()))
        self.launch=self.library.native_fusion_head_input
        self.launch.argtypes=[ct.c_void_p]*5+[ct.c_int]*4+[ct.c_void_p]
        self.launch.restype=ct.c_int
        self.launches=0
        self.compose_launch=self.library.native_fusion_head_compose
        self.compose_launch.argtypes=[ct.c_void_p]*3+[ct.c_int]*3+[ct.c_void_p]
        self.compose_launch.restype=ct.c_int
        self.compose_launches=0
        self.gather_launch=self.library.native_fusion_head_gather_e4
        self.gather_launch.argtypes=[ct.c_void_p]*3+[ct.c_uint64]+[ct.c_int]*2+[ct.c_void_p]
        self.gather_launch.restype=ct.c_int
        self.gather_launches=0
        self.in_block=False
        self.enable_gather=gather
        self.enable_epilogue=epilogue;self.epilogue_active=False;self.add_launches=0
        self.enable_qkv=qkv;self.qkv_active=False;self.qkv_launches=0
        if qkv:
            self.qkv_launch=self.library.native_fusion_head_qkv
            self.qkv_launch.argtypes=[ct.c_void_p]*5+[ct.c_uint64,ct.c_void_p];self.qkv_launch.restype=ct.c_int
        if epilogue:
            self.add_launch=self.library.native_fusion_head_add
            self.add_launch.argtypes=[ct.c_void_p]*4+[ct.c_uint64,ct.c_int,ct.c_void_p]
            self.add_launch.restype=ct.c_int

    def qkv(self,projection,scale):
        import torch
        if projection.ndim!=3 or projection.shape[-1]!=96 or scale.numel()!=1:
            raise ValueError('head QKV shape mismatch')
        if not torch.version.hip or any(not x.is_cuda or x.device!=projection.device or x.dtype!=torch.float16 or not x.is_contiguous() or x.requires_grad for x in (projection,scale)):
            raise ValueError('head QKV requires contiguous inference HIP FP16 inputs')
        with torch.cuda.device(projection.device):
            outputs=[torch.empty((*projection.shape[:-1],32),device=projection.device,dtype=projection.dtype) for _ in range(3)]
            code=self.qkv_launch(projection.data_ptr(),scale.data_ptr(),*(x.data_ptr() for x in outputs),projection.numel()//96,
                                 torch.cuda.current_stream(projection.device).cuda_stream)
        if code:raise RuntimeError(f'head QKV launch failed {code}')
        self.qkv_launches+=1;return outputs

    def add(self,seed,product,scale=None):
        import torch
        if not self.enable_epilogue:raise ValueError('head epilogue is not enabled')
        if seed.shape!=product.shape or seed.dtype!=torch.float16 or product.dtype!=torch.float32 or seed.numel()==0:
            raise ValueError('head epilogue requires matching half seed and float product')
        tensors=(seed,product) if scale is None else (seed,product,scale)
        if not torch.version.hip or any(not x.is_cuda or x.device!=seed.device or not x.is_contiguous() or x.requires_grad for x in tensors):
            raise ValueError('head epilogue requires contiguous inference tensors on one ROCm device')
        if scale is not None and (scale.dtype!=torch.float16 or scale.shape!=(seed.shape[-1],)):
            raise ValueError('head scale shape/dtype mismatch')
        with torch.cuda.device(seed.device):
            out=torch.empty_like(seed)
            code=self.add_launch(seed.data_ptr(),product.data_ptr(),None if scale is None else scale.data_ptr(),out.data_ptr(),
                                 seed.numel(),seed.shape[-1],torch.cuda.current_stream(seed.device).cuda_stream)
        if code:raise RuntimeError(f'head epilogue launch failed {code}')
        self.add_launches+=1;return out

    @contextmanager
    def block_scope(self):
        previous=self.in_block;self.in_block=self.enable_gather
        previous_epilogue=self.epilogue_active;self.epilogue_active=self.enable_epilogue
        previous_qkv=self.qkv_active;self.qkv_active=self.enable_qkv
        try:yield
        finally:self.in_block=previous;self.epilogue_active=previous_epilogue;self.qkv_active=previous_qkv

    def gather_e4(self,x,index):
        import torch
        if not x.is_cuda or x.dtype!=torch.float16 or not x.is_contiguous() or x.requires_grad:
            raise ValueError('head gather expects contiguous ROCm FP16 inference data')
        if index.device!=x.device or index.dtype!=torch.int64 or not index.is_contiguous():
            raise ValueError('head gather requires same-device contiguous int64 model indices')
        # Only immutable model-authored a_index/permutation buffers reach this
        # method; arbitrary caller indices are not an accepted external interface.
        with torch.cuda.device(x.device):
            out=torch.empty((*x.shape[:-1],*index.shape),device=x.device,dtype=x.dtype)
            code=self.gather_launch(x.data_ptr(),index.data_ptr(),out.data_ptr(),out.numel(),x.shape[-1],index.numel(),
                                   torch.cuda.current_stream(x.device).cuda_stream)
        if code:raise RuntimeError(f'head gather launch failed {code}')
        self.gather_launches+=1
        return out

    def compose(self,residual,base,pw,ph):
        import torch
        h,w,c=base.shape
        if c!=4 or w>pw or h>ph or pw%8 or ph%8 or residual.shape!=(((pw+11)//8)*((ph+11)//8),64,4):
            raise ValueError('compose geometry mismatch')
        if not torch.version.hip or any(not x.is_cuda or x.device!=base.device or x.dtype!=torch.float16
                or not x.is_contiguous() or x.requires_grad for x in (residual,base)):
            raise ValueError('compose requires contiguous inference FP16 ROCm tensors')
        with torch.cuda.device(base.device):
            out=torch.empty_like(base)
            code=self.compose_launch(residual.data_ptr(),base.data_ptr(),out.data_ptr(),w,h,pw,torch.cuda.current_stream(base.device).cuda_stream)
        if code:raise RuntimeError(f'head compose launch failed {code}')
        self.compose_launches+=1
        return out

    def __call__(self,main,skip,ms,ss,start,count,width,height):
        import torch
        tensors=(main,skip,ms,ss)
        if not torch.version.hip or any(not x.is_cuda or x.device!=main.device or x.dtype!=torch.float16
                or not x.is_contiguous() or x.requires_grad for x in tensors):
            raise ValueError('head fusion needs contiguous inference FP16 tensors on one ROCm device')
        if main.numel()!=width*height*8 or skip.numel()!=width*height*32 or ms.numel()!=32 or ss.numel()!=32:
            raise ValueError('head fusion input size mismatch')
        if min(width,height,count)<=0 or width%8 or height%8 or start<0 or start+count>((width+11)//8)*((height+11)//8):
            raise ValueError('head fusion geometry/range invalid')
        with torch.cuda.device(main.device):
            out=torch.empty((count,2048),device=main.device,dtype=main.dtype)
            code=self.launch(*(x.data_ptr() for x in tensors),out.data_ptr(),width,height,start,count,
                             torch.cuda.current_stream(main.device).cuda_stream)
        if code:raise RuntimeError(f'head HIP launch failed {code}; no retry')
        self.launches+=1
        return out

@contextmanager
def head_input_fusion(dll=None,*,gather=False,epilogue=False,qkv=False):
    op=None if dll is None else HeadInput(dll,gather=gather,epilogue=epilogue,qkv=qkv)
    token=_active.set(op)
    try:yield op
    finally:_active.reset(token)
