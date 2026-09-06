"""Opt-in experimental integer E4M3 fusion; not enabled by any model profile.

Caller retains input/output until current stream completes. No CPU fallback,
implicit layout copy, synchronization, autograd or dispatch retries.
"""
import ctypes as ct
from pathlib import Path


class QuantizeE4:
    entrypoint='native_fusion_quantize_f16'

    def __init__(self, dll):
        self.library=ct.CDLL(str(Path(dll).resolve()))
        self.launch=getattr(self.library,self.entrypoint)
        self.launch.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64,ct.c_void_p]
        self.launch.restype=ct.c_int

    def __call__(self,x):
        import torch
        if not torch.version.hip or not x.is_cuda:
            raise ValueError('ROCm GPU tensor required; no CPU fallback')
        if x.dtype!=torch.float16 or not x.is_contiguous() or x.numel()==0:
            raise ValueError('nonempty contiguous FP16 tensor required')
        if x.requires_grad:
            raise ValueError('inference-only fusion; use original STE when training')
        with torch.cuda.device(x.device):
            y=torch.empty_like(x)
            code=self.launch(x.data_ptr(),y.data_ptr(),x.numel(),torch.cuda.current_stream(x.device).cuda_stream)
        if code: raise RuntimeError(f'HIP launch error {code}; no retry')
        return y


class CubicQuantizeE4(QuantizeE4):
    """Exact three FP16-round cubic_silu followed by E4 quantization, opt-in."""
    entrypoint='native_fusion_cubic_quantize_f16'
