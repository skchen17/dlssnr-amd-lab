"""Explicit experimental inference-only fusion scope; baseline is unchanged by default."""
from contextlib import contextmanager
from contextvars import ContextVar

_active=ContextVar('native_experimental_fusions',default=None)


def active_fusions():return _active.get()


class FusedOps:
    def __init__(self,dll):
        from native_fusion_quantize import QuantizeE4,CubicQuantizeE4
        self.quantizer=QuantizeE4(dll);self.cubic_quantizer=CubicQuantizeE4(dll)
        self.quantize_calls=0;self.cubic_calls=0;self.explicit_layout_copies=0

    def prepare(self,x):
        import torch
        if not torch.version.hip or not x.is_cuda or x.dtype!=torch.float16 or x.requires_grad:
            raise ValueError('fusion requires inference FP16 ROCm tensor; no arithmetic fallback')
        if not x.is_contiguous():
            self.explicit_layout_copies+=1
            x=x.contiguous()  # explicit measured copy, not a hidden change to the op API
        return x

    def quantize(self,x):
        y=self.quantizer(self.prepare(x));self.quantize_calls+=1;return y

    def cubic(self,x):
        y=self.cubic_quantizer(self.prepare(x));self.cubic_calls+=1;return y

    def counts(self):
        return {'quantize_launches':self.quantize_calls,'cubic_quantize_launches':self.cubic_calls,
                'explicit_layout_copies':self.explicit_layout_copies,
                'scope':'authored HIP launches only; does not count library or layout-copy GPU kernels'}


@contextmanager
def fusion_policy(dll=None):
    ops=None if dll is None else FusedOps(dll)
    token=_active.set(ops)
    try:yield ops
    finally:_active.reset(token)
