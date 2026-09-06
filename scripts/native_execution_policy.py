"""Explicit arithmetic profiles; original model bytes remain immutable.

recovered_k32 is the existing diagnostic baseline. native_fp16 permits one
library GEMM per matrix instead of FP32 products with FP16 rounding every K32.
It changes accumulation/rounding, NOT architecture, layouts or learned weights.
Neither profile establishes complete-network, image-quality or realtime status.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import torch

PROFILES = ('recovered_k32', 'native_fp16')
_profile = ContextVar('dlssnr_native_arithmetic_profile', default='recovered_k32')


def current_profile():
    return _profile.get()


@contextmanager
def execution_policy(profile):
    if profile not in PROFILES:
        raise ValueError('unknown native arithmetic profile')
    token = _profile.set(profile)
    try:
        yield
    finally:
        _profile.reset(token)


def use_native_fp16():
    return current_profile() == 'native_fp16'


def native_gemm(x, weight, seed=None):
    """FP16 operands/library GEMM, then optional FP16 residual addition.

FP8 boundaries stay at callers. CPU tensors are supported only for reference
tests; the deployment caller must independently require ROCm GPU tensors.
No output-dependent host copies, tensor resizing, or fallback implementation.
"""
    if (x.dtype != torch.float16 or weight.dtype != torch.float16 or x.device != weight.device
            or x.ndim < 2 or weight.ndim < 2 or x.shape[-1] != weight.shape[-2]):
        raise ValueError('native GEMM requires compatible FP16 operands on one device')
    if seed is not None and (seed.dtype != torch.float16 or seed.device != x.device):
        raise ValueError('residual seed must share FP16 dtype and device')
    result = torch.matmul(x, weight)
    return result if seed is None else result + seed


def profile_description():
    return {'name': current_profile(), 'original_weights_modified': False,
            'accumulation': 'FP16_LIBRARY_GEMM_THEN_FP16_RESIDUAL' if use_native_fp16() else
            'FP32_K32_PRODUCTS_WITH_FP16_PARTIAL_ACCUMULATION',
            'fp8_boundaries_preserved': True,
            'native_fp16_scope': ['grouped_ffn', 'window_attention', 'split512', 'vit1024',
                                 'learned_boundary_projections'],
            'legacy_head_and_swin32_fast_path_complete': False,
            'precision_equivalent_to_nvidia': False, 'realtime_accepted': False}
