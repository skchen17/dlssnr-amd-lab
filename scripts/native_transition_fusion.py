"""Opt-in bounded native scale transitions; no CPU fallback or synchronization."""
import ctypes as ct
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch


_active=ContextVar('native_transition_fusion',default=None)


def active_transition_fusion():return _active.get()


class EncoderTransitionFusion:
    def __init__(self,dll,*,encoder=True,decoder=True):
        if type(encoder) is not bool or type(decoder) is not bool:raise ValueError('explicit transition module flags required')
        self.encoder_enabled,self.decoder_enabled=encoder,decoder
        self.library=ct.CDLL(str(Path(dll).resolve()))
        self.pool=self.library.native_transition_encoder_pool_permute_skip
        self.pool.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,
                            ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
        self.pool.restype=ct.c_int
        self.pack=self.library.native_transition_encoder_quantize_pack
        self.pack.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
        self.pack.restype=ct.c_int
        self.decoder_unpack=self.library.native_transition_decoder_unpack_permute
        self.decoder_unpack.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,
                                      ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
        self.decoder_unpack.restype=ct.c_int
        self.decoder_pack=self.library.native_transition_decoder_expand_skip_pack
        self.decoder_pack.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,
                                    ct.c_int,ct.c_int,ct.c_int,ct.c_void_p]
        self.decoder_pack.restype=ct.c_int
        self.pool_launches=0;self.pack_launches=0;self.projection_calls=0;self.debug_outviews=0
        self.decoder_unpack_launches=0;self.decoder_pack_launches=0;self.decoder_projection_calls=0

    @staticmethod
    def _require(tensor,device):
        if (not torch.version.hip or not tensor.is_cuda or tensor.device!=device
                or tensor.dtype!=torch.float16 or not tensor.is_contiguous() or tensor.requires_grad):
            raise ValueError('inference-only contiguous ROCm FP16 tensors required; no CPU fallback')

    def encoder(self,source,width,height,channels,permutation,weight,*,capture_layouts=False):
        from native_split_swin512 import chunked_linear
        from native_multiscale import pack_outview
        if width%8 or height%8 or channels%32 or source.numel()!=width*height*channels:
            raise ValueError('reviewed encoder transition geometry required')
        self._require(source,weight.device)
        if (permutation.device!=source.device or permutation.dtype!=torch.int64
                or not permutation.is_contiguous() or weight.dtype!=torch.float16):
            raise ValueError('resident permutation and projection must share the GPU')
        pooled=torch.empty((height//2,width//2,channels),device=source.device,dtype=torch.float16)
        skip=torch.empty_like(source)
        stream=torch.cuda.current_stream(source.device).cuda_stream
        code=self.pool(source.data_ptr(),permutation.data_ptr(),pooled.data_ptr(),skip.data_ptr(),
                       width,height,channels,stream)
        if code:raise RuntimeError(f'encoder pool/skip HIP launch error {code}; no retry')
        self.pool_launches+=1
        projected=chunked_linear(pooled,weight)
        self.projection_calls+=1
        resident=torch.empty(projected.numel(),device=source.device,dtype=torch.float16)
        code=self.pack(projected.data_ptr(),resident.data_ptr(),width//2,height//2,channels*2,stream)
        if code:raise RuntimeError(f'encoder quantize/pack HIP launch error {code}; no retry')
        self.pack_launches+=1
        result={'skip':skip,'resident':resident,'source_size':(width,height),'target_size':(width//2,height//2)}
        if capture_layouts:
            # Compatibility/debug only; hot path never requests this outview.
            from native_multiscale import unpack_image
            result['downsampled']=resident
            result['captured_outview']=pack_outview(unpack_image(resident,width//2,height//2,channels*2))
            self.debug_outviews+=1
        return result

    def decoder(self,low,skip,width,height,channels,permutation,weight,skip_scale):
        from native_split_swin512 import chunked_linear
        if width%8 or height%8 or channels%32:
            raise ValueError('reviewed decoder transition geometry required')
        source_channels=channels*2
        if low.numel()!=width//2*(height//2)*source_channels or skip.numel()!=width*height*channels:
            raise ValueError('decoder resident feature size mismatch')
        self._require(low,weight.device);self._require(skip,weight.device)
        if (permutation.device!=low.device or permutation.dtype!=torch.int64 or not permutation.is_contiguous()
                or skip_scale.device!=low.device or skip_scale.dtype!=torch.float16 or not skip_scale.is_contiguous()):
            raise ValueError('decoder parameters must be resident on the same GPU')
        logical=torch.empty((height//2,width//2,source_channels),device=low.device,dtype=torch.float16)
        stream=torch.cuda.current_stream(low.device).cuda_stream
        code=self.decoder_unpack(low.data_ptr(),permutation.data_ptr(),logical.data_ptr(),
                                 width//2,height//2,source_channels,stream)
        if code:raise RuntimeError(f'decoder unpack/permutation HIP launch error {code}; no retry')
        self.decoder_unpack_launches+=1
        projected=chunked_linear(logical,weight);self.decoder_projection_calls+=1
        resident=torch.empty(width*height*channels,device=low.device,dtype=torch.float16)
        code=self.decoder_pack(projected.data_ptr(),skip.data_ptr(),skip_scale.data_ptr(),resident.data_ptr(),
                               width,height,channels,stream)
        if code:raise RuntimeError(f'decoder expand/skip/pack HIP launch error {code}; no retry')
        self.decoder_pack_launches+=1
        return resident

    def counts(self):
        return {'pool_permute_skip':self.pool_launches,'projection_calls':self.projection_calls,
                'quantize_pack':self.pack_launches,'debug_outviews':self.debug_outviews,
                'decoder_unpack_permute':self.decoder_unpack_launches,
                'decoder_projection_calls':self.decoder_projection_calls,
                'decoder_expand_skip_pack':self.decoder_pack_launches}


@contextmanager
def transition_fusion(dll=None,*,encoder=False,decoder=False):
    if (encoder or decoder) and dll is None:raise ValueError('native transition requires an explicit reviewed DLL')
    value=EncoderTransitionFusion(dll,encoder=encoder,decoder=decoder) if encoder or decoder else None
    token=_active.set(value)
    try:yield value
    finally:_active.reset(token)
