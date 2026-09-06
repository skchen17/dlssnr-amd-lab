"""Opt-in inference-only WMMA candidates. Never promotes approximate math."""
import ctypes as ct
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import weakref

PROFILES=('reference','wmma_fp16','wmma_fp8')
_active=ContextVar('native_matrix_fusion',default=None)


def validate_config(profile,modules,waves):
    if profile not in PROFILES or waves not in (1,2,4):raise ValueError('unreviewed WMMA profile/wave count')
    if not modules or set(modules)-{'head_ffn','head_attention','head_attention_bounded','head_softmax','head_output','pre_input','c32_ffn','c32_attention','c32_attention_bounded','c32_attention_staged','c32_attention_core','c64_ffn','c64_ffn_grouped','c64_ffn_grouped_fp8w','c64_attention','c64_attention_bounded','c64_attention_query','c128_ffn','c128_ffn_grouped','c128_ffn_grouped_fp8w','c128_attention','c128_attention_bounded','c128_attention_query','c256_ffn','c256_ffn_grouped','c256_ffn_grouped_fp8w','c512_ffn'}:raise ValueError('module does not yet have an implemented WMMA path')
    if any(name.endswith('_fp8w') for name in modules) and profile!='wmma_fp8':raise ValueError('resident FP8 weights require wmma_fp8 profile')
    if len({'c32_attention_bounded','c32_attention_staged','c32_attention_core'}&set(modules))>1:raise ValueError('select exactly one C32 fused attention strategy')


class MatrixFusion:
    def __init__(self,dll,profile,modules,waves):
        validate_config(profile,modules,waves)
        self.profile,self.modules,self.waves=profile,frozenset(modules),waves
        self.library=ct.CDLL(str(Path(dll).resolve()))
        self.launch=self.library.nr_head_ffn_wmma
        self.launch.argtypes=[ct.c_void_p]*8+[ct.c_int]*3+[ct.c_void_p]
        self.launch.restype=ct.c_int
        self.project_launch=self.library.nr_head_project_wmma
        self.project_launch.argtypes=[ct.c_void_p]*6+[ct.c_int]*3+[ct.c_void_p]
        self.project_launch.restype=ct.c_int
        self.tail_launch=self.library.nr_head_tail_exact
        self.tail_launch.argtypes=[ct.c_void_p]*3+[ct.c_uint64,ct.c_void_p]
        self.tail_launch.restype=ct.c_int
        self.qk_launch=self.library.nr_head_qk_wmma
        self.qk_launch.argtypes=[ct.c_void_p]*4+[ct.c_int]*3+[ct.c_void_p]
        self.qk_launch.restype=ct.c_int
        self.pv_launch=self.library.nr_head_pv_wmma
        self.pv_launch.argtypes=[ct.c_void_p]*3+[ct.c_int]*3+[ct.c_void_p]
        self.pv_launch.restype=ct.c_int
        self.qkv_launch=self.library.nr_head_qkv_project_wmma
        self.qkv_launch.argtypes=[ct.c_void_p]*4+[ct.c_int]*3+[ct.c_void_p]
        self.qkv_launch.restype=ct.c_int
        self.softmax_launch=self.library.nr_head_softmax_e4
        self.softmax_launch.argtypes=[ct.c_void_p]*2+[ct.c_uint64,ct.c_void_p]
        self.softmax_launch.restype=ct.c_int
        self.bounded_attention_launch=None
        if 'head_attention_bounded' in self.modules:
            self.bounded_attention_launch=self.library.nr_head_attention_window_fused
            self.bounded_attention_launch.argtypes=[ct.c_void_p]*8+[ct.c_int]*2+[ct.c_void_p]
            self.bounded_attention_launch.restype=ct.c_int
        self.pre_launch=self.library.nr_pre_project_exact
        self.pre_launch.argtypes=[ct.c_void_p]*3+[ct.c_uint64,ct.c_void_p]
        self.pre_launch.restype=ct.c_int
        self.c32_launch=self.library.nr_c32_ffn_wmma
        self.c32_launch.argtypes=[ct.c_void_p]*8+[ct.c_int]*3+[ct.c_void_p]
        self.c32_launch.restype=ct.c_int
        self.c32_norm_launch=self.library.nr_c32_qkv_norm;self.c32_norm_launch.argtypes=[ct.c_void_p]*5+[ct.c_uint64,ct.c_void_p];self.c32_norm_launch.restype=ct.c_int
        self.c32_qk_launch=self.library.nr_c32_qk_wmma;self.c32_qk_launch.argtypes=[ct.c_void_p]*4+[ct.c_int]*3+[ct.c_void_p];self.c32_qk_launch.restype=ct.c_int
        self.c32_pv_launch=self.library.nr_c32_pv_wmma;self.c32_pv_launch.argtypes=[ct.c_void_p]*3+[ct.c_int]*3+[ct.c_void_p];self.c32_pv_launch.restype=ct.c_int
        self.c32_project_launch=self.library.nr_c32_project_wmma;self.c32_project_launch.argtypes=[ct.c_void_p]*6+[ct.c_int]*3+[ct.c_void_p];self.c32_project_launch.restype=ct.c_int
        self.c32_bounded_attention_launch=None
        if 'c32_attention_bounded' in self.modules:
            self.c32_bounded_attention_launch=self.library.nr_c32_attention_window_fused
            self.c32_bounded_attention_launch.argtypes=[ct.c_void_p]*8+[ct.c_int]*2+[ct.c_void_p]
            self.c32_bounded_attention_launch.restype=ct.c_int
        self.c32_staged_launch=None;self.c32_attention_core_launch=None
        if {'c32_attention_staged','c32_attention_core'}&self.modules:
            core=self.library.nr_c32_attention_core_staged
            core.argtypes=[ct.c_void_p]*5+[ct.c_int]*2+[ct.c_void_p];core.restype=ct.c_int
            self.c32_attention_core_launch=core
        if 'c32_attention_staged' in self.modules:
            qkv=self.library.nr_c32_qkv_norm_staged
            qkv.argtypes=[ct.c_void_p]*7+[ct.c_int]*2+[ct.c_void_p];qkv.restype=ct.c_int
            self.c32_staged_launch=(qkv,self.c32_attention_core_launch)
        self.wide_launch={}
        for channels in (64,128,256):
            fn=getattr(self.library,f'nr_c{channels}_ffn_wmma');fn.argtypes=[ct.c_void_p]*9+[ct.c_int]*3+[ct.c_void_p];fn.restype=ct.c_int;self.wide_launch[channels]=fn
        self.wide_group_launch={};self.wide_group_fp8w_launch={}
        for channels in (64,128,256):
            fn=getattr(self.library,f'nr_c{channels}_group_ffn_wmma');fn.argtypes=[ct.c_void_p]*10+[ct.c_int]*2+[ct.c_void_p];fn.restype=ct.c_int;self.wide_group_launch[channels]=fn
            if f'c{channels}_ffn_grouped_fp8w' in self.modules:
                fn=getattr(self.library,f'nr_c{channels}_group_ffn_fp8w');fn.argtypes=[ct.c_void_p]*10+[ct.c_int,ct.c_void_p];fn.restype=ct.c_int;self.wide_group_fp8w_launch[channels]=fn
        self.wide_attention_launch={}
        for channels in (64,128):
            functions=[]
            for suffix,pointers in (('qkv_wmma',4),('qkv_norm',5),('qk_wmma',4),('pv_wmma',3),('project_wmma',6)):
                fn=getattr(self.library,f'nr_c{channels}_{suffix}')
                fn.argtypes=[ct.c_void_p]*pointers+([ct.c_uint64,ct.c_void_p] if suffix=='qkv_norm' else [ct.c_int]*3+[ct.c_void_p])
                fn.restype=ct.c_int;functions.append(fn)
            self.wide_attention_launch[channels]=tuple(functions)
        self.wide_bounded_attention_launch={}
        for channels in (64,128):
            name=f'c{channels}_attention_bounded'
            if name in self.modules:
                fn=getattr(self.library,f'nr_c{channels}_attention_head_fused')
                fn.argtypes=[ct.c_void_p]*6+[ct.c_int]*2+[ct.c_void_p];fn.restype=ct.c_int
                self.wide_bounded_attention_launch[channels]=fn
        self.wide_query_attention_launch={}
        for channels in (64,128):
            name=f'c{channels}_attention_query'
            if name in self.modules:
                fn=getattr(self.library,f'nr_c{channels}_attention_query_fused')
                fn.argtypes=[ct.c_void_p]*5+[ct.c_int]*2+[ct.c_void_p];fn.restype=ct.c_int
                self.wide_query_attention_launch[channels]=fn
        self.c512_launch=self.library.nr_c512_group_ffn_wmma;self.c512_launch.argtypes=[ct.c_void_p]*6+[ct.c_int]*3+[ct.c_void_p];self.c512_launch.restype=ct.c_int
        self.cache=weakref.WeakKeyDictionary();self.c32_cache=weakref.WeakKeyDictionary();self.wide_group_cache=weakref.WeakKeyDictionary();self.c32_attention_cache=weakref.WeakKeyDictionary();self.c512_cache=weakref.WeakKeyDictionary();self.c32_workspace={};self.c32_staged_workspace={};self.c32_core_workspace={};self.wide_group_workspace={};self.wide_attention_workspace={};self.wide_bounded_workspace={};self.wide_query_workspace={};self.grouped_ffn_geometry={};self.grouped_ffn_logical_dispatches=0;self.launches=0

    def prepare(self,module):
        import torch
        if module.training or torch.is_grad_enabled() or any(p.requires_grad for p in module.parameters()):
            raise ValueError('WMMA is frozen inference only')
        params=(module.expand,module.contract,module.project,module.qkv)
        identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params)
        old=self.cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            packed=tuple(quantize_e4(p).contiguous() for p in params)
            self.cache[module]=(identity,packed)
        return self.cache[module][1]

    def head_ffn(self,module,x,out=None):
        import torch
        if x.ndim!=2 or x.shape[1]!=2048 or not 0<x.shape[0]<=262144:
            raise ValueError('packed Head [windows,2048] required')
        weights=self.prepare(module)[:2]
        tensors=(x,*weights,module.ffn_scale,module.a_index,module.residual_index,module.permutation)
        if not torch.version.hip or any(not t.is_cuda or t.device!=x.device or not t.is_contiguous() for t in tensors):
            raise ValueError('same-device contiguous HIP tensors required')
        if 'gfx1201' not in torch.cuda.get_device_properties(x.device).gcnArchName:raise ValueError('gfx1201 required')
        if any(t.dtype!=torch.float16 for t in tensors[:4]) or any(t.dtype!=torch.int64 for t in tensors[4:]):
            raise ValueError('FP16 data/weights and int64 indices required')
        if x.requires_grad:raise ValueError('WMMA is inference only')
        with torch.cuda.device(x.device):
            if out is None:out=torch.empty((len(x),64,32),dtype=x.dtype,device=x.device)
            if out.shape!=(len(x),64,32) or out.device!=x.device or out.dtype!=x.dtype or not out.is_contiguous() or out.requires_grad:
                raise ValueError('caller output mismatch')
            out_start=out.data_ptr();out_end=out_start+out.numel()*out.element_size()
            if any(out_start<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<out_end for t in tensors):
                raise ValueError('WMMA output must not alias an input or weight')
            code=self.launch(*(t.data_ptr() for t in tensors),out.data_ptr(),len(x),self.waves,int(self.profile=='wmma_fp8'),torch.cuda.current_stream(x.device).cuda_stream)
        if code:raise RuntimeError(f'WMMA launch failed {code}; no fallback/retry')
        self.launches+=1
        return out

    def head_project(self,module,first,attention,out=None):
        import torch
        if first.shape!=attention.shape or first.ndim!=3 or first.shape[1:]!=(64,32):
            raise ValueError('Head projection needs matching [windows,64,32] tensors')
        weight=self.prepare(module)[2]
        tensors=(first,attention,weight,module.attention_scale,module.permutation)
        if not torch.version.hip or any(not t.is_cuda or t.device!=first.device or not t.is_contiguous() for t in tensors):
            raise ValueError('same-device contiguous HIP tensors required')
        if any(t.dtype!=torch.float16 for t in tensors[:4]) or tensors[4].dtype!=torch.int64:
            raise ValueError('Head projection dtype mismatch')
        if first.requires_grad or attention.requires_grad:raise ValueError('WMMA is inference only')
        with torch.cuda.device(first.device):
            if out is None:out=torch.empty_like(first)
            if out.shape!=first.shape or out.dtype!=first.dtype or out.device!=first.device or not out.is_contiguous() or out.requires_grad:
                raise ValueError('caller output mismatch')
            out_start=out.data_ptr();out_end=out_start+out.numel()*out.element_size()
            if any(out_start<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<out_end for t in tensors):
                raise ValueError('WMMA output must not alias an input or weight')
            code=self.project_launch(*(t.data_ptr() for t in tensors),out.data_ptr(),len(first),self.waves,
                int(self.profile=='wmma_fp8'),torch.cuda.current_stream(first.device).cuda_stream)
        if code:raise RuntimeError(f'Head project launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def head_tail(self,module,x,out=None):
        import torch
        if x.ndim!=3 or x.shape[1:]!=(64,32) or x.dtype!=torch.float16 or x.requires_grad:
            raise ValueError('Head tail needs inference FP16 [windows,64,32]')
        weight=module.tail
        if not torch.version.hip or any(not t.is_cuda or t.device!=x.device or not t.is_contiguous() for t in (x,weight)):
            raise ValueError('Head tail needs contiguous same-device HIP tensors')
        with torch.cuda.device(x.device):
            if out is None:out=torch.empty((*x.shape[:-1],4),dtype=x.dtype,device=x.device)
            if out.shape!=(*x.shape[:-1],4) or out.dtype!=x.dtype or out.device!=x.device or not out.is_contiguous() or out.requires_grad:
                raise ValueError('caller output mismatch')
            out_start=out.data_ptr();out_end=out_start+out.numel()*out.element_size()
            if any(out_start<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<out_end for t in (x,weight)):
                raise ValueError('Head tail output must not alias its input or weight')
            code=self.tail_launch(x.data_ptr(),weight.data_ptr(),out.data_ptr(),x.numel()//32,
                torch.cuda.current_stream(x.device).cuda_stream)
        if code:raise RuntimeError(f'Head tail launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def head_scores(self,module,q,k,out=None):
        import torch
        if q.shape!=k.shape or q.ndim!=3 or q.shape[1:]!=(64,32):raise ValueError('Head QK shape mismatch')
        tensors=(q,k,module.position_bias)
        if not torch.version.hip or any(not t.is_cuda or t.device!=q.device or t.dtype!=torch.float16 or not t.is_contiguous() or t.requires_grad for t in tensors):
            raise ValueError('Head QK needs inference FP16 contiguous HIP tensors')
        with torch.cuda.device(q.device):
            if out is None:out=torch.empty((len(q),64,64),dtype=q.dtype,device=q.device)
            if out.shape!=(len(q),64,64) or out.dtype!=q.dtype or out.device!=q.device or not out.is_contiguous() or out.requires_grad:raise ValueError('caller output mismatch')
            out_start=out.data_ptr();out_end=out_start+out.numel()*out.element_size()
            if any(out_start<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<out_end for t in tensors):raise ValueError('Head QK output aliases input')
            code=self.qk_launch(*(t.data_ptr() for t in tensors),out.data_ptr(),len(q),self.waves,int(self.profile=='wmma_fp8'),torch.cuda.current_stream(q.device).cuda_stream)
        if code:raise RuntimeError(f'Head QK launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def head_qkv(self,module,x,out=None):
        import torch
        if x.ndim!=3 or x.shape[1:]!=(64,32) or x.dtype!=torch.float16 or x.requires_grad:raise ValueError('Head QKV input mismatch')
        weight=self.prepare(module)[3];tensors=(x,weight,module.permutation)
        if not torch.version.hip or any(not t.is_cuda or t.device!=x.device or not t.is_contiguous() for t in tensors):raise ValueError('Head QKV needs contiguous same-device HIP tensors')
        with torch.cuda.device(x.device):
            if out is None:out=torch.empty((len(x),64,96),dtype=x.dtype,device=x.device)
            if out.shape!=(len(x),64,96) or out.dtype!=x.dtype or out.device!=x.device or not out.is_contiguous() or out.requires_grad:raise ValueError('caller output mismatch')
            code=self.qkv_launch(*(t.data_ptr() for t in tensors),out.data_ptr(),len(x),self.waves,int(self.profile=='wmma_fp8'),torch.cuda.current_stream(x.device).cuda_stream)
        if code:raise RuntimeError(f'Head QKV projection launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def head_pv(self,probability,v,out=None):
        import torch
        if probability.ndim!=3 or probability.shape[1:]!=(64,64) or v.shape!=(len(probability),64,32):raise ValueError('Head PV shape mismatch')
        if not torch.version.hip or any(not t.is_cuda or t.device!=probability.device or t.dtype!=torch.float16 or not t.is_contiguous() or t.requires_grad for t in (probability,v)):
            raise ValueError('Head PV needs inference FP16 contiguous HIP tensors')
        with torch.cuda.device(probability.device):
            if out is None:out=torch.empty_like(v)
            if out.shape!=v.shape or out.dtype!=v.dtype or out.device!=v.device or not out.is_contiguous() or out.requires_grad:raise ValueError('caller output mismatch')
            out_start=out.data_ptr();out_end=out_start+out.numel()*out.element_size()
            if any(out_start<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<out_end for t in (probability,v)):raise ValueError('Head PV output aliases input')
            code=self.pv_launch(probability.data_ptr(),v.data_ptr(),out.data_ptr(),len(probability),self.waves,int(self.profile=='wmma_fp8'),torch.cuda.current_stream(probability.device).cuda_stream)
        if code:raise RuntimeError(f'Head PV launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def head_probability(self,scores,out=None):
        import torch
        if scores.ndim!=3 or scores.shape[1:]!=(64,64) or scores.dtype!=torch.float16 or scores.requires_grad:raise ValueError('Head softmax input mismatch')
        if not torch.version.hip or not scores.is_cuda or not scores.is_contiguous():raise ValueError('Head softmax needs contiguous HIP input')
        with torch.cuda.device(scores.device):
            if out is None:out=torch.empty_like(scores)
            if out.shape!=scores.shape or out.dtype!=scores.dtype or out.device!=scores.device or not out.is_contiguous() or out.requires_grad:raise ValueError('caller output mismatch')
            code=self.softmax_launch(scores.data_ptr(),out.data_ptr(),scores.numel()//64,torch.cuda.current_stream(scores.device).cuda_stream)
        if code:raise RuntimeError(f'Head softmax launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def head_attention_bounded(self,module,first,out=None):
        import torch
        if self.bounded_attention_launch is None:raise ValueError('bounded Head attention was not enabled')
        if first.ndim!=3 or first.shape[1:]!=(64,32) or first.dtype!=torch.float16 or first.requires_grad:
            raise ValueError('bounded Head attention needs inference FP16 [windows,64,32]')
        qkv_weight,project_weight=self.prepare(module)[3],self.prepare(module)[2]
        tensors=(first,qkv_weight,module.q_scale,module.position_bias,project_weight,
                 module.attention_scale,module.permutation)
        if not torch.version.hip or any(not t.is_cuda or t.device!=first.device or not t.is_contiguous() for t in tensors):
            raise ValueError('bounded Head attention needs contiguous same-device HIP tensors')
        with torch.cuda.device(first.device):
            if out is None:out=torch.empty_like(first)
            if out.shape!=first.shape or out.dtype!=first.dtype or out.device!=first.device or not out.is_contiguous() or out.requires_grad:
                raise ValueError('bounded Head attention caller output mismatch')
            code=self.bounded_attention_launch(*(t.data_ptr() for t in tensors),out.data_ptr(),len(first),
                int(self.profile=='wmma_fp8'),torch.cuda.current_stream(first.device).cuda_stream)
        if code:raise RuntimeError(f'bounded Head attention launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def pre_project(self,features,weight,out=None):
        import torch
        if self.profile!='wmma_fp16':raise ValueError('Pre input projection has no reviewed FP8 contract')
        if features.ndim!=3 or features.shape[-1]!=16 or features.dtype!=torch.float16 or weight.shape!=(16,32) or weight.dtype!=torch.float16:
            raise ValueError('Pre projection shape/dtype mismatch')
        if features.numel()//16%16:raise ValueError('Pre projection requires complete WMMA tiles after full-frame padding')
        if not torch.version.hip or any(not t.is_cuda or t.device!=features.device or not t.is_contiguous() or t.requires_grad for t in (features,weight)):
            raise ValueError('Pre projection needs frozen contiguous HIP tensors')
        with torch.cuda.device(features.device):
            if out is None:out=torch.empty((*features.shape[:2],32),dtype=features.dtype,device=features.device)
            if out.shape!=(*features.shape[:2],32) or out.dtype!=features.dtype or out.device!=features.device or not out.is_contiguous() or out.requires_grad:raise ValueError('caller output mismatch')
            code=self.pre_launch(features.data_ptr(),weight.data_ptr(),out.data_ptr(),features.numel()//16,torch.cuda.current_stream(features.device).cuda_stream)
        if code:raise RuntimeError(f'Pre projection launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def c32_ffn(self,module,raw,out=None):
        import torch
        if module.channels!=32 or raw.ndim!=2 or raw.shape[1]!=2048 or raw.dtype!=torch.float16 or raw.requires_grad:raise ValueError('C32 packed FFN input mismatch')
        ffn=module.block.ffn;params=(ffn.expand,ffn.contract)
        identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params)
        old=self.c32_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c32_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        expand,contract=self.c32_cache[module][1]
        tensors=(raw,expand,contract,ffn.residual_scale,module.a_index,module.residual_index,ffn.permutation)
        if not torch.version.hip or any(not t.is_cuda or t.device!=raw.device or not t.is_contiguous() for t in tensors):raise ValueError('C32 FFN needs contiguous same-device HIP tensors')
        with torch.cuda.device(raw.device):
            if out is None:out=torch.empty((len(raw),64,32),dtype=raw.dtype,device=raw.device)
            if out.shape!=(len(raw),64,32) or out.dtype!=raw.dtype or out.device!=raw.device or not out.is_contiguous() or out.requires_grad:raise ValueError('caller output mismatch')
            code=self.c32_launch(*(t.data_ptr() for t in tensors),out.data_ptr(),len(raw),self.waves,int(self.profile=='wmma_fp8'),torch.cuda.current_stream(raw.device).cuda_stream)
        if code:raise RuntimeError(f'C32 FFN launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def c32_attention(self,module,post):
        import torch
        if module.channels!=32 or post.ndim!=3 or post.shape[1:]!=(64,32) or post.dtype!=torch.float16 or post.requires_grad:raise ValueError('C32 attention input mismatch')
        params=(module.qkv,module.project);identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params);old=self.c32_attention_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c32_attention_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        qkv_weight,project_weight=self.c32_attention_cache[module][1];stream=torch.cuda.current_stream(post.device).cuda_stream;windows=len(post);rows=windows*64;fp8=int(self.profile=='wmma_fp8')
        with torch.cuda.device(post.device):
            key=(post.device,post.dtype,windows);workspace=self.c32_workspace.get(key)
            if workspace is None:
                workspace=(torch.empty((windows,64,96),dtype=post.dtype,device=post.device),
                    *(torch.empty_like(post) for _ in range(3)),
                    torch.empty((windows,64,64),dtype=post.dtype,device=post.device),
                    torch.empty((windows,64,64),dtype=post.dtype,device=post.device),torch.empty_like(post))
                self.c32_workspace[key]=workspace
            projection,q,k,v,scores,probability,value=workspace
            code=self.qkv_launch(post.data_ptr(),qkv_weight.data_ptr(),module.permutation.data_ptr(),projection.data_ptr(),windows,self.waves,fp8,stream)
            if code:raise RuntimeError(f'C32 QKV projection failed {code}')
            code=self.c32_norm_launch(projection.data_ptr(),module.q_scale.data_ptr(),q.data_ptr(),k.data_ptr(),v.data_ptr(),rows,stream)
            if code:raise RuntimeError(f'C32 QKV norm failed {code}')
            code=self.c32_qk_launch(q.data_ptr(),k.data_ptr(),module.position_bias.data_ptr(),scores.data_ptr(),windows,self.waves,fp8,stream)
            if code:raise RuntimeError(f'C32 QK failed {code}')
            code=self.softmax_launch(scores.data_ptr(),probability.data_ptr(),rows,stream)
            if code:raise RuntimeError(f'C32 softmax failed {code}')
            code=self.c32_pv_launch(probability.data_ptr(),v.data_ptr(),value.data_ptr(),windows,self.waves,fp8,stream)
            if code:raise RuntimeError(f'C32 PV failed {code}')
            output=torch.empty_like(post);code=self.c32_project_launch(post.data_ptr(),value.data_ptr(),project_weight.data_ptr(),module.attention_scale.data_ptr(),module.permutation.data_ptr(),output.data_ptr(),windows,self.waves,fp8,stream)
        if code:raise RuntimeError(f'C32 output projection failed {code}')
        self.launches+=6;return output

    def c32_attention_bounded(self,module,post,out=None):
        import torch
        if self.c32_bounded_attention_launch is None:raise ValueError('bounded C32 attention is not enabled')
        if module.channels!=32 or post.ndim!=3 or post.shape[1:]!=(64,32) or post.dtype!=torch.float16 or post.requires_grad:
            raise ValueError('bounded C32 attention input mismatch')
        params=(module.qkv,module.project);identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params)
        old=self.c32_attention_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c32_attention_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        qkv_weight,project_weight=self.c32_attention_cache[module][1]
        tensors=(post,qkv_weight,module.q_scale,module.position_bias,project_weight,module.attention_scale,module.permutation)
        if not torch.version.hip or any(not t.is_cuda or t.device!=post.device or not t.is_contiguous() for t in tensors):
            raise ValueError('bounded C32 attention needs contiguous same-device HIP tensors')
        with torch.cuda.device(post.device):
            if out is None:out=torch.empty_like(post)
            if out.shape!=post.shape or out.dtype!=post.dtype or out.device!=post.device or not out.is_contiguous() or out.requires_grad:
                raise ValueError('bounded C32 attention caller output mismatch')
            code=self.c32_bounded_attention_launch(*(t.data_ptr() for t in tensors),out.data_ptr(),len(post),
                int(self.profile=='wmma_fp8'),torch.cuda.current_stream(post.device).cuda_stream)
        if code:raise RuntimeError(f'bounded C32 attention launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def c32_attention_staged(self,module,post,out=None):
        import torch
        if self.c32_staged_launch is None:raise ValueError('staged C32 attention is not enabled')
        if module.channels!=32 or post.ndim!=3 or post.shape[1:]!=(64,32) or post.dtype!=torch.float16 or post.requires_grad:
            raise ValueError('staged C32 attention input mismatch')
        params=(module.qkv,module.project);identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params)
        old=self.c32_attention_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c32_attention_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        qkv_weight,project_weight=self.c32_attention_cache[module][1]
        tensors=(post,qkv_weight,project_weight,module.permutation,module.q_scale,module.position_bias,module.attention_scale)
        if not torch.version.hip or any(not t.is_cuda or t.device!=post.device or not t.is_contiguous() for t in tensors):
            raise ValueError('staged C32 attention needs contiguous same-device HIP tensors')
        windows=len(post);stream=torch.cuda.current_stream(post.device).cuda_stream;fp8=int(self.profile=='wmma_fp8')
        with torch.cuda.device(post.device):
            key=(post.device,post.dtype,windows);workspace=self.c32_staged_workspace.get(key)
            if workspace is None:
                workspace=tuple(torch.empty_like(post) for _ in range(4))
                self.c32_staged_workspace[key]=workspace
            q,k,v,value=workspace
            qkv_launch,core_launch=self.c32_staged_launch
            code=qkv_launch(post.data_ptr(),qkv_weight.data_ptr(),module.q_scale.data_ptr(),module.permutation.data_ptr(),
                q.data_ptr(),k.data_ptr(),v.data_ptr(),windows,fp8,stream)
            if code:raise RuntimeError(f'staged C32 QKV/norm launch failed {code}')
            code=core_launch(q.data_ptr(),k.data_ptr(),v.data_ptr(),module.position_bias.data_ptr(),value.data_ptr(),windows,fp8,stream)
            if code:raise RuntimeError(f'staged C32 attention core launch failed {code}')
            if out is None:out=torch.empty_like(post)
            if out.shape!=post.shape or out.dtype!=post.dtype or out.device!=post.device or not out.is_contiguous() or out.requires_grad:
                raise ValueError('staged C32 attention caller output mismatch')
            code=self.c32_project_launch(post.data_ptr(),value.data_ptr(),project_weight.data_ptr(),module.attention_scale.data_ptr(),
                module.permutation.data_ptr(),out.data_ptr(),windows,self.waves,fp8,stream)
        if code:raise RuntimeError(f'staged C32 projection launch failed {code}')
        self.launches+=3;return out

    def c32_attention_core(self,module,post,out=None):
        import torch
        if self.c32_attention_core_launch is None:raise ValueError('core-fused C32 attention is not enabled')
        if module.channels!=32 or post.ndim!=3 or post.shape[1:]!=(64,32) or post.dtype!=torch.float16 or post.requires_grad:
            raise ValueError('core-fused C32 attention input mismatch')
        params=(module.qkv,module.project);identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params)
        old=self.c32_attention_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c32_attention_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        qkv_weight,project_weight=self.c32_attention_cache[module][1]
        tensors=(post,qkv_weight,project_weight,module.permutation,module.q_scale,module.position_bias,module.attention_scale)
        if not torch.version.hip or any(not t.is_cuda or t.device!=post.device or not t.is_contiguous() for t in tensors):
            raise ValueError('core-fused C32 attention needs contiguous same-device HIP tensors')
        windows=len(post);rows=windows*64;stream=torch.cuda.current_stream(post.device).cuda_stream;fp8=int(self.profile=='wmma_fp8')
        with torch.cuda.device(post.device):
            key=(post.device,post.dtype,windows);workspace=self.c32_core_workspace.get(key)
            if workspace is None:
                workspace=(torch.empty((windows,64,96),dtype=post.dtype,device=post.device),
                    *(torch.empty_like(post) for _ in range(4)))
                self.c32_core_workspace[key]=workspace
            projection,q,k,v,value=workspace
            code=self.qkv_launch(post.data_ptr(),qkv_weight.data_ptr(),module.permutation.data_ptr(),projection.data_ptr(),windows,self.waves,fp8,stream)
            if code:raise RuntimeError(f'core-fused C32 QKV projection failed {code}')
            code=self.c32_norm_launch(projection.data_ptr(),module.q_scale.data_ptr(),q.data_ptr(),k.data_ptr(),v.data_ptr(),rows,stream)
            if code:raise RuntimeError(f'core-fused C32 QKV norm failed {code}')
            code=self.c32_attention_core_launch(q.data_ptr(),k.data_ptr(),v.data_ptr(),module.position_bias.data_ptr(),value.data_ptr(),windows,fp8,stream)
            if code:raise RuntimeError(f'core-fused C32 attention core failed {code}')
            if out is None:out=torch.empty_like(post)
            if out.shape!=post.shape or out.dtype!=post.dtype or out.device!=post.device or not out.is_contiguous() or out.requires_grad:
                raise ValueError('core-fused C32 attention caller output mismatch')
            code=self.c32_project_launch(post.data_ptr(),value.data_ptr(),project_weight.data_ptr(),module.attention_scale.data_ptr(),
                module.permutation.data_ptr(),out.data_ptr(),windows,self.waves,fp8,stream)
        if code:raise RuntimeError(f'core-fused C32 projection failed {code}')
        self.launches+=4;return out

    def wide_ffn(self,module,raw,out=None):
        import torch
        c=module.channels
        if c not in (64,128,256) or raw.ndim!=2 or raw.shape[1]!=64*c or raw.dtype!=torch.float16 or raw.requires_grad:raise ValueError('wide packed FFN input mismatch')
        if c==256 and self.waves==4:raise ValueError('C256 wave4 requires 128 KiB LDS; candidate rejected before launch')
        ffn=module.block.ffn;params=(ffn.expand,ffn.contract,ffn.mix);identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params);old=self.c32_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c32_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        expand,contract,mix=self.c32_cache[module][1];tensors=(raw,expand,contract,mix,ffn.residual_scale,module.a_index,module.residual_index,ffn.permutation)
        if not torch.version.hip or any(not t.is_cuda or t.device!=raw.device or not t.is_contiguous() for t in tensors):raise ValueError('wide FFN needs contiguous same-device HIP tensors')
        if 'gfx1201' not in torch.cuda.get_device_properties(raw.device).gcnArchName:raise ValueError('gfx1201 required')
        if any(t.dtype!=torch.float16 for t in tensors[:5]) or any(t.dtype!=torch.int64 for t in tensors[5:]):raise ValueError('wide FFN dtype mismatch')
        with torch.cuda.device(raw.device):
            if out is None:out=torch.empty((len(raw),64,c),dtype=raw.dtype,device=raw.device)
            if out.shape!=(len(raw),64,c) or out.dtype!=raw.dtype or out.device!=raw.device or not out.is_contiguous() or out.requires_grad:raise ValueError('caller output mismatch')
            out_start=out.data_ptr();out_end=out_start+out.numel()*out.element_size()
            if any(out_start<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<out_end for t in tensors):raise ValueError('wide FFN output aliases input or model data')
            code=self.wide_launch[c](*(t.data_ptr() for t in tensors),out.data_ptr(),len(raw),self.waves,int(self.profile=='wmma_fp8'),torch.cuda.current_stream(raw.device).cuda_stream)
        if code:raise RuntimeError(f'C{c} FFN launch failed {code}; no fallback/retry')
        self.launches+=1;return out

    def wide_group_ffn(self,module,raw,out=None):
        import torch
        c=module.channels
        if c not in (64,128,256) or raw.ndim!=2 or raw.shape[1]!=64*c or raw.dtype!=torch.float16 or raw.requires_grad:
            raise ValueError('grouped wide FFN input mismatch')
        ffn=module.block.ffn;params=(ffn.expand,ffn.contract,ffn.mix)
        resident=f'c{c}_ffn_grouped_fp8w' in self.modules
        identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in (*params,ffn.permutation))
        old=self.wide_group_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            packed=tuple(quantize_e4(p).contiguous() for p in params)
            fp8_weights=(tuple(p.clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                               for p in params[:2]) if resident else None)
            inverse=torch.argsort(ffn.permutation).contiguous()
            self.wide_group_cache[module]=(identity,packed,inverse,fp8_weights)
        (expand,contract,mix),inverse_perm,fp8_weights=self.wide_group_cache[module][1:]
        if resident:expand,contract=fp8_weights
        tensors=(raw,expand,contract,ffn.residual_scale,module.a_index,module.residual_index,ffn.permutation,inverse_perm,mix)
        if not torch.version.hip or any(not t.is_cuda or t.device!=raw.device or not t.is_contiguous() for t in tensors):
            raise ValueError('grouped wide FFN needs contiguous same-device HIP tensors')
        if 'gfx1201' not in torch.cuda.get_device_properties(raw.device).gcnArchName:raise ValueError('gfx1201 required')
        expected_weight_dtype=torch.uint8 if resident else torch.float16
        if raw.dtype!=torch.float16 or expand.dtype!=expected_weight_dtype or contract.dtype!=expected_weight_dtype or ffn.residual_scale.dtype!=torch.float16 or any(t.dtype!=torch.int64 for t in tensors[4:8]) or mix.dtype!=torch.float16:
            raise ValueError('grouped wide FFN dtype mismatch')
        with torch.cuda.device(raw.device):
            key=(raw.device,raw.dtype,c,len(raw));workspace=self.wide_group_workspace.get(key)
            if workspace is None:
                workspace=(torch.empty((len(raw),64,c),dtype=raw.dtype,device=raw.device),
                           torch.empty((len(raw),64,c),dtype=raw.dtype,device=raw.device))
                self.wide_group_workspace[key]=workspace
            grouped,seed=workspace
            if out is None:out=torch.empty_like(grouped)
            if out.shape!=grouped.shape or out.dtype!=raw.dtype or out.device!=raw.device or not out.is_contiguous() or out.requires_grad:
                raise ValueError('grouped wide FFN caller output mismatch')
            arguments=(raw.data_ptr(),expand.data_ptr(),contract.data_ptr(),ffn.residual_scale.data_ptr(),
                module.a_index.data_ptr(),module.residual_index.data_ptr(),ffn.permutation.data_ptr(),inverse_perm.data_ptr(),
                grouped.data_ptr(),seed.data_ptr(),len(raw))
            if resident:code=self.wide_group_fp8w_launch[c](*arguments,torch.cuda.current_stream(raw.device).cuda_stream)
            else:code=self.wide_group_launch[c](*arguments,int(self.profile=='wmma_fp8'),torch.cuda.current_stream(raw.device).cuda_stream)
            if code:raise RuntimeError(f'C{c} grouped FFN launch failed {code}; no fallback/retry')
            torch.matmul(grouped,mix,out=out)
            torch.add(out,seed,out=out)
        geometry=f'c{c}_w{len(raw)}';self.grouped_ffn_geometry[geometry]=self.grouped_ffn_geometry.get(geometry,0)+1
        self.grouped_ffn_logical_dispatches+=3;self.launches+=1;return out

    def wide_attention(self,module,post):
        import torch
        c=module.channels
        if c not in (64,128) or post.ndim!=3 or post.shape[1:]!=(64,c) or post.dtype!=torch.float16 or post.requires_grad:raise ValueError('wide attention input mismatch')
        params=(module.qkv,module.project);identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params);old=self.c32_attention_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c32_attention_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        qkv_weight,project_weight=self.c32_attention_cache[module][1];tensors=(post,qkv_weight,project_weight,module.permutation,module.q_scale,module.position_bias,module.attention_scale)
        if not torch.version.hip or any(not t.is_cuda or t.device!=post.device or not t.is_contiguous() for t in tensors):raise ValueError('wide attention needs contiguous same-device HIP tensors')
        if 'gfx1201' not in torch.cuda.get_device_properties(post.device).gcnArchName:raise ValueError('gfx1201 required')
        windows=len(post);heads=c//32;vectors=windows*heads*64;stream=torch.cuda.current_stream(post.device).cuda_stream;fp8=int(self.profile=='wmma_fp8');qkv,norm,qk,pv,project=self.wide_attention_launch[c]
        with torch.cuda.device(post.device):
            key=(post.device,post.dtype,c,windows);workspace=self.wide_attention_workspace.get(key)
            if workspace is None:
                workspace=(torch.empty((windows,64,3*c),dtype=post.dtype,device=post.device),
                    *(torch.empty((windows,heads,64,32),dtype=post.dtype,device=post.device) for _ in range(3)),
                    torch.empty((windows,heads,64,64),dtype=post.dtype,device=post.device),
                    torch.empty((windows,heads,64,64),dtype=post.dtype,device=post.device),
                    torch.empty((windows,heads,64,32),dtype=post.dtype,device=post.device))
                self.wide_attention_workspace[key]=workspace
            projection,q,k,v,scores,probability,value=workspace
            code=qkv(post.data_ptr(),qkv_weight.data_ptr(),module.permutation.data_ptr(),projection.data_ptr(),windows,self.waves,fp8,stream)
            if code:raise RuntimeError(f'C{c} QKV projection failed {code}')
            code=norm(projection.data_ptr(),module.q_scale.data_ptr(),q.data_ptr(),k.data_ptr(),v.data_ptr(),vectors,stream)
            if code:raise RuntimeError(f'C{c} QKV norm failed {code}')
            code=qk(q.data_ptr(),k.data_ptr(),module.position_bias.data_ptr(),scores.data_ptr(),windows,self.waves,fp8,stream)
            if code:raise RuntimeError(f'C{c} QK failed {code}')
            code=self.softmax_launch(scores.data_ptr(),probability.data_ptr(),vectors,stream)
            if code:raise RuntimeError(f'C{c} softmax failed {code}')
            code=pv(probability.data_ptr(),v.data_ptr(),value.data_ptr(),windows,self.waves,fp8,stream)
            if code:raise RuntimeError(f'C{c} PV failed {code}')
            output=torch.empty_like(post);code=project(post.data_ptr(),value.data_ptr(),project_weight.data_ptr(),module.attention_scale.data_ptr(),module.permutation.data_ptr(),output.data_ptr(),windows,self.waves,fp8,stream)
        if code:raise RuntimeError(f'C{c} output projection failed {code}')
        self.launches+=6;return output

    def wide_attention_bounded(self,module,post,out=None):
        import torch
        c=module.channels
        if c not in self.wide_bounded_attention_launch or post.ndim!=3 or post.shape[1:]!=(64,c) or post.dtype!=torch.float16 or post.requires_grad:
            raise ValueError('bounded wide attention input/module mismatch')
        params=(module.qkv,module.project);identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params)
        old=self.c32_attention_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c32_attention_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        qkv_weight,project_weight=self.c32_attention_cache[module][1]
        tensors=(post,qkv_weight,project_weight,module.permutation,module.q_scale,module.position_bias,module.attention_scale)
        if not torch.version.hip or any(not t.is_cuda or t.device!=post.device or not t.is_contiguous() for t in tensors):
            raise ValueError('bounded wide attention needs contiguous same-device HIP tensors')
        windows=len(post);heads=c//32;stream=torch.cuda.current_stream(post.device).cuda_stream;fp8=int(self.profile=='wmma_fp8')
        with torch.cuda.device(post.device):
            key=(post.device,post.dtype,c,windows);value=self.wide_bounded_workspace.get(key)
            if value is None:
                value=torch.empty((windows,heads,64,32),dtype=post.dtype,device=post.device)
                self.wide_bounded_workspace[key]=value
            code=self.wide_bounded_attention_launch[c](post.data_ptr(),qkv_weight.data_ptr(),module.q_scale.data_ptr(),
                module.position_bias.data_ptr(),module.permutation.data_ptr(),value.data_ptr(),windows,fp8,stream)
            if code:raise RuntimeError(f'bounded C{c} attention-head launch failed {code}')
            if out is None:out=torch.empty_like(post)
            if out.shape!=post.shape or out.dtype!=post.dtype or out.device!=post.device or not out.is_contiguous() or out.requires_grad:
                raise ValueError('bounded wide attention output mismatch')
            project=self.wide_attention_launch[c][-1]
            code=project(post.data_ptr(),value.data_ptr(),project_weight.data_ptr(),module.attention_scale.data_ptr(),
                module.permutation.data_ptr(),out.data_ptr(),windows,self.waves,fp8,stream)
        if code:raise RuntimeError(f'bounded C{c} output projection failed {code}')
        self.launches+=2;return out

    def wide_attention_query(self,module,post,out=None):
        import torch
        c=module.channels
        if c not in self.wide_query_attention_launch or post.ndim!=3 or post.shape[1:]!=(64,c) or post.dtype!=torch.float16 or post.requires_grad:
            raise ValueError('query-tiled wide attention input/module mismatch')
        params=(module.qkv,module.project);identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params)
        old=self.c32_attention_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c32_attention_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        qkv_weight,project_weight=self.c32_attention_cache[module][1]
        tensors=(post,qkv_weight,project_weight,module.permutation,module.q_scale,module.position_bias,module.attention_scale)
        if not torch.version.hip or any(not t.is_cuda or t.device!=post.device or not t.is_contiguous() for t in tensors):
            raise ValueError('query-tiled wide attention needs contiguous same-device HIP tensors')
        windows=len(post);heads=c//32;vectors=windows*heads*64
        stream=torch.cuda.current_stream(post.device).cuda_stream;fp8=int(self.profile=='wmma_fp8')
        qkv,norm,_,_,project=self.wide_attention_launch[c]
        with torch.cuda.device(post.device):
            key=(post.device,post.dtype,c,windows);workspace=self.wide_query_workspace.get(key)
            if workspace is None:
                workspace=(torch.empty((windows,64,3*c),dtype=post.dtype,device=post.device),
                    *(torch.empty((windows,heads,64,32),dtype=post.dtype,device=post.device) for _ in range(4)))
                self.wide_query_workspace[key]=workspace
            projection,q,k,v,value=workspace
            code=qkv(post.data_ptr(),qkv_weight.data_ptr(),module.permutation.data_ptr(),projection.data_ptr(),windows,self.waves,fp8,stream)
            if code:raise RuntimeError(f'C{c} query-tiled QKV projection failed {code}')
            code=norm(projection.data_ptr(),module.q_scale.data_ptr(),q.data_ptr(),k.data_ptr(),v.data_ptr(),vectors,stream)
            if code:raise RuntimeError(f'C{c} query-tiled QKV norm failed {code}')
            code=self.wide_query_attention_launch[c](q.data_ptr(),k.data_ptr(),v.data_ptr(),module.position_bias.data_ptr(),
                value.data_ptr(),windows,fp8,stream)
            if code:raise RuntimeError(f'C{c} query-tiled attention core failed {code}')
            if out is None:out=torch.empty_like(post)
            if out.shape!=post.shape or out.dtype!=post.dtype or out.device!=post.device or not out.is_contiguous() or out.requires_grad:
                raise ValueError('query-tiled wide attention output mismatch')
            code=project(post.data_ptr(),value.data_ptr(),project_weight.data_ptr(),module.attention_scale.data_ptr(),
                module.permutation.data_ptr(),out.data_ptr(),windows,self.waves,fp8,stream)
        if code:raise RuntimeError(f'C{c} query-tiled projection failed {code}')
        self.launches+=4;return out

    def c512_group_ffn(self,module,projected,out=None):
        import torch
        if projected.ndim!=3 or projected.shape[1:]!=(64,512) or projected.dtype!=torch.float16 or projected.requires_grad:raise ValueError('C512 grouped FFN input mismatch')
        params=(module.expand,module.contract);identity=tuple((id(p),p.data_ptr(),p._version,p.device,p.dtype) for p in params);old=self.c512_cache.get(module)
        if old is None or old[0]!=identity:
            from native_swin_torch import quantize_e4
            self.c512_cache[module]=(identity,tuple(quantize_e4(p).contiguous() for p in params))
        expand,contract=self.c512_cache[module][1];tensors=(projected,expand,contract,module.perm64,module.perm256)
        if not torch.version.hip or any(not t.is_cuda or t.device!=projected.device or not t.is_contiguous() for t in tensors):raise ValueError('C512 grouped FFN needs contiguous same-device HIP tensors')
        if 'gfx1201' not in torch.cuda.get_device_properties(projected.device).gcnArchName:raise ValueError('gfx1201 required')
        if any(t.dtype!=torch.float16 for t in tensors[:3]) or any(t.dtype!=torch.int64 for t in tensors[3:]):raise ValueError('C512 grouped FFN dtype mismatch')
        with torch.cuda.device(projected.device):
            if out is None:out=torch.empty_like(projected)
            if out.shape!=projected.shape or out.dtype!=projected.dtype or out.device!=projected.device or not out.is_contiguous() or out.requires_grad:raise ValueError('C512 grouped FFN output mismatch')
            out_start=out.data_ptr();out_end=out_start+out.numel()*out.element_size()
            if any(out_start<t.data_ptr()+t.numel()*t.element_size() and t.data_ptr()<out_end for t in tensors):raise ValueError('C512 grouped FFN output aliases input or model data')
            code=self.c512_launch(*(t.data_ptr() for t in tensors),out.data_ptr(),len(projected),self.waves,int(self.profile=='wmma_fp8'),torch.cuda.current_stream(projected.device).cuda_stream)
        if code:raise RuntimeError(f'C512 grouped FFN launch failed {code}; no fallback/retry')
        self.launches+=1;return out


def active_matrix_fusion():return _active.get()


@contextmanager
def matrix_fusion(dll=None,profile='reference',modules=('head_ffn',),waves=1):
    validate_config(profile,modules,waves)
    if profile!='reference' and dll is None:raise ValueError('explicit WMMA DLL required')
    op=None if profile=='reference' else MatrixFusion(dll,profile,modules,waves)
    token=_active.set(op)
    try:yield op
    finally:_active.reset(token)
