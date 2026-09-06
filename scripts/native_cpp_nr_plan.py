"""C++-owned HIP Graph runner; Python participates only during initialization.

This bridge captures the currently selected, already-correct kernel topology.
It removes Python/PyTorch operation submission from the per-frame hot path but
does not claim that all captured ATen/library kernels have been reauthored.
"""
from __future__ import annotations

import ctypes as ct
import threading
from pathlib import Path


class Desc(ct.Structure):
    _fields_=(('workspace_bytes',ct.c_uint64),('weight_bytes',ct.c_uint64),
              ('max_width',ct.c_uint32),('max_height',ct.c_uint32))


class Bindings(ct.Structure):
    _fields_=(('input',ct.c_void_p),('output',ct.c_void_p),('width',ct.c_uint32),
              ('height',ct.c_uint32),('frame_id',ct.c_uint64),('resource_generation',ct.c_uint64))


class GraphStats(ct.Structure):
    _fields_=(('total_nodes',ct.c_uint64),('kernel_nodes',ct.c_uint64),('memcpy_nodes',ct.c_uint64))


class CapturedNRPlan:
    classification='CPP_OWNED_CAPTURED_KERNEL_GRAPH'
    full_math_reauthored=False
    python_in_hot_path=False

    def __init__(self,dll,forward,example,*,wait,max_width=3840,max_height=2160,lifetimes=()):
        import torch
        if not torch.version.hip or not example.is_cuda or example.dtype!=torch.float16 or example.ndim!=3:
            raise ValueError('contiguous ROCm FP16 HWC input required')
        if not example.is_contiguous():raise ValueError('contiguous input required')
        self.owner=threading.get_ident();self.closed=False;self.sequence=0
        self.library=ct.CDLL(str(Path(dll).resolve()));self.lifetimes=tuple(lifetimes)
        self._bind_abi()
        self.handle=ct.c_void_p();desc=Desc(1,1,max_width,max_height)
        self._call(self.library.nrPlanCreate(ct.byref(desc),ct.byref(self.handle)),'create')
        stream_ptr=ct.c_void_p();self._call(self.library.nrPlanGetStream(self.handle,ct.byref(stream_ptr)),'get stream')
        self.stream=torch.cuda.ExternalStream(stream_ptr.value)
        self.input=example.clone();self.output=None
        # Warm handles, workspaces and allocator state on the exact capture stream.
        with torch.no_grad(),torch.cuda.stream(self.stream):warm=forward(self.input)
        wait(self.stream);del warm
        # PyTorch owns the capture pool during initialization so every ATen
        # temporary has a stable address. NRPlan clones/instantiates the raw
        # graph and is the only per-frame submitter after this point.
        self.capture_graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.no_grad(),torch.cuda.stream(self.stream):
            self.capture_graph.capture_begin()
            try:self.output=forward(self.input)
            except BaseException:
                # End an invalidated capture before destroying the owning HIP
                # stream. The original exception is the useful diagnostic.
                try:self.capture_graph.capture_end()
                except BaseException:pass
                self.close_noexcept();raise
            else:self.capture_graph.capture_end()
        self._call(self.library.nrPlanAdoptGraph(self.handle,ct.c_void_p(self.capture_graph.raw_cuda_graph())),'adopt graph')
        if self.output is None or not self.output.is_cuda or not self.output.is_contiguous():
            self.close_noexcept();raise RuntimeError('capture did not produce contiguous GPU output')
        stats=GraphStats();self._call(self.library.nrPlanGetGraphStats(self.handle,ct.byref(stats)),'get graph stats')
        self.graph_stats={'total_nodes':stats.total_nodes,'kernel_nodes':stats.kernel_nodes,'memcpy_nodes':stats.memcpy_nodes}
        self.target=torch.empty_like(self.output)
        self._call(self.library.nrPlanSetStaticIO(self.handle,ct.c_void_p(self.input.data_ptr()),self.input.numel()*self.input.element_size(),
            ct.c_void_p(self.output.data_ptr()),self.output.numel()*self.output.element_size(),example.shape[1],example.shape[0]),'set static IO')

    def _bind_abi(self):
        d=self.library
        d.nrPlanCreate.argtypes=[ct.POINTER(Desc),ct.POINTER(ct.c_void_p)]
        d.nrPlanGetStream.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p)]
        d.nrPlanAdoptGraph.argtypes=[ct.c_void_p,ct.c_void_p]
        d.nrPlanGetGraphStats.argtypes=[ct.c_void_p,ct.POINTER(GraphStats)]
        d.nrPlanSetRecorder.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p]
        d.nrPlanFinalize.argtypes=[ct.c_void_p]
        d.nrPlanSetStaticIO.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64,ct.c_void_p,ct.c_uint64,ct.c_uint32,ct.c_uint32]
        d.nrPlanSubmit.argtypes=[ct.c_void_p,ct.POINTER(Bindings),ct.c_void_p,ct.c_void_p]
        d.nrPlanDestroy.argtypes=[ct.c_void_p]
        for name in ('nrPlanCreate','nrPlanGetStream','nrPlanAdoptGraph','nrPlanGetGraphStats','nrPlanSetRecorder','nrPlanFinalize','nrPlanSetStaticIO','nrPlanSubmit','nrPlanDestroy'):
            getattr(d,name).restype=ct.c_int

    @staticmethod
    def _call(code,operation):
        if code:raise RuntimeError(f'NRPlan {operation} failed: HIP error {code}')

    def submit(self,source,target=None,*,resource_generation=1):
        import torch
        if self.closed or threading.get_ident()!=self.owner:raise RuntimeError('closed plan or changed owner')
        if source.shape!=self.input.shape or source.dtype!=self.input.dtype or source.device!=self.input.device or not source.is_contiguous():
            raise ValueError('input geometry/dtype/device changed; build another plan')
        target=self.target if target is None else target
        if target.shape!=self.output.shape or target.dtype!=self.output.dtype or target.device!=self.output.device or not target.is_contiguous():
            raise ValueError('output target mismatch')
        h,w,_=source.shape;binding=Bindings(source.data_ptr(),target.data_ptr(),w,h,self.sequence,resource_generation)
        self._call(self.library.nrPlanSubmit(self.handle,ct.byref(binding),None,None),'submit')
        self.sequence+=1;return target

    def close(self,wait):
        if self.closed:return
        wait(self.stream);self._call(self.library.nrPlanDestroy(self.handle),'destroy');self.closed=True

    def close_noexcept(self):
        if not getattr(self,'closed',True) and getattr(self,'handle',None):
            try:self.library.nrPlanDestroy(self.handle)
            finally:self.closed=True

    def __del__(self):
        self.close_noexcept()
