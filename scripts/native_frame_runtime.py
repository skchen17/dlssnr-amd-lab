"""Resident ROCm whole-frame candidate. No image readback or capture-plan loading.

Submission failures poison the executor: host timeout is NOT GPU cancellation.
Caller retains imported buffers through completion and output consumption.
Only diagnostic single-color input is supported, not formal game/HDR acceptance.
"""
from dataclasses import dataclass
import threading
import torch
from native_whole_frame import SingleColorWholeFrame,feature_geometry
from native_model_package import load_package
from native_execution_policy import execution_policy
from native_compact_layout import compact_layout
from native_inference_schedule import select_schedule


@dataclass(frozen=True)
class FrameSpec:
    width:int
    height:int
    generation:int
    format:str='RGBA16F'
    color_contract:str='LEGACY_SDR_DIAGNOSTIC_ONLY'

    def validate(self,budget):
        if type(self.generation) is not int or self.generation<1 or self.format!='RGBA16F':
            raise ValueError('positive generation and RGBA16F required')
        if self.color_contract!='LEGACY_SDR_DIAGNOSTIC_ONLY':
            raise ValueError('formal SDR/HDR exposure contract not implemented')
        return feature_geometry(self.width,self.height,budget)


class FrameLifecycle:
    """CPU metadata only; never treats elapsed time as GPU completion."""
    def __init__(self):
        self.state='NEW'; self.generation=0; self.frame_id=-1

    def prepare(self,generation):
        if self.state not in ('NEW','READY') or generation<=self.generation:
            raise RuntimeError('pending/failed/closed executor or stale generation')
        self.generation=generation; self.state='READY'; self.frame_id=-1

    def validate_submit(self,generation,frame_id):
        if self.state!='READY' or generation!=self.generation or type(frame_id) is not int or frame_id<=self.frame_id:
            raise RuntimeError('busy/failed executor or stale frame/resource generation')

    def submit(self,generation,frame_id):
        self.validate_submit(generation,frame_id); self.frame_id=frame_id; self.state='PENDING'

    def complete(self):
        if self.state!='PENDING': raise RuntimeError('no submitted frame')
        self.state='OUTPUT_READY'

    def consume(self):
        if self.state!='OUTPUT_READY': raise RuntimeError('output completion/consumption not established')
        self.state='READY'

    def fail(self): self.state='FAILED_RETAIN_RESOURCES'

    def close(self):
        if self.state not in ('NEW','READY'): raise RuntimeError('cannot release pending/unconsumed/uncertain GPU resources')
        self.state='CLOSED'


class ResidentNativeFrame:
    game_runtime_ready=False
    def __init__(self,package,manifest_sha256,*,max_padded_pixels=1048576,optimization_profile='baseline12'):
        self.schedule=select_schedule(optimization_profile)
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('ROCm required; no CPU or NVIDIA fallback')
        if 'gfx1201' not in torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName:
            raise RuntimeError('unreviewed GPU architecture')
        # Process-wide allocator ceiling, intended for the isolated resident worker.
        # Do not raise a pre-existing stricter ceiling; HIP/D3D allocations are external.
        if self.schedule.allocator_budget_bytes is not None:
            total=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            cap=min(torch.cuda.get_per_process_memory_fraction(),self.schedule.allocator_budget_bytes/total)
            torch.cuda.set_per_process_memory_fraction(cap)
        self.owner=threading.get_ident(); self.stream=torch.cuda.current_stream()
        self.lifecycle=FrameLifecycle(); self.budget=max_padded_pixels
        records,origins,settings=load_package(package,manifest_sha256)
        self.model=SingleColorWholeFrame(records,origins,**settings,max_padded_pixels=max_padded_pixels).eval().cuda()
        for p in self.model.parameters(): p.requires_grad_(False)
        if any(t.device.type!='cuda' for t in [*self.model.parameters(),*self.model.buffers()]):
            raise RuntimeError('CPU neural tensor')
        self.spec=None; self.input=None; self.output=None; self.event=None

    def require_owner(self):
        if threading.get_ident()!=self.owner or torch.cuda.current_stream().cuda_stream!=self.stream.cuda_stream:
            raise RuntimeError('owner or stream changed')

    def prepare(self,spec):
        self.require_owner(); spec.validate(self.budget)
        # Fail before allocation if an old output is still in use.
        if self.lifecycle.state not in ('NEW','READY') or spec.generation<=self.lifecycle.generation:
            raise RuntimeError('unreleased output or stale generation')
        value=torch.empty((spec.height,spec.width,4),dtype=torch.float16,device=self.stream.device)
        self.lifecycle.prepare(spec.generation); self.spec=spec; self.input=value

    def submit(self,source,*,frame_id,generation,frame_seed):
        self.require_owner(); self.lifecycle.validate_submit(generation,frame_id)
        if (not isinstance(frame_seed,int) or not 0<=frame_seed<=0xffffffff or source.shape!=self.input.shape
                or source.dtype!=self.input.dtype or source.device!=self.input.device):
            raise ValueError('invalid full-frame GPU input/seed')
        self.lifecycle.submit(generation,frame_id)
        try:
            with torch.no_grad(),execution_policy('native_fp16'),compact_layout(self.schedule.compact_layout):
                if source.data_ptr()!=self.input.data_ptr(): self.input.copy_(source)
                self.output=self.model(self.input,frame_seed,window_batch=self.schedule.window_batch,query_chunk=self.schedule.query_chunk)
                self.event=torch.cuda.Event(); self.event.record(self.stream)
        except BaseException:
            self.lifecycle.fail(); raise
        return self.event

    def complete(self):
        self.require_owner()
        try:
            if self.lifecycle.state!='PENDING': raise RuntimeError('no pending frame')
            if not self.event.query(): return None
            self.lifecycle.complete(); return self.output
        except BaseException:
            self.lifecycle.fail(); raise

    def release_output(self,consumer_complete):
        self.require_owner()
        # Must be a real event/fence adapter supplied by GPU transport, not bool.
        try:
            if not consumer_complete.query(): return False
        except BaseException:
            self.lifecycle.fail(); raise
        self.lifecycle.consume(); self.output=None; self.event=None
        return True

    def reset(self,spec):
        # No temporal state in this branch; generation reset still invalidates old frames.
        self.prepare(spec)

    def close(self):
        self.require_owner(); self.lifecycle.close()
        self.model=None; self.input=None; self.output=None; self.event=None
