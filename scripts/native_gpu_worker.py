"""Separate-process original-weight ROCm worker; IPC carries metadata, never pixels.

Native texture lease must come from the launching producer via inherited pipe.
This version has a bounded diagnostic color/size contract; no game acceptance.
"""
import argparse
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import sys
import time
from native_gpu_protocol import parse,descriptor,Commands,MAX_MESSAGE_BYTES
from native_model_package import sha
from validate_rocm_lifecycle import configure_fault_logging


def emit(value):print(json.dumps(value,allow_nan=False),flush=True)


def main(a):
    import torch
    from native_frame_runtime import ResidentNativeFrame,FrameSpec
    trace=(a.output/'worker_traceback.log').open('x');configure_fault_logging(trace);globals()['_trace']=trace
    torch.set_num_threads(2)
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
    torch.cuda.init()
    def read():
        line=sys.stdin.readline(MAX_MESSAGE_BYTES+1)
        if not line:raise RuntimeError('producer pipe closed; no retry')
        return parse(line)
    def wait(event=None):
        if event is None:event=torch.cuda.Event();event.record()
        deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:raise TimeoutError('worker GPU timeout; no cancellation/retry')
            time.sleep(.005)
        return event
    lib=ct.CDLL(str(a.dll.resolve()));lib.nr_bridge_error.restype=ct.c_char_p
    lib.nr_current_hip_luid.argtypes=[];lib.nr_current_hip_luid.restype=ct.c_uint64
    luid=lib.nr_current_hip_luid()
    if not luid:raise RuntimeError('worker HIP adapter LUID unavailable')
    emit({'op':'hello','pid':os.getpid(),'protocol':1,'adapter_luid':luid})
    message=read()
    if set(message)!={'op','descriptor'} or message['op']!='prepare':raise ValueError('initial descriptor required')
    fields=message['descriptor'];w,h,generation=descriptor(fields,a.producer,os.getpid(),max_width=a.max_width,max_height=a.max_height)
    if not 1<=a.max_padded_pixels<=8847360:raise ValueError('workspace budget outside explicit limit')
    runtime=ResidentNativeFrame('local_models/native_single_color_v1','AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3',max_padded_pixels=a.max_padded_pixels,optimization_profile=a.optimization_profile) if a.network else None
    if runtime:runtime.prepare(FrameSpec(w,h,generation));value=runtime.input
    else:value=torch.empty((h,w,4),dtype=torch.float16,device='cuda')
    wait()
    def check(v):
        if v:raise RuntimeError(lib.nr_bridge_error().decode())
    lib.nr_remote_open.argtypes=[ct.POINTER(ct.c_uint64),ct.c_uint32];lib.nr_remote_open.restype=ct.c_void_p
    for name in ('begin','finish'):
        f=getattr(lib,'nr_remote_'+name);f.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64,ct.c_uint64,ct.c_uint64,ct.c_void_p];f.restype=ct.c_int
    lib.nr_remote_consumed.argtypes=[ct.c_void_p,ct.c_uint64];lib.nr_remote_consumed.restype=ct.c_int
    lib.nr_remote_close.argtypes=[ct.c_void_p];lib.nr_remote_close.restype=ct.c_int
    remote=lib.nr_remote_open((ct.c_uint64*16)(*fields),a.producer)
    if not remote:check(-1)
    state=Commands(generation);frames=[];pending_output=None;copy_done=None;finite=None
    emit({'op':'ready','generation':generation,'pid':os.getpid(),'network':a.network,'optimization_profile':a.optimization_profile})
    while True:
        message=read()
        if message.get('op')=='close':state.close(message);break
        if message.get('op')=='consumed':
            state.consumed(message);check(lib.nr_remote_consumed(remote,state.sequence))
            if runtime and not runtime.release_output(copy_done):raise RuntimeError('output copy incomplete')
            pending_output=None;copy_done=None
            emit({'op':'released','generation':generation,'sequence':state.sequence});continue
        if len(frames)>=a.max_frames:raise RuntimeError('frame limit exceeded')
        state.frame(message);seq=state.sequence
        check(lib.nr_remote_begin(remote,value.data_ptr(),w*h*8,generation,seq,torch.cuda.current_stream().cuda_stream))
        wait()
        input_raw=value.cpu().numpy().tobytes() if a.diagnostic_readback else None
        input_sha=sha(input_raw) if input_raw is not None else None
        start=time.perf_counter()
        if runtime:
            wait(runtime.submit(value,frame_id=seq,generation=generation,frame_seed=message['seed']))
            pending_output=runtime.complete()
        else:pending_output=value.flip(1).contiguous();wait()
        if pending_output is None:raise RuntimeError('output completion missing')
        finite=torch.isfinite(pending_output).all();wait()
        if not bool(finite):raise RuntimeError('nonfinite network result; no output submission')
        elapsed=(time.perf_counter()-start)*1000
        check(lib.nr_remote_finish(remote,pending_output.data_ptr(),w*h*8,generation,seq,torch.cuda.current_stream().cuda_stream))
        copy_done=wait()
        output_raw=pending_output.cpu().numpy().tobytes() if a.diagnostic_readback else None
        if a.diagnostic_readback and seq<=2:
            for role,raw in (('input',input_raw),('output',output_raw)):
                with (a.output/f'audit_{role}{seq}.rgba16f').open('xb') as f:f.write(raw)
        result={'op':'done','sequence':seq,'generation':generation,'pid':os.getpid(),
                'input_sha256':input_sha,'output_sha256':sha(output_raw) if output_raw is not None else None,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'allocator_budget_bytes':runtime.schedule.allocator_budget_bytes if runtime else None,
                'frame_ms':elapsed,'finite':True,'metadata_only_ipc':True,'cpu_neural_fallback':False,'optimization_profile':a.optimization_profile}
        frames.append(result)
        with (a.output/'worker_frames.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        emit(result)
    if runtime:runtime.close()
    check(lib.nr_remote_close(remote));del runtime,value,pending_output,copy_done,finite
    gc.collect();torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
    result={'op':'closed','pid':os.getpid(),'frames':len(frames),'allocated_after_release_bytes':torch.cuda.memory_allocated(),
            'reserved_after_release_bytes':torch.cuda.memory_reserved(),'metadata_only_ipc':True,'game_runtime_ready':False}
    with (a.output/'worker_close.json').open('x') as f:json.dump(result,f,indent=2)
    emit(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--producer',type=int,required=True)
    p.add_argument('--dll',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-frames',type=int,choices=(1,12),default=1)
    p.add_argument('--network',action='store_true');p.add_argument('--diagnostic-readback',action='store_true')
    p.add_argument('--max-width',type=int,default=768);p.add_argument('--max-height',type=int,default=512)
    p.add_argument('--max-padded-pixels',type=int,default=1048576)
    p.add_argument('--optimization-profile',choices=('baseline12','native_opt2','native_opt3'),default='baseline12')
    a=p.parse_args();main(a)
