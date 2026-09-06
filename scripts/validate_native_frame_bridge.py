"""Supervised single-frame standalone bridge; no game injection or automatic retry."""
import argparse
import atexit
import gc
import json
import os
from pathlib import Path
import sys
import time
from native_model_package import sha,ORIGINAL_SHA
from validate_rocm_lifecycle import supervise,save,configure_fault_logging


def exercise(args,phase):
    import torch
    from native_bridge_harness import SharedTextureHarness
    from native_frame_runtime import ResidentNativeFrame,FrameSpec
    torch.set_num_threads(2)
    if not torch.version.hip or not torch.cuda.is_available(): raise RuntimeError('ROCm only')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName: raise RuntimeError('unreviewed architecture')
    phase('gpu_initialized')
    def wait(event=None):
        if event is None: event=torch.cuda.Event();event.record()
        deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline: raise TimeoutError('GPU event timeout; retain resources, no retry/cancel')
            time.sleep(.005)
        return event
    runtime=None
    if args.mode=='network':
        root=Path('results/20260906_single_color_chain128_v1' if args.width==128 else 'results/20260906_single_color_gamecrop640_v1')
        spec=json.loads((root/'manifest.json').read_bytes())
        w,h=spec['input_geometry'];raw=(root/'input.rgba16f').read_bytes()
        if sha(raw)!=spec['input_sha256']: raise ValueError('input baseline hash')
        package=Path('local_models/native_single_color_v1')
        # Explicit pinned identity, not trust-on-first-use of package hashes.json.
        runtime=ResidentNativeFrame(package,'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
        runtime.prepare(FrameSpec(w,h,1));value=runtime.input
    else:
        w,h=args.width,129
        host=(torch.arange(h*w*4,dtype=torch.int32).reshape(h,w,4)%2048).half()/2048
        raw=host.numpy().tobytes();value=torch.empty((h,w,4),dtype=torch.float16,device='cuda')
    wait();phase('model_or_transport_prepared')
    bridge=SharedTextureHarness(args.dll,w,h);description=bridge.description()
    frames=[];input_exact=True;baseline_exact=None
    flipped=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(h,w,4).flip(1).numpy().tobytes()
    for iteration in range(args.iterations):
        source_raw=raw if iteration%2==0 else flipped
        if iteration:bridge.next_frame()
        bridge.upload_fixture(source_raw);phase(f'frame{iteration}_d3d12_input_texture_and_fence_complete')
        bridge.to_tensor(value);wait();phase(f'frame{iteration}_shared_buffer_to_gpu_tensor_complete')
        # Diagnostic input readback, not a neural fallback or production image path.
        input_exact=value.cpu().numpy().tobytes()==source_raw
        if not input_exact: raise RuntimeError('shared input layout mismatch; stop before neural work')
        start=time.perf_counter()
        if runtime:
            event=runtime.submit(value,frame_id=iteration,generation=1,frame_seed=0)
            wait(event);output=runtime.complete()
            if output is None: raise RuntimeError('completion disagreement')
        else:
            output=value.flip(1).contiguous();wait()
        inference_ms=(time.perf_counter()-start)*1000
        phase(f'frame{iteration}_native_frame_or_transport_transform_complete')
        bridge.from_tensor(output);copy_done=wait();phase(f'frame{iteration}_hip_output_fence_complete')
        actual=bridge.read_fixture();phase(f'frame{iteration}_d3d12_output_texture_consumed')
        expected=output.cpu().numpy().tobytes()
        finite=bool(torch.isfinite(output).all())
        if actual!=expected or not finite: raise RuntimeError('consumer output differs or nonfinite')
        if not runtime and actual!=(flipped if iteration%2==0 else raw): raise RuntimeError('pitched transport transform differs')
        if runtime:
            if iteration==0:
                baseline=Path('results/20260906_rocm_whole640_monitorv2_hundred/output.rgba16f') if w==640 else root/'output.rgba16f'
                baseline_exact=actual==baseline.read_bytes()
            if not runtime.release_output(copy_done): raise RuntimeError('output copy not completed')
        frames.append({'frame_id':iteration,'input_sha256':sha(source_raw),'output_sha256':sha(actual),
                       'frame_submit_and_wait_ms':inference_ms,'allocated_bytes':torch.cuda.memory_allocated(),
                       'input_exact':input_exact,'consumer_exact':True,'finite':finite})
        with (args.output/'completed_frames.jsonl').open('a') as f:f.write(json.dumps(frames[-1])+'\n')
        # Remove caller's references before next submit; runtime/transport are resident.
        del output,copy_done
        if runtime:del event
    alternating_consistent=all(len({f['output_sha256'] for f in frames if f['frame_id']%2==parity})==1
                               for parity in range(min(args.iterations,2)))
    changed_input_changes_output=args.iterations==1 or frames[0]['output_sha256']!=frames[1]['output_sha256']
    if runtime:runtime.close()
    bridge.close();phase('gpu_work_complete')
    with (args.output/'output.rgba16f').open('xb') as f:f.write(actual)
    return {'checks_pass':alternating_consistent and changed_input_changes_output,'backend':'pytorch_rocm','device':props.name,'architecture':props.gcnArchName,
            'mode':args.mode,'input_geometry':[w,h],'input_sha256':sha(source_raw),'output_sha256':sha(actual),
            'baseline_input_sha256':sha(raw),'baseline_comparison_frame_id':0,
            'input_transport_exact':input_exact,'d3d12_consumer_matches_gpu_tensor':True,'finite':finite,
            'same_candidate_staged_output_exact':baseline_exact,'frame_submit_and_wait_ms':inference_ms,
            'frames':frames,'iterations':args.iterations,'alternating_outputs_consistent':alternating_consistent,
            'changed_input_changes_output':changed_input_changes_output,
            'timing_scope':'whole-frame end-wait; first cold then warm, not game FPS',
            'original_model_sha256':ORIGINAL_SHA if runtime else None,'cpu_neural_fallback':False,
            'cpu_image_io_scope':'offline fixture upload and diagnostic readback only',
            'transport':description,'bridge_dll_sha256':sha(args.dll.read_bytes()),
            'native_graph_complete':False,'game_runtime_ready':False,'hdr_supported':False,
            'realtime_4k60_accepted':False,'training_started':False}


def child(args):
    trace=(args.output/'traceback.log').open('x');configure_fault_logging(trace)
    globals()['_trace']=trace
    def phase(name):
        with (args.output/'phases.jsonl').open('a') as f:
            f.write(json.dumps({'phase':name,'pid':os.getpid(),'monotonic':time.monotonic()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter')
    names=('native_model_package.py','native_frame_runtime.py','native_bridge_harness.py','validate_native_frame_bridge.py')
    sources={n:sha((Path(__file__).parent/n).read_bytes()) for n in names}
    save(args.output/'implementation_sources_at_start.json',sources)
    result=exercise(args,phase);phase('release_begin');gc.collect()
    import torch
    torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
    result['allocated_after_release_bytes']=torch.cuda.memory_allocated()
    result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
    result['source_files_changed_during_probe']=[n for n,s in sources.items() if sha((Path(__file__).parent/n).read_bytes())!=s]
    result['checks_pass'] &= not result['source_files_changed_during_probe'] and result['reserved_after_release_bytes']==0
    phase('resources_released');save(args.output/'child.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('transport','network'),required=True)
    p.add_argument('--width',type=int,choices=(128,640,641),default=128)
    p.add_argument('--iterations',type=int,choices=(1,12),default=1)
    p.add_argument('--dll',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.mode=='network' and a.width==641:p.error('odd network GPU budget not reviewed; transport only')
    if a.child:child(a)
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,str(Path(__file__).resolve()),'--child','--mode',a.mode,'--width',str(a.width),'--iterations',str(a.iterations),
                 '--dll',str(a.dll.resolve()),'--output',str(a.output.resolve())]
        raise SystemExit(0 if supervise(command,a.output,timeout=90) else 2)
