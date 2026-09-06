"""One-window strict gate for the bounded native Decoder scale transition.

This is deliberately not a stress test and never invokes RGP/counter capture.
"""
from __future__ import annotations

import argparse,atexit,gc,hashlib,json,os,sys,time
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from validate_rocm_lifecycle import save,supervise


MODEL_MANIFEST='AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3'
BOUNDARIES={256:48,128:56,64:62,32:66}


def digest(tensor):return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest().upper()


def compare(reference,candidate):
    import torch
    delta=candidate.float()-reference.float()
    different=int(torch.count_nonzero(reference.view(torch.int16)!=candidate.view(torch.int16)).item())
    rmse=float(torch.sqrt(torch.mean(delta.square())).item())
    denominator=float(torch.sqrt(torch.mean(reference.float().square())).item())
    return {'bitwise_exact':different==0,'different_components':different,
            'max_absolute_error':float(delta.abs().max().item()),'rmse':rmse,
            'nrmse':rmse/denominator if denominator else (0.0 if rmse==0 else None)}


def wait(event):
    deadline=time.monotonic()+10
    while not event.query():
        if time.monotonic()>deadline:raise TimeoutError('GPU event timeout; stop without retry')
        time.sleep(.001)


def exercise(args,phase):
    import torch
    from native_compact_layout import compact_layout
    from native_execution_policy import execution_policy
    from native_model_package import load_package
    from native_multiscale import pack_image,pack_outview
    from native_transition_fusion import EncoderTransitionFusion
    from native_upsample_swin import RecoveredUpsampleSwin
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm GPU required; no fallback')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('unreviewed GPU')
    torch.set_num_threads(2);torch.manual_seed(1201);phase('gpu_initialized')
    records,_,_=load_package('local_models/native_single_color_v1',MODEL_MANIFEST)
    c=args.channels;block=BOUNDARIES[c]
    model=RecoveredUpsampleSwin(records[(block,0)],record_kind=f'swin{c}').eval().cuda().requires_grad_(False)
    h=w=8
    low_logical=torch.linspace(-1.5,1.5,h//2*(w//2)*2*c,device='cuda',dtype=torch.float16).reshape(h//2,w//2,2*c)
    skip_logical=torch.linspace(1.25,-1.25,h*w*c,device='cuda',dtype=torch.float16).reshape(h,w,c)
    fusion=EncoderTransitionFusion(args.dll)
    stream=torch.cuda.current_stream()
    with torch.no_grad(),execution_policy('native_fp16'),compact_layout(True):
        low_outview=pack_outview(low_logical);low_resident=pack_image(low_logical);skip=pack_image(skip_logical)
        torch.cuda.reset_peak_memory_stats();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record(stream);reference=pack_image(model.fuse(low_outview,skip,w,h));end.record(stream);wait(end)
        baseline_ms=begin.elapsed_time(end);baseline_peak=torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats();begin2,end2=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin2.record(stream);candidate=fusion.decoder(low_resident,skip,w,h,c,model.permutation,model.project,model.skip_scale)
        end2.record(stream);wait(end2);candidate_ms=begin2.elapsed_time(end2);candidate_peak=torch.cuda.max_memory_allocated()
        finite=bool(torch.isfinite(candidate).all());comparison=compare(reference,candidate)
        result={'checks_pass':finite and comparison['bitwise_exact'],
                'scope':'one 8x8 destination window, one baseline and one candidate submission; no profiler or stress loop',
                'channels':c,'source_geometry':[w//2,h//2,c*2],'target_geometry':[w,h,c],
                'baseline_gpu_event_ms':baseline_ms,'candidate_gpu_event_ms':candidate_ms,
                'timing_suitable_for_throughput_claim':False,
                'baseline_peak_allocated_bytes':baseline_peak,'candidate_peak_allocated_bytes':candidate_peak,
                'comparison':comparison,'reference_sha256':digest(reference),'candidate_sha256':digest(candidate),
                'authored_counts':fusion.counts(),'finite':finite,'device':props.name,
                'weights_modified':False,'default_promoted':False,'rgp_used':False}
    del low_logical,skip_logical,low_outview,low_resident,skip,reference,candidate,model,fusion,begin,end,begin2,end2
    phase('gpu_work_complete');gc.collect();torch.cuda.empty_cache()
    if hasattr(torch._C,'_cuda_clearCublasWorkspaces'):torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();result['allocated_after_release_bytes']=torch.cuda.memory_allocated();result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
    result['checks_pass']&=result['allocated_after_release_bytes']==result['reserved_after_release_bytes']==0
    phase('resources_released');return result


def child(args):
    def phase(name):
        with (args.output/'phases.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'phase':name,'pid':os.getpid(),'monotonic':time.monotonic()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter')
    result=exercise(args,phase);save(args.output/'child.json',result);return result['checks_pass']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--channels',type=int,choices=tuple(BOUNDARIES),default=32);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--child',action='store_true');args=p.parse_args()
    if args.child:return child(args)
    require_gpu_tests_enabled('one-window decoder transition gate')
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(args.dll.resolve()),
             '--channels',str(args.channels),'--output',str(args.output.resolve())]
    return supervise(command,args.output,timeout=90)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
