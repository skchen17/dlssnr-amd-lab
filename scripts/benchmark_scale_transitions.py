"""Reproducible non-RGP A-B-B-A benchmark for one resolution's scale boundaries.

Run only after the corresponding one-window correctness gates pass. HIP events
measure stream intervals; actual kernel counts/busy sums and hardware counters
remain unavailable while the RGP safety lock is active.
"""
from __future__ import annotations

import argparse,atexit,gc,hashlib,json,math,os,statistics,sys,time
from pathlib import Path

from audit_scale_transitions import SIZES,audit
from gpu_safety import require_gpu_tests_enabled
from validate_rocm_lifecycle import save,supervise


MODEL_MANIFEST='AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3'
ENCODER_BLOCKS={32:4,64:8,128:14,256:22}
DECODER_BLOCKS={256:48,128:56,64:62,32:66}


def abba_schedule(groups=2):
    if groups<=0:raise ValueError('positive A-B-B-A group count required')
    return [label for _ in range(groups) for label in ('reference','native','native','reference')]


def wait(event):
    deadline=time.monotonic()+30
    while not event.query():
        if time.monotonic()>deadline:raise TimeoutError('GPU event timeout; stop without cancellation or retry')
        time.sleep(.001)


def digest(value):
    if isinstance(value,dict):return {k:digest(value[k]) for k in ('skip','resident')}
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest().upper()


def exact(reference,candidate):
    import torch
    names=('skip','resident') if isinstance(reference,dict) else (None,)
    rows={}
    for name in names:
        a=reference[name] if name else reference;b=candidate[name] if name else candidate
        delta=b.float()-a.float();different=int(torch.count_nonzero(a.view(torch.int16)!=b.view(torch.int16)).item())
        rmse=float(torch.sqrt(torch.mean(delta.square())).item());den=float(torch.sqrt(torch.mean(a.float().square())).item())
        rows[name or 'resident']={'bitwise_exact':different==0,'different_components':different,
            'max_absolute_error':float(delta.abs().max().item()),'rmse':rmse,
            'nrmse':rmse/den if den else (0.0 if rmse==0 else None)}
    return rows


def timed_phase(torch,fn,iterations):
    stream=torch.cuda.current_stream();times=[];torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
    output=None
    for _ in range(iterations):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record(stream);output=fn();end.record(stream);wait(end);elapsed=begin.elapsed_time(end)
        if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid HIP event duration; stop without retry')
        times.append(elapsed)
    peak=torch.cuda.max_memory_allocated()
    return output,{'gpu_event_ms':times,'median_gpu_event_ms':statistics.median(times),
        'peak_allocated_bytes':peak,'incremental_peak_allocated_bytes':peak-before}


def geometry(size,kind,channels):
    width,height=SIZES[size];pw,ph=(width+127)//128*128,(height+127)//128*128
    if kind=='encoder':
        level=(32,64,128,256).index(channels)+1
        return pw//(2**level),ph//(2**level)
    divisor={256:16,128:8,64:4,32:2}[channels]
    return pw//divisor,ph//divisor


def exercise(args,phase):
    import torch
    from native_compact_layout import compact_layout
    from native_execution_policy import execution_policy
    from native_model_package import load_package
    from native_multiscale import RecoveredDownsampleSwin,average_pool2x2,pack_image,pack_outview,unpack_image
    from native_split_swin512 import chunked_linear
    from native_swin_torch import quantize_e4
    from native_transition_fusion import EncoderTransitionFusion
    from native_upsample_swin import RecoveredUpsampleSwin
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required; no fallback')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('unreviewed GPU')
    torch.set_num_threads(2);torch.manual_seed(1201)
    torch.cuda.set_per_process_memory_fraction(5000000000/props.total_memory);phase('gpu_initialized')
    records,_,_=load_package('local_models/native_single_color_v1',MODEL_MANIFEST)
    static=audit(Path(__file__).resolve().parents[1])['sizes'][args.size][args.kind]
    selected=(32,64,128,256) if args.kind=='encoder' else (256,128,64,32)
    rows=[]
    with torch.no_grad(),execution_policy('native_fp16'),compact_layout(True):
        for channels in selected:
            width,height=geometry(args.size,args.kind,channels);fusion=EncoderTransitionFusion(args.dll)
            owned=[]
            if args.kind=='encoder':
                model=RecoveredDownsampleSwin(records[(ENCODER_BLOCKS[channels],0)],record_kind=f'swin{channels}').eval().cuda().requires_grad_(False)
                logical=torch.empty((height,width,channels),device='cuda',dtype=torch.float16).uniform_(-2,2)
                source=pack_image(logical)
                owned.extend((logical,source))
                def reference_fn():
                    pooled=average_pool2x2(unpack_image(source,width,height,channels))
                    down=quantize_e4(chunked_linear(pooled[...,model.permutation],model.pool_project))
                    return {'skip':quantize_e4(source),'resident':pack_image(down)}
                def native_fn():return fusion.encoder(source,width,height,channels,model.permutation,model.pool_project,capture_layouts=False)
            else:
                model=RecoveredUpsampleSwin(records[(DECODER_BLOCKS[channels],0)],record_kind=f'swin{channels}').eval().cuda().requires_grad_(False)
                low_logical=torch.empty((height//2,width//2,channels*2),device='cuda',dtype=torch.float16).uniform_(-2,2)
                skip_logical=torch.empty((height,width,channels),device='cuda',dtype=torch.float16).uniform_(-2,2)
                low_outview=pack_outview(low_logical);low_resident=pack_image(low_logical);skip=pack_image(skip_logical)
                owned.extend((low_logical,skip_logical,low_outview,low_resident,skip))
                def reference_fn():return pack_image(model.fuse(low_outview,skip,width,height))
                def native_fn():return fusion.decoder(low_resident,skip,width,height,channels,model.permutation,model.project,model.skip_scale)
            # Warmup is excluded and immediately enforces the strict gate.
            warm_reference=reference_fn();warm_native=native_fn();warm_end=torch.cuda.Event();warm_end.record();wait(warm_end)
            comparison=exact(warm_reference,warm_native)
            if not all(v['bitwise_exact'] for v in comparison.values()):raise RuntimeError(f'{args.kind} C{channels} warm correctness failed')
            del warm_native
            phases=[]
            for label in abba_schedule():
                before=fusion.counts();output,timing=timed_phase(torch,reference_fn if label=='reference' else native_fn,args.iterations)
                if digest(output)!=digest(warm_reference):raise RuntimeError(f'{args.kind} C{channels} output changed')
                after=fusion.counts();timing.update({'implementation':label,'output_sha256':digest(output),
                    'authored_count_delta':{k:after[k]-before[k] for k in before}});phases.append(timing);del output
            reference_samples=[x for p in phases if p['implementation']=='reference' for x in p['gpu_event_ms']]
            native_samples=[x for p in phases if p['implementation']=='native' for x in p['gpu_event_ms']]
            static_row=next(row for row in static if row['transition']==(f'C{channels}->C{channels*2}' if args.kind=='encoder' else f'C{channels*2}->C{channels}'))
            rows.append({'transition':static_row['transition'],'geometry':[width,height,channels],
                'reference_median_gpu_event_ms':statistics.median(reference_samples),
                'native_median_gpu_event_ms':statistics.median(native_samples),
                'speedup':statistics.median(reference_samples)/statistics.median(native_samples),
                'phases':phases,'correctness':comparison,'static_storage_audit':static_row,
                'actual_gpu_dispatch_count':None,'kernel_busy_sum_ms':None,'dram_l2_counters':None,
                'counter_unavailable_reason':'RGP counter collection safety lock remains active'})
            del model,fusion,warm_reference,warm_end,reference_fn,native_fn,owned
            gc.collect();torch.cuda.empty_cache();phase(f'{args.kind}_c{channels}_completed')
    return {'checks_pass':all(all(v['bitwise_exact'] for v in row['correctness'].values()) for row in rows),
        'scope':f'{args.size} {args.kind} transition A-B-B-A, two groups, {args.iterations} iterations per phase',
        'rows':rows,'device':props.name,'rgp_used':False,'automatic_retry':False,
        'timing_scope':'HIP event stream interval for isolated transition; excludes validation readback',
        'whole_frame_time_included':False,'weights_modified':False,'default_promoted':False}


def child(args):
    def phase(name):
        with (args.output/'phases.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'phase':name,'pid':os.getpid(),'monotonic':time.monotonic()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter');result=exercise(args,phase);phase('gpu_work_complete')
    import torch
    gc.collect();torch.cuda.empty_cache()
    if hasattr(torch._C,'_cuda_clearCublasWorkspaces'):torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();result['allocated_after_release_bytes']=torch.cuda.memory_allocated();result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
    result['checks_pass']&=result['allocated_after_release_bytes']==result['reserved_after_release_bytes']==0
    phase('resources_released');save(args.output/'child.json',result);return result['checks_pass']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--size',choices=tuple(SIZES),required=True)
    p.add_argument('--kind',choices=('encoder','decoder'),required=True);p.add_argument('--iterations',type=int,choices=(1,12),default=12)
    p.add_argument('--ack-bounded-performance',action='store_true');p.add_argument('--child',action='store_true');args=p.parse_args()
    if args.child:return child(args)
    if not args.ack_bounded_performance:p.error('explicit --ack-bounded-performance is required')
    require_gpu_tests_enabled(f'{args.size} {args.kind} transition performance gate')
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(args.dll.resolve()),'--output',str(args.output.resolve()),
             '--size',args.size,'--kind',args.kind,'--iterations',str(args.iterations),'--ack-bounded-performance']
    return supervise(command,args.output,timeout=600)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
