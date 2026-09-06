"""Bounded offline schedule A/B; does not authorize game deployment or quality.

Default changes only preblock/output-head batches; --middle-batch is separate
opt-in for other window stages. Retains formulas, weights, seed and full-frame
coordinates. Final pixels are read
back for diagnostic exact comparison; never a production frame transport.
"""
import argparse
import atexit
import gc
import hashlib
import json
import os
import statistics
from pathlib import Path
import sys
import time
from validate_rocm_lifecycle import supervise,save,configure_fault_logging


def sha(raw):return hashlib.sha256(raw).hexdigest().upper()


def fingerprints():
    root=Path(__file__).parent
    paths=set(root.glob('native_*.py'))|{Path(__file__),root/'validate_rocm_lifecycle.py',root/'diagnose_native_projection.py'}
    return {p.name:sha(p.read_bytes()) for p in sorted(paths)}


def case_spec(name):
    if name in ('128','640','641','3840'):
        root=Path({'128':'results/20260906_single_color_chain128_v1',
                   '640':'results/20260906_single_color_gamecrop640_v1',
                   '641':'results/20260906_single_color_odd641_v1',
                   '3840':'results/20260906_opt3_fixture4k_v1'}[name])
        m=json.loads((root/'manifest.json').read_bytes())
        return root/'input.rgba16f',m['input_geometry'],m['input_sha256'],m['input_source']
    if name=='2342':
        root=Path('results/20260906_reshade_network10_2342_one_v1')
        m=json.loads((root/'worker_frames.jsonl').read_text().splitlines()[0])
        return root/'audit_input1.rgba16f',[2342,1382],m['input_sha256'],'SYNTHETIC_HOST_STRIPES_NOT_TEACHER'
    raise ValueError('unreviewed input case')


def compare_bytes(reference,actual):
    import numpy as np
    a=np.frombuffer(reference,dtype='<f2').astype('f4')
    b=np.frombuffer(actual,dtype='<f2').astype('f4')
    if not len(a) or a.shape!=b.shape:raise ValueError('equal nonempty buffers required')
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:raise ValueError('nonfinite comparison')
    delta=b-a
    return {'bitwise_exact':reference==actual,'different_components':int(np.count_nonzero(delta)),
            'max_absolute_error':float(abs(delta).max()),'rmse':float(np.sqrt(np.mean(delta**2)))}


def exercise(args,phase):
    import torch
    from native_model_package import load_package
    from native_whole_frame import SingleColorWholeFrame
    from native_execution_policy import execution_policy
    from native_compact_layout import compact_layout
    torch.set_num_threads(2)
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('unreviewed GPU')
    # Allocator cap leaves 1 GB of the user's 6 GB envelope for non-tensor resources.
    # HIP/BLAS/D3D allocations are outside this cap; report device-wide samples too.
    memory_budget=args.allocator_budget_mb*1000000
    torch.cuda.set_per_process_memory_fraction(memory_budget/props.total_memory)
    free_initial,total_device=torch.cuda.mem_get_info()
    def wait():
        event=torch.cuda.Event();event.record()
        deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:raise TimeoutError('GPU timeout; no cancellation/retry')
            time.sleep(.001)
    phase('gpu_initialized')
    source,(w,h),expected,kind=case_spec(args.case)
    raw=source.read_bytes()
    if sha(raw)!=expected or len(raw)!=w*h*8:raise ValueError('input identity mismatch')
    package=Path('local_models/native_single_color_v1')
    manifest_sha='AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3'
    records,origins,settings=load_package(package,manifest_sha)
    model=SingleColorWholeFrame(records,origins,**settings,max_padded_pixels=8847360 if args.case=='3840' else 4000000).eval().cuda()
    model.requires_grad_(False)
    color=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(h,w,4).cuda()
    if any(t.device.type!='cuda' for t in [color,*model.parameters(),*model.buffers()]):
        raise RuntimeError('CPU neural tensor')
    wait();phase('weights_and_actual_input_ready')
    if args.pre_only:
        # Diagnostic wrappers only: hash function boundaries, return original tensors.
        # CPU readback is outside measured math's normal deployment path.
        import native_preblock as pre_module
        originals={}
        def observe(label,tensor):
            data=tensor.detach().cpu().numpy().tobytes()
            with (args.output/'pre_boundaries.jsonl').open('a') as stream:
                stream.write(json.dumps({'run':run,'boundary':label,'shape':list(tensor.shape),'sha256':sha(data)})+'\n')
        def wrap(name):
            original=getattr(pre_module,name);originals[name]=original
            def traced(*a,**kw):
                if name in ('pack_image','scatter_packed','average_pool2x2','pack_outview'):observe(name+'_input',a[0])
                value=original(*a,**kw)
                observe(name+'_output',value[0] if name=='gather_packed' else value)
                return value
            setattr(pre_module,name,traced)
        for name in ('single_color_features','pack_image','gather_packed','scatter_packed','average_pool2x2','pack_outview'):wrap(name)
    outputs=[];runs=[];reference_stages={};first_difference=None
    # First run of each schedule is labeled cold; no hot/cold speedup comparison.
    schedule=[12,12] if args.diagnose_repeat else [args.reference_batch,args.reference_batch]+[args.batch]*args.candidate_repeats
    for run,batch in enumerate(schedule):
        middle_batch=(12 if args.diagnose_repeat else args.reference_batch) if run<2 else args.middle_batch
        model.head.cache_layout_enabled=args.cache_head_layout and run>=2
        from native_swin_torch import RecoveredSwin32
        for module in model.modules():
            if isinstance(module,RecoveredSwin32):module.inference_cache_enabled=args.cache_head_weights and run>=2
        torch.cuda.reset_peak_memory_stats()
        stages=[];input_hash_before=sha(color.cpu().numpy().tobytes())
        if input_hash_before!=expected:raise RuntimeError('input mutated between runs')
        query_chunk=args.reference_query_chunk if run<2 else args.query_chunk
        use_compact=args.reference_compact_layout if run<2 else args.compact_layout
        with torch.no_grad(),execution_policy('native_fp16'),compact_layout(use_compact):
            iterator=iter(model.stages(color,0,window_batch=middle_batch,boundary_batch=batch,query_chunk=query_chunk))
            for index in range(1 if args.pre_only else 71):
                start=time.perf_counter()
                b,value=next(iterator)
                finite=torch.isfinite(value).all();wait()
                elapsed=(time.perf_counter()-start)*1000
                if b!=index or not bool(finite):raise RuntimeError('missing/nonfinite stage; no retry')
                item={'block':b,'host_submit_wait_finite_ms':elapsed}
                free_now,_=torch.cuda.mem_get_info()
                item['device_used_bytes_sample']=total_device-free_now
                if torch.cuda.max_memory_reserved()>memory_budget:
                    raise RuntimeError('allocator memory budget exceeded; stop without retry')
                if total_device-free_now>6000000000:
                    raise RuntimeError('sampled device-wide use exceeds 6 GB envelope; stop after completed stage')
                if args.diagnose_repeat:
                    stage_raw=value.cpu().numpy().tobytes()
                    item['sha256']=sha(stage_raw)
                    if run==0:reference_stages[b]=stage_raw
                    else:
                        item['same_stage_comparison']=compare_bytes(reference_stages[b],stage_raw)
                        if not item['same_stage_comparison']['bitwise_exact']:
                            first_difference={'block':b,'shape':list(value.shape),**item['same_stage_comparison']}
                            for label,data in [('first',reference_stages[b]),('second',stage_raw)]:
                                with (args.output/f'difference_block{b}_{label}.rgba16f').open('xb') as stream:stream.write(data)
                stages.append(item)
                phase(f'run{run}_batch{batch}_block{b}_complete')
                if first_difference:break
            if first_difference is None and not args.pre_only:
                try:next(iterator)
                except StopIteration:pass
                else:raise RuntimeError('unexpected extra stage')
        if first_difference:
            runs.append({'run':run,'boundary_batch':batch,'stages':stages,'input_sha256_before':input_hash_before})
            phase('first_different_stage_stop');break
        output=value.cpu().numpy().tobytes()
        if not outputs:outputs.append(output)
        comparison=compare_bytes(outputs[0],output)
        item={'run':run,'boundary_batch':batch,'middle_batch':middle_batch,'query_chunk':query_chunk,'compact_layout':use_compact,
              'cache_head_weights':args.cache_head_weights and run>=2,'schedule_warm':run==1 or run>=3,
              'cache_head_layout':args.cache_head_layout and run>=2,'head_layout_cache_bytes':model.head._layout_bytes,
              'stages':stages,'input_sha256_before':input_hash_before,
              'host_stage_sum_ms':sum(s['host_submit_wait_finite_ms'] for s in stages),
              'output_sha256':sha(output),'comparison_to_first_baseline':comparison,
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        runs.append(item)
        with (args.output/'runs.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
        if run<4 or run==len(schedule)-1:
            with (args.output/f'output{run}.rgba16f').open('xb') as stream:stream.write(output)
        # Numerical differences are a failed acceptance, not an excuse to retry.
        if not comparison['bitwise_exact']:
            phase('numerical_difference_stop');break
    phase('gpu_work_complete')
    if args.pre_only:
        for name,original in originals.items():setattr(pre_module,name,original)
    ok=first_difference is None and len(runs)==len(schedule) and all(r['comparison_to_first_baseline']['bitwise_exact'] for r in runs)
    return {'name':'baseline_stage_repeat_diagnostic' if args.diagnose_repeat else 'boundary_batch_ab',
            'checks_pass':ok,'first_different_stage':first_difference,'runs':runs,'input_geometry':[w,h],
            'input_source':kind,'input_sha256':expected,'package_manifest_sha256':manifest_sha,
            'backend':'pytorch_rocm','device':props.name,'architecture':props.gcnArchName,
            'cpu_neural_fallback':False,'weights_modified':False,'candidate_middle_batch':args.middle_batch,
            'allocator_budget_bytes':memory_budget,'device_used_before_model_bytes':total_device-free_initial,
            'memory_scope':'allocator capped; device-wide used sampled at stage boundaries, not process total or an unsampled peak',
            'warm_speedup':runs[1]['host_stage_sum_ms']/statistics.median(r['host_stage_sum_ms'] for r in runs[3:]) if ok and not args.diagnose_repeat else None,
            'timing_scope':'one warm baseline versus candidate warm median; staged host wait+finite check, NOT pure GPU/game FPS',
            'rtx_quality_accepted':False,'hdr_accepted':False,'game_runtime_ready':False}


def child(args):
    initial=fingerprints();save(args.output/'sources_at_start.json',initial)
    trace=(args.output/'traceback.log').open('x');configure_fault_logging(trace)
    globals()['_trace']=trace
    def phase(name):
        with (args.output/'phases.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase':name,'pid':os.getpid(),'monotonic':time.monotonic()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter')
    result=exercise(args,phase)
    result['sources_unchanged']=initial==fingerprints()
    import torch
    gc.collect();torch.cuda.empty_cache()
    if hasattr(torch._C,'_cuda_clearCublasWorkspaces'):torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    result['allocated_after_release_bytes']=torch.cuda.memory_allocated()
    result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
    result['checks_pass'] &= result['sources_unchanged'] and result['reserved_after_release_bytes']==0
    phase('resources_released');save(args.output/'child.json',result)
    return result['checks_pass']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=('128','640','641','2342','3840'),required=True)
    p.add_argument('--batch',type=int,choices=(48,96,192,384,768,1536,3072),default=96)
    p.add_argument('--reference-batch',type=int,choices=(12,96,384,768,1536),default=12)
    p.add_argument('--query-chunk',type=int,choices=(32,128,256,512,1024),default=32)
    p.add_argument('--reference-query-chunk',type=int,choices=(32,128),default=32)
    p.add_argument('--compact-layout',action='store_true')
    p.add_argument('--reference-compact-layout',action='store_true')
    p.add_argument('--allocator-budget-mb',type=int,choices=(4000,5000),default=5000,
                   help='decimal MB, allocator only; reserve headroom for non-tensor resources')
    p.add_argument('--candidate-repeats',type=int,choices=(2,12),default=2)
    p.add_argument('--cache-head-weights',action='store_true')
    p.add_argument('--cache-head-layout',action='store_true')
    p.add_argument('--middle-batch',type=int,choices=(12,96,192,384,768,1536,3072),default=12)
    p.add_argument('--diagnose-repeat',action='store_true',help='baseline only; snapshot first different stage, no optimized run')
    p.add_argument('--pre-only',action='store_true',help='diagnose only block0 with function boundary readbacks; not a timing benchmark')
    p.add_argument('--child',action='store_true');args=p.parse_args()
    if args.pre_only and not args.diagnose_repeat:raise ValueError('pre-only requires explicit repeat diagnosis')
    if args.child:return child(args)
    args.output.mkdir(parents=True,exist_ok=False)
    return supervise([sys.executable,str(Path(__file__).resolve()),'--child','--output',str(args.output.resolve()),
                      '--case',args.case,'--batch',str(args.batch),'--middle-batch',str(args.middle_batch),'--reference-batch',str(args.reference_batch),
                      '--query-chunk',str(args.query_chunk),'--candidate-repeats',str(args.candidate_repeats)]+
                     ['--reference-query-chunk',str(args.reference_query_chunk)]+(['--compact-layout'] if args.compact_layout else [])+
                     ['--allocator-budget-mb',str(args.allocator_budget_mb)]+(['--reference-compact-layout'] if args.reference_compact_layout else [])+
                     (['--cache-head-weights'] if args.cache_head_weights else [])+(['--diagnose-repeat'] if args.diagnose_repeat else [])+
                     (['--cache-head-layout'] if args.cache_head_layout else [])+
                     (['--pre-only'] if args.pre_only else []),
                     args.output,timeout=180 if args.case in ('2342','3840') else 90)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
