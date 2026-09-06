"""Bounded full-frame native NR benchmark; FSR/quality claims are forbidden here.

CPU area resampling creates explicitly synthetic lower-resolution inputs from
one fixed synthetic4K source before GPU work. This is NOT the FSR output stage.
Uninstrumented full-forward timing and a separate ATen-counted pass are distinct.
"""
import argparse,atexit,gc,hashlib,json,math,os,statistics,sys,time
from pathlib import Path
from validate_rocm_lifecycle import supervise,save,configure_fault_logging


def sha(raw):return hashlib.sha256(raw).hexdigest().upper()


def fixture(output,size):
    import torch
    torch.set_num_threads(2)
    if size in ('128','640'):
        root=Path('results/20260906_single_color_chain128_v1' if size=='128' else 'results/20260906_single_color_gamecrop640_v1')
        info=json.loads((root/'manifest.json').read_bytes());raw=(root/'input.rgba16f').read_bytes()
        if sha(raw)!=info['input_sha256']:raise ValueError('source fixture hash mismatch')
        w,h=info['input_geometry'];source_kind=info['input_source'];source_sha=sha(raw)
    else:
        root=Path('results/20260906_opt3_fixture4k_v1')
        raw=(root/'input.rgba16f').read_bytes();info=json.loads((root/'manifest.json').read_bytes())
        if sha(raw)!=info['input_sha256']:raise ValueError('source fixture hash mismatch')
        source_sha=sha(raw);w,h={'1080':(1920,1080),'1440':(2560,1440),'2160':(3840,2160),
                             '641':(641,361),'2342':(2342,1382),'3440':(3440,1440)}[size]
        if size!='2160':
            x=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(2160,3840,4).permute(2,0,1)[None].float()
            raw=torch.nn.functional.interpolate(x,size=(h,w),mode='area')[0].permute(1,2,0).half().contiguous().numpy().tobytes()
        source_kind='SYNTHETIC_4K_GRADIENT_CHECKER_CPU_AREA_DERIVATIVE_NOT_GAME_OR_TEACHER'
    with (output/'input.rgba16f').open('xb') as f:f.write(raw)
    save(output/'input.json',{'geometry':[w,h],'sha256':sha(raw),'source_sha256':source_sha,'source':source_kind,
                             'fsr_applied':False,'resampling':'identity' if size in ('128','640','2160') else 'CPU_AREA_FLOAT32_THEN_FP16'})


def exercise(a,phase):
    import torch
    from native_whole_frame import SingleColorWholeFrame
    from native_model_package import load_package
    from native_compact_layout import compact_layout
    from native_execution_policy import execution_policy
    from nr_operator_counter import OperatorCounter
    from native_fusion_policy import fusion_policy
    from native_head_fusion import head_input_fusion
    from native_pre_fusion import pre_features_fusion
    from native_c32_layout_fusion import c32_layout_fusion
    from native_matrix_fusion import matrix_fusion
    from native_grid_policy import grid_policy
    from native_transition_policy import transition_policy
    from native_transition_fusion import transition_fusion
    torch.set_num_threads(2)
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('unreviewed GPU')
    torch.cuda.set_per_process_memory_fraction(5000000000/props.total_memory)
    def wait(event=None):
        if event is None:event=torch.cuda.Event();event.record()
        deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:raise TimeoutError('GPU uncertain; no cancel/retry')
            time.sleep(.001)
    phase('gpu_initialized')
    # Timer sanity check only. Event span is not summed GPU busy time.
    clock=[]
    with torch.no_grad():
        x=torch.ones((1024,1024),device='cuda',dtype=torch.float16)
        y=x@x;wait()
        for _ in range(2):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            wait();start=time.perf_counter();begin.record()
            for i in range(16):y=x@x
            end.record();wait(end);host=(time.perf_counter()-start)*1000
            elapsed=begin.elapsed_time(end)
            clock.append({'gpu_event_ms':elapsed,'host_submit_wait_ms':host,
                          'plausible':math.isfinite(elapsed) and 0<elapsed<=host*1.25})
        del x,y,begin,end
    clock_ok=all(c['plausible'] for c in clock)
    phase('clock_sanity_completed')
    meta=json.loads((a.output/'input.json').read_bytes());w,h=meta['geometry']
    raw=(a.output/'input.rgba16f').read_bytes()
    if sha(raw)!=meta['sha256']:raise ValueError('input changed')
    records,origins,settings=load_package('local_models/native_single_color_v1','AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    model=SingleColorWholeFrame(records,origins,**settings,max_padded_pixels=8847360).eval().cuda().requires_grad_(False)
    model.head.block.inference_cache_enabled=a.head_weight_cache
    color=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(h,w,4).cuda()
    if any(t.device.type!='cuda' for t in [color,*model.parameters(),*model.buffers()]):raise RuntimeError('CPU neural tensor')
    wait();phase('model_and_input_ready')
    runs=[];reference=None
    if a.reference_output:
        refmeta=json.loads((a.reference_output.parent/'input.json').read_bytes())
        if refmeta['sha256']!=meta['sha256'] or refmeta['geometry']!=meta['geometry']:raise ValueError('reference input differs')
        reference=sha(a.reference_output.read_bytes())
    grid_families=tuple(filter(None,a.whole_grid_families.split(',')))
    with torch.no_grad(),execution_policy('native_fp16'),compact_layout(True),fusion_policy(a.fusion_dll) as fused,head_input_fusion(a.head_input_dll,gather=a.head_gather,epilogue=a.head_epilogue,qkv=a.head_qkv) as head_fused,pre_features_fusion(a.pre_features_dll,project_pack=a.pre_project_pack) as pre_fused,c32_layout_fusion(a.c32_layout_dll) as c32_fused,matrix_fusion(a.matrix_dll,a.matrix_profile,modules=tuple(a.matrix_modules.split(',')),waves=a.matrix_waves) as matrix,grid_policy(grid_families),transition_policy(resident=a.resident_transitions,capture_outviews=a.capture_transition_outviews),transition_fusion(a.transition_dll,encoder=a.encoder_transition,decoder=a.decoder_transition) as transitions:
        for index in range(a.iterations):
            wait();torch.cuda.reset_peak_memory_stats()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start=time.perf_counter();begin.record()
            value=model(color,0,window_batch=768,query_chunk=1024)
            end.record();wait(end);host=(time.perf_counter()-start)*1000
            event_ms=begin.elapsed_time(end)
            # Validation, finite scan and readback are OUTSIDE measured interval.
            finite=torch.isfinite(value).all();wait()
            if not bool(finite):raise RuntimeError('nonfinite output')
            out=value.cpu().numpy().tobytes()
            if reference is None:reference=sha(out)
            if index==0:
                with (a.output/'output.rgba16f').open('xb') as f:f.write(out)
            if sha(out)!=reference:raise RuntimeError('same input changed output; stop')
            free,total=torch.cuda.mem_get_info()
            item={'index':index,'warm':index>0,'host_forward_submit_wait_ms':host,'gpu_event_raw_ms':event_ms,
                  'gpu_stream_elapsed_ms':event_ms if clock_ok and math.isfinite(event_ms) and 0<event_ms<=host*1.25 else None,
                  'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                  'device_used_bytes_sample':total-free,'output_sha256':reference}
            item['authored_fusions_cumulative']=fused.counts() if fused else None
            item['head_input_launches_cumulative']=head_fused.launches if head_fused else 0
            item['head_compose_launches_cumulative']=head_fused.compose_launches if head_fused else 0
            item['head_gather_launches_cumulative']=head_fused.gather_launches if head_fused else 0
            item['head_epilogue_launches_cumulative']=head_fused.add_launches if head_fused else 0
            item['head_qkv_launches_cumulative']=head_fused.qkv_launches if head_fused else 0
            item['matrix_launches_cumulative']=matrix.launches if matrix else 0
            item['pre_features_launches_cumulative']=pre_fused.launches if pre_fused else 0
            item['pre_project_pack_launches_cumulative']=pre_fused.project_pack_launches if pre_fused else 0
            item['c32_layout_launches_cumulative']={'gather':c32_fused.gather_calls,'scatter':c32_fused.scatter_calls} if c32_fused else None
            item['transition_counts_cumulative']=transitions.counts() if transitions else None
            if item['peak_reserved_bytes']>5000000000 or total-free>6000000000:raise RuntimeError('memory envelope exceeded')
            runs.append(item)
            with (a.output/'runs.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            phase(f'frame{index}_validated')
            del value,finite,out,begin,end
        counter=OperatorCounter()
        if a.drain_stages:
            # Diagnostic only: per-stage draining changes overlap/scheduling.
            # Never sum these measurements and label them normal forward time.
            drained=[]
            for repeat in range(3):
                iterator=iter(model.stages(color,0,window_batch=768,query_chunk=1024))
                for b in range(71):
                    wait();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter();actual,value=next(iterator);wait()
                    wall=(time.perf_counter()-start)*1000
                    if actual!=b:raise RuntimeError('drained stage order mismatch')
                    drained.append({'repeat':repeat,'block':b,'host_submit_wait_ms':wall,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                if sha(value.cpu().numpy().tobytes())!=reference:raise RuntimeError('drained pass differs')
                del iterator,value
                phase(f'drained_pass{repeat}_validated')
            save(a.output/'drained_stages.json',{'scope':'isolated stage submit/wait; perturbs scheduling; NOT GPU busy time or additive forward time',
                 'runs':drained,'all_outputs_exact':True})
        if a.measure_head:
            head_runs=[]
            for repeat in range(3):
                iterator=iter(model.stages(color,0,window_batch=768,query_chunk=1024))
                for b in range(a.measure_block):actual,value=next(iterator)
                wait();torch.cuda.reset_peak_memory_stats()
                prior=fused.counts() if fused else {}
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();begin.record();actual,value=next(iterator);end.record();wait(end)
                wall=(time.perf_counter()-start)*1000;event=begin.elapsed_time(end)
                peak_live=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
                if actual!=a.measure_block:raise RuntimeError('isolated module order differs')
                after=fused.counts() if fused else {}
                for _,value in iterator:pass
                wait()
                if sha(value.cpu().numpy().tobytes())!=reference:raise RuntimeError('isolated module output differs')
                head_runs.append({'host_submit_wait_ms':wall,'gpu_stream_ms':event,
                    'gpu_event_accepted_for_module_cost':False,
                    'gpu_event_note':'Driver event spans can omit host submission gaps; use drained host submit/wait for module cost',
                    'allocator_peak_live_bytes':peak_live,
                    'allocator_peak_reserved_bytes':peak_reserved,
                    'authored_counts':{k:after[k]-prior[k] for k in prior if isinstance(prior[k],int)},
                    'gpu_total_dispatches':None,'output_exact':True})
                del iterator,value,begin,end
            name='head_times.json' if a.measure_block==70 else f'module{a.measure_block}_times.json'
            save(a.output/name,{'scope':f'separate passes with GPU drain before and after block{a.measure_block}; stream interval includes submission gaps', 'runs':head_runs})
        if a.count_operators:
            # Instrumented run is NOT a latency sample or GPU dispatch measurement.
            with counter:
                iterator=iter(model.stages(color,0,window_batch=768,query_chunk=1024))
                for b in range(71):
                    counter.block=b;actual,value=next(iterator)
                    if actual!=b:raise RuntimeError('stage order mismatch')
                try:next(iterator)
                except StopIteration:pass
                else:raise RuntimeError('extra stage')
            wait()
            if sha(value.cpu().numpy().tobytes())!=reference:raise RuntimeError('instrumentation changed output')
            save(a.output/'aten_counts.json',counter.report());del iterator,value
            phase('instrumented_run_validated')
        if a.stage_timestamps:
            wait();events=[];submits=[];iterator=iter(model.stages(color,0,window_batch=768,query_chunk=1024));pass_start=time.perf_counter()
            for b in range(71):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();submit_start=time.perf_counter();actual,value=next(iterator);submit_ms=(time.perf_counter()-submit_start)*1000;end.record()
                if actual!=b:raise RuntimeError('timestamp stage mismatch')
                events.append((b,begin,end))
                submits.append(submit_ms)
            try:next(iterator)
            except StopIteration:pass
            else:raise RuntimeError('extra timestamp stage')
            wait();pass_wall=(time.perf_counter()-pass_start)*1000
            if sha(value.cpu().numpy().tobytes())!=reference:raise RuntimeError('timestamp pass changed output')
            stage_times=[{'block':b,'gpu_stream_interval_ms':begin.elapsed_time(end),'host_stage_submit_ms':submits[b]} for b,begin,end in events]
            whole=events[0][1].elapsed_time(events[-1][2])
            gaps=[{'before_block':events[i][0],'gpu_stream_gap_ms':events[i-1][2].elapsed_time(events[i][1])} for i in range(1,len(events))]
            partition=sum(t['gpu_stream_interval_ms'] for t in stage_times)+sum(t['gpu_stream_gap_ms'] for t in gaps)
            coherent=math.isfinite(whole) and 0<whole<=pass_wall*1.25 and abs(partition-whole)<1 and all(t['gpu_stream_gap_ms']>=0 for t in gaps)
            save(a.output/'stage_times.json',{'scope':'event spans in a separate uncounted pass, no per-stage wait; includes stream idle gaps, NOT kernel busy sum',
                 'clock_sanity_pass':clock_ok,'partition_coherent':coherent,'whole_stream_ms':whole,'pass_host_submit_wait_ms':pass_wall,
                 'inter_stage_gaps':gaps,'valid':clock_ok and coherent and all(math.isfinite(t['gpu_stream_interval_ms']) and t['gpu_stream_interval_ms']>=0 for t in stage_times),'stages':stage_times})
            del events,iterator,value,begin,end
            phase('timestamp_pass_validated')
    phase('gpu_work_complete')
    return {'checks_pass':True,'input':meta,'runs':runs,'clock_sanity':clock,'clock_sanity_pass':clock_ok,
            'host_warm_median_ms':statistics.median(r['host_forward_submit_wait_ms'] for r in runs[1:]) if len(runs)>1 else None,
            'gpu_dispatch_count':None,'gpu_busy_kernel_sum_ms':None,'gpu_profiler_available':False,
            'fusion_dll_sha256':sha(a.fusion_dll.read_bytes()) if a.fusion_dll else None,
            'head_input_dll_sha256':sha(a.head_input_dll.read_bytes()) if a.head_input_dll else None,
            'matrix_dll_sha256':sha(a.matrix_dll.read_bytes()) if a.matrix_dll else None,
            'matrix_profile':a.matrix_profile,'matrix_waves':a.matrix_waves,
            'matrix_modules':a.matrix_modules,
            'pre_features_dll_sha256':sha(a.pre_features_dll.read_bytes()) if a.pre_features_dll else None,
            'pre_project_pack':a.pre_project_pack,'whole_grid_families':list(grid_families),
            'resident_transitions':a.resident_transitions,'capture_transition_outviews':a.capture_transition_outviews,
            'transition_dll_sha256':sha(a.transition_dll.read_bytes()) if a.transition_dll else None,
            'encoder_transition':a.encoder_transition,
            'decoder_transition':a.decoder_transition,
            'c32_layout_dll_sha256':sha(a.c32_layout_dll.read_bytes()) if a.c32_layout_dll else None,
            'reference_output_sha256':sha(a.reference_output.read_bytes()) if a.reference_output else None,
            'timing_scope':'Full-forward host submit/wait excludes finite checks and image readback; event span includes stream idle gaps and is not kernel busy sum',
            'fsr_applied':False,'rtx_quality_accepted':False,'temporal_quality_accepted':False,'game_runtime_ready':False}


def child(a):
    trace=(a.output/'traceback.log').open('x');configure_fault_logging(trace);globals()['_trace']=trace
    paths=list(Path(__file__).parent.glob('native_*.py'))+[Path(__file__),Path(__file__).with_name('nr_operator_counter.py')]
    sources={str(p):sha(p.read_bytes()) for p in paths};save(a.output/'sources.json',sources)
    def phase(name):
        with (a.output/'phases.jsonl').open('a') as f:f.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
    atexit.register(phase,'python_atexit')
    import torch,traceback
    try:
        result=exercise(a,phase)
    except BaseException as error:
        # Do not leave traceback frames retaining multi-GB CUDA tensors until
        # interpreter teardown. Record failure, then make one normal cleanup
        # attempt; never retry the workload or claim the GPU was cancelled.
        traceback.print_exc(file=trace);trace.flush();phase('gpu_work_failed')
        result={'checks_pass':False,'error_type':type(error).__name__,'error':str(error),
                'gpu_work_cancelled':False,'automatic_retry':False}
    gc.collect();torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
    result['allocated_after_release_bytes']=torch.cuda.memory_allocated();result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
    result['sources_unchanged']=all(sha(Path(p).read_bytes())==h for p,h in sources.items())
    result['checks_pass'] &= result['sources_unchanged'] and result['allocated_after_release_bytes']==result['reserved_after_release_bytes']==0
    phase('resources_released');save(a.output/'child.json',result)
    return result['checks_pass']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',choices=('128','640','1080','1440','2160','641','2342','3440'),required=True)
    p.add_argument('--iterations',type=int,choices=(1,3,12),default=3);p.add_argument('--count-operators',action='store_true');p.add_argument('--child',action='store_true')
    p.add_argument('--fusion-dll',type=Path);p.add_argument('--reference-output',type=Path)
    p.add_argument('--head-input-dll',type=Path)
    p.add_argument('--matrix-dll',type=Path)
    p.add_argument('--matrix-profile',choices=('reference','wmma_fp16','wmma_fp8'),default='reference')
    p.add_argument('--matrix-waves',type=int,choices=(1,2,4),default=1)
    p.add_argument('--matrix-modules',default='head_ffn',help='comma-separated reviewed module names; policy rejects unknown values')
    p.add_argument('--measure-head',action='store_true')
    p.add_argument('--head-gather',action='store_true')
    p.add_argument('--head-weight-cache',action='store_true')
    p.add_argument('--head-epilogue',action='store_true')
    p.add_argument('--head-qkv',action='store_true')
    p.add_argument('--pre-features-dll',type=Path)
    p.add_argument('--pre-project-pack',action='store_true')
    p.add_argument('--whole-grid-families',default='',help='comma-separated reviewed families such as pre,c32')
    p.add_argument('--c32-layout-dll',type=Path)
    p.add_argument('--resident-transitions',action='store_true')
    p.add_argument('--capture-transition-outviews',action='store_true')
    p.add_argument('--transition-dll',type=Path)
    p.add_argument('--encoder-transition',action='store_true')
    p.add_argument('--decoder-transition',action='store_true')
    p.add_argument('--measure-block',type=int,choices=(0,2,70),default=70)
    p.add_argument('--stage-timestamps',action='store_true')
    p.add_argument('--drain-stages',action='store_true')
    a=p.parse_args()
    if (a.head_epilogue or a.head_gather or a.head_qkv) and not a.head_input_dll:p.error('head kernel options require --head-input-dll')
    if a.fusion_dll and not a.reference_output:p.error('fusion requires same-input baseline reference output')
    if a.head_input_dll and not a.reference_output:p.error('head fusion requires same-input reference')
    if a.pre_features_dll and not a.reference_output:p.error('pre fusion requires same-input reference')
    if a.pre_project_pack and not a.pre_features_dll:p.error('pre project/pack requires --pre-features-dll')
    if a.c32_layout_dll and not a.reference_output:p.error('C32 fusion requires same-input reference')
    if a.capture_transition_outviews and not a.resident_transitions:p.error('transition outview capture requires resident transitions')
    if a.encoder_transition and (not a.transition_dll or not a.resident_transitions or not a.reference_output):p.error('encoder transition requires DLL, resident routing, and same-input reference')
    if a.decoder_transition and (not a.transition_dll or not a.resident_transitions or not a.reference_output):p.error('decoder transition requires DLL, resident routing, and same-input reference')
    if a.transition_dll and not (a.encoder_transition or a.decoder_transition):p.error('transition DLL requires an explicit transition selection')
    if a.matrix_profile!='reference' and (not a.matrix_dll or not a.reference_output):p.error('matrix candidate requires DLL and explicit reference')
    if a.child:raise SystemExit(0 if child(a) else 2)
    a.output.mkdir(parents=True,exist_ok=False);fixture(a.output,a.size)
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--output',str(a.output.resolve()),'--size',a.size,'--iterations',str(a.iterations)]
    if a.count_operators:command.append('--count-operators')
    if a.stage_timestamps:command.append('--stage-timestamps')
    if a.drain_stages:command.append('--drain-stages')
    if a.fusion_dll:command+=['--fusion-dll',str(a.fusion_dll.resolve())]
    if a.head_input_dll:command+=['--head-input-dll',str(a.head_input_dll.resolve())]
    if a.matrix_dll:command+=['--matrix-dll',str(a.matrix_dll.resolve())]
    command+=['--matrix-profile',a.matrix_profile,'--matrix-waves',str(a.matrix_waves),'--matrix-modules',a.matrix_modules]
    if a.measure_head:command+=['--measure-head']
    command+=['--measure-block',str(a.measure_block)]
    if a.head_gather:command+=['--head-gather']
    if a.head_weight_cache:command+=['--head-weight-cache']
    if a.head_epilogue:command+=['--head-epilogue']
    if a.head_qkv:command+=['--head-qkv']
    if a.pre_features_dll:command+=['--pre-features-dll',str(a.pre_features_dll.resolve())]
    if a.pre_project_pack:command.append('--pre-project-pack')
    if a.whole_grid_families:command+=['--whole-grid-families',a.whole_grid_families]
    if a.c32_layout_dll:command+=['--c32-layout-dll',str(a.c32_layout_dll.resolve())]
    if a.resident_transitions:command.append('--resident-transitions')
    if a.capture_transition_outviews:command.append('--capture-transition-outviews')
    if a.transition_dll:command+=['--transition-dll',str(a.transition_dll.resolve())]
    if a.encoder_transition:command.append('--encoder-transition')
    if a.decoder_transition:command.append('--decoder-transition')
    if a.reference_output:command+=['--reference-output',str(a.reference_output.resolve())]
    raise SystemExit(0 if supervise(command,a.output,timeout=180) else 2)
