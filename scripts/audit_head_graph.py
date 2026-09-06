"""Bounded HIP graph audit of Head/project or packed Swin cores (not full stages)."""
import argparse,atexit,ctypes as ct,gc,hashlib,json,os,sys,time
from pathlib import Path
from validate_rocm_lifecycle import save,supervise

def kernel_nodes(graph):
    # Match the ROCm 7 runtime linked by the project, not a system HIP 6 DLL.
    runtime=Path(sys.prefix)/'Lib/site-packages/_rocm_sdk_core/bin/amdhip64_7.dll'
    hip=ct.CDLL(str(runtime.resolve()))
    get=hip.hipGraphGetNodes;get.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t)]
    kind=hip.hipGraphNodeGetType;kind.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
    child=hip.hipGraphChildGraphNodeGetGraph;child.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p)]
    def check(code):
        if code:raise RuntimeError(f'HIP graph query error {code}')
    def walk(ptr):
        count=ct.c_size_t();check(get(ptr,None,ct.byref(count)))
        nodes=(ct.c_void_p*count.value)();check(get(ptr,nodes,ct.byref(count)))
        counts={}
        for node in nodes:
            t=ct.c_int();check(kind(node,ct.byref(t)));key=str(t.value);counts[key]=counts.get(key,0)+1
            if t.value==4:
                sub=ct.c_void_p();check(child(node,ct.byref(sub)))
                for k,n in walk(sub).items():counts[k]=counts.get(k,0)+n
        return counts
    return walk(graph.raw_cuda_graph())

def exercise(a):
    import torch
    from native_head_torch import RecoveredHead32
    from native_packed_swin import RecoveredPackedSwin
    from native_preblock import RecoveredSingleColorPreblock
    from native_split_swin512 import RecoveredSplitSwin512,RECORD_SIZES as SPLIT_PARTS
    from native_packed_swin import logical_indices
    from native_vit1024 import RecoveredVit1024,RECORD_SIZES as VIT_PARTS
    from native_model_package import load_package
    from native_fusion_policy import fusion_policy
    from native_execution_policy import execution_policy
    from native_matrix_fusion import matrix_fusion
    from native_head_fusion import head_input_fusion
    torch.set_num_threads(2);torch.manual_seed(83)
    if not torch.version.hip or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(2000000000/torch.cuda.get_device_properties(0).total_memory)
    records,_,_=load_package('local_models/native_single_color_v1','AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    channels={'head':32,'pre':32,'c32':32,'c64':64,'c128':128,'c256':256,'c512':512,'vit':1024}[a.family]
    if a.family=='head':head=RecoveredHead32(records[(70,0)])
    elif a.family=='pre':head=RecoveredSingleColorPreblock(records[(0,0)]).swin
    elif a.family=='c512':
        head=RecoveredSplitSwin512(**{k:records[(24,i)] for i,k in enumerate(SPLIT_PARTS)})
        ai,ri=logical_indices(512)
        head.register_buffer('audit_a',ai);head.register_buffer('audit_r',ri)
        del ai,ri
    elif a.family=='vit':head=RecoveredVit1024(**{k:records[(31,i)] for i,k in enumerate(VIT_PARTS)})
    else:
        block={'c32':2,'c64':6,'c128':10,'c256':16}[a.family]
        head=RecoveredPackedSwin(records[(block,0)],record_kind=f'swin{channels}')
    head=head.eval().cuda().requires_grad_(False)
    project=head.project if a.family=='head' else head
    if a.family=='c512':project=lambda x:head.unquantized(x[:,head.audit_a],x[:,head.audit_r])
    if a.family=='vit':project=lambda x:head(x,query_chunk=1024)
    shape=(1,a.batch,1024) if a.family=='vit' else (a.batch,64*channels)
    x=(torch.randn(*shape,dtype=torch.float16)*.1).cuda()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    def wait():
        event=torch.cuda.Event();event.record();deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:
                save(a.output/'gpu_timeout.json',{'pid':os.getpid(),'resources_retained':True})
                while True:time.sleep(1)
            time.sleep(.001)
    reference=None;rows=[]
    with torch.no_grad(),torch.cuda.stream(stream),execution_policy('native_fp16'),fusion_policy(a.quant_dll):
        for optimized in ((False,True) if a.family=='head' or a.matrix_profile!='reference' else (False,)):
            if a.family=='head':head.block.inference_cache_enabled=optimized
            with head_input_fusion(a.dll,epilogue=optimized,qkv=optimized and a.qkv),matrix_fusion(a.matrix_dll,a.matrix_profile if optimized else 'reference',modules=tuple(a.matrix_modules.split(',')),waves=a.matrix_waves):
                for _ in range(2):expected=project(x)
                wait()
                if not bool(torch.isfinite(expected).all()):raise RuntimeError('nonfinite head fixture')
                raw=expected.cpu().numpy().tobytes();sha=hashlib.sha256(raw).hexdigest()
                if reference is None:reference=sha
                if sha!=reference:raise RuntimeError('optimized head differs')
                eager=[]
                for _ in range(a.iterations):
                    wait();start=time.perf_counter();expected=project(x);wait();eager.append((time.perf_counter()-start)*1000)
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):actual=project(x)
                nodes=kernel_nodes(graph);graph.instantiate()
                replay=[]
                for _ in range(a.iterations):
                    wait();start=time.perf_counter();graph.replay();wait();replay.append((time.perf_counter()-start)*1000)
                    if hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()!=reference:raise RuntimeError('graph output differs')
                # Amortize the Windows fence polling/flush cost: no per-call wait.
                bulk={}
                for mode in ('eager','graph'):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();wait();start=time.perf_counter()
                    for _ in range(a.iterations):
                        if mode=='graph':graph.replay()
                        else:expected=project(x)
                    end.record();wait();wall=(time.perf_counter()-start)*1000;elapsed=begin.elapsed_time(end)
                    bulk[mode]={'host_total_ms':wall,'calls':a.iterations,'host_per_call_ms':wall/a.iterations,
                        'gpu_stream_total_ms':elapsed,'scope':'begin event drained before submission; includes idle, not kernel busy sum'}
                original=x.clone();x.add_(.03125);changed=project(x);graph.replay();wait()
                if not torch.equal(changed.view(torch.int16),actual.view(torch.int16)):raise RuntimeError('graph failed changed-input check')
                x.copy_(original);wait();del original,changed,begin,end
                rows.append({'optimized':optimized,'batch':a.batch,'captured_node_types':nodes,'kernel_nodes':nodes.get('0',0),
                    'host_eager_ms':eager,'host_graph_replay_ms':replay,'output_sha256':reference,
                    'bulk':bulk,'changed_input_exact':True,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                graph.reset();del graph,actual,expected
    del project,head,x,stream
    return {'checks_pass':True,'family':a.family,'rows':rows,'scope':'Synthetic packed-window core only; excludes image gather/scatter, Pre input construction and full-frame scheduling. Graph replay is NOT kernel fusion or GPU busy time.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True);p.add_argument('--quant-dll',type=Path,required=True)
    p.add_argument('--batch',type=int,choices=(1,144,753,768,2160),default=1);p.add_argument('--iterations',type=int,choices=(1,12),default=1)
    p.add_argument('--qkv',action='store_true')
    p.add_argument('--matrix-dll',type=Path)
    p.add_argument('--matrix-profile',choices=('reference','wmma_fp16','wmma_fp8'),default='reference')
    p.add_argument('--matrix-waves',type=int,choices=(1,2,4),default=2)
    p.add_argument('--matrix-modules',default='head_ffn')
    p.add_argument('--family',choices=('head','pre','c32','c64','c128','c256','c512','vit'),default='head')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.batch==2160 and a.family!='vit':p.error('2160 is a ViT token count only')
    if a.matrix_profile!='reference' and (a.family not in ('head','c32','c64','c128','c256','c512') or a.matrix_dll is None):p.error('matrix audit requires a reviewed family and explicit DLL')
    if a.child:
        def phase(name):
            with (a.output/'phases.jsonl').open('a') as f:f.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
        paths=list(Path(__file__).parent.glob('native_*.py'))+[Path(__file__),a.dll,a.quant_dll]+([a.matrix_dll] if a.matrix_dll else [])
        fingerprints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        atexit.register(phase,'python_atexit');r=exercise(a);phase('gpu_work_complete')
        r['sources']=fingerprints
        r['sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in fingerprints.items())
        r['checks_pass'] &= r['sources_unchanged']
        import torch
        gc.collect();torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
        r['allocated_after_release_bytes']=torch.cuda.memory_allocated();r['reserved_after_release_bytes']=torch.cuda.memory_reserved()
        r['checks_pass'] &= r['allocated_after_release_bytes']==r['reserved_after_release_bytes']==0
        phase('resources_released');save(a.output/'child.json',r)
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(a.dll.resolve()),'--quant-dll',str(a.quant_dll.resolve()),
                 '--batch',str(a.batch),'--iterations',str(a.iterations),'--family',a.family,'--output',str(a.output.resolve())]
        if a.qkv:command.append('--qkv')
        if a.matrix_dll:command+=['--matrix-dll',str(a.matrix_dll.resolve())]
        command+=['--matrix-profile',a.matrix_profile,'--matrix-waves',str(a.matrix_waves),'--matrix-modules',a.matrix_modules]
        raise SystemExit(0 if supervise(command,a.output,timeout=90) else 2)
