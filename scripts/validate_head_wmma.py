"""Supervised Head FFN WMMA gate/sweep; approximate results never promoted."""
import argparse,atexit,gc,hashlib,json,os,sys,time,statistics
from pathlib import Path
from validate_rocm_lifecycle import save,supervise


def errors(actual,expected):
    import torch
    a,b=actual.detach().cpu(),expected.detach().cpu()
    diff=(a.float()-b.float()).double()
    neq=a.view(torch.int16)!=b.view(torch.int16)
    positions=neq.flatten().nonzero()
    return {'max_abs':diff.abs().max().item(),'rmse':diff.square().mean().sqrt().item(),
        'different_fraction':neq.double().mean().item(),'first_different_flat':int(positions[0]) if len(positions) else None,
        'bitwise_exact':not bool(neq.any())}


def exercise(a,phase):
    import torch
    from native_head_torch import RecoveredHead32
    from native_model_package import load_package
    from native_matrix_fusion import matrix_fusion
    from native_fusion_policy import fusion_policy
    from audit_head_graph import kernel_nodes
    torch.set_num_threads(2);torch.manual_seed(93)
    if not torch.version.hip or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(2000000000/torch.cuda.get_device_properties(0).total_memory)
    records,_,_=load_package('local_models/native_single_color_v1','AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    model=RecoveredHead32(records[(70,0)]).block.eval().cuda().requires_grad_(False)
    model.inference_cache_enabled=True
    def wait():
        event=torch.cuda.Event();event.record();deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:
                save(a.output/'gpu_timeout.json',{'pid':os.getpid(),'resources_retained':True})
                while True:time.sleep(1)
            time.sleep(.001)
    # Guard both source and caller-owned destination. Distinct windows/rows/channels
    # exercise WMMA direction and packed/permuted index maps using actual weights.
    guard=torch.full((a.batch*2048+256,),17.25,device='cuda',dtype=torch.float16)
    x=guard[128:-128].reshape(a.batch,2048)
    x.copy_((torch.randn_like(x)*.1))
    source=x.clone()
    output=torch.full_like(guard,-13.25);out=output[128:-128].reshape(a.batch,64,32)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    rows=[]
    with torch.no_grad(),torch.cuda.stream(stream),fusion_policy(a.quant_dll):
        expected=model.forward_ffn(x);wait()
        reference_graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(reference_graph,stream=stream):reference_value=model.forward_ffn(x)
        reference_nodes=kernel_nodes(reference_graph);reference_graph.reset()
        del reference_graph,reference_value
        # Empty graph baseline uses the same current reference, not an RTX teacher.
        for profile in ('wmma_fp16','wmma_fp8'):
            for waves in (1,2,4):
                with matrix_fusion(a.dll,profile,waves=waves) as op:
                    op.prepare(model);op.head_ffn(model,x,out);wait()
                    if not bool(torch.isfinite(out).all()):raise RuntimeError('nonfinite candidate')
                    metric=errors(out,expected)
                    if not torch.equal(x,source) or not bool((guard[:128]==17.25).all() & (guard[-128:]==17.25).all() & (output[:128]==-13.25).all() & (output[-128:]==-13.25).all()):raise RuntimeError('guard/input corruption')
                    fixed=out.clone()
                    graph=torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph,stream=stream):op.head_ffn(model,x,out)
                    nodes=kernel_nodes(graph);graph.instantiate()
                    timings={}
                    for mode in ('reference','candidate','graph'):
                        wait();start=time.perf_counter()
                        for _ in range(a.iterations):
                            if mode=='reference':
                                with matrix_fusion():value=model.forward_ffn(x)
                            elif mode=='candidate':op.head_ffn(model,x,out)
                            else:graph.replay()
                        wait();timings[mode]=(time.perf_counter()-start)*1000/a.iterations
                        if mode!='reference' and not torch.equal(out.view(torch.int16),fixed.view(torch.int16)):raise RuntimeError('candidate repeat mismatch')
                    x.add_(.03125)
                    op.head_ffn(model,x,out);wait();changed=out.clone();graph.replay();wait()
                    if not torch.equal(changed.view(torch.int16),out.view(torch.int16)):raise RuntimeError('graph changed-input mismatch')
                    with matrix_fusion():changed_reference=model.forward_ffn(x)
                    wait();changed_metric=errors(out,changed_reference)
                    edge_metrics=[]
                    # Finite operating-domain edges, including signed zero and
                    # E4 rounding neighborhoods. Nonfinite inputs are not a
                    # full-network acceptance case and are not hidden as passes.
                    for kind in ('zero','negative_zero','alternating','midpoints'):
                        pattern=torch.arange(x.numel(),device=x.device).reshape(x.shape)
                        if kind=='zero':x.zero_()
                        elif kind=='negative_zero':x.fill_(-0.0)
                        elif kind=='alternating':x.copy_(torch.where(pattern%2==0,4.,-4.).half())
                        else:x.copy_(((pattern%129-64).float()/32+1/64).half())
                        with matrix_fusion():edge_ref=model.forward_ffn(x)
                        op.head_ffn(model,x,out);wait()
                        if not bool(torch.isfinite(out).all() & torch.isfinite(edge_ref).all()):raise RuntimeError('nonfinite edge fixture')
                        edge_metrics.append({'fixture':kind,**errors(out,edge_ref)})
                    # Version-aware prepack must consume modified weights.
                    previous=model.expand.clone();model.expand.add_(.001)
                    with matrix_fusion():mut_ref=model.forward_ffn(x)
                    op.head_ffn(model,x,out);wait();mutation_metric=errors(out,mut_ref)
                    model.expand.copy_(previous)
                    x.copy_(source);wait()
                    rows.append({'profile':profile,'waves':waves,'batch':a.batch,'errors':metric,'changed_errors':changed_metric,
                        'edge_errors':edge_metrics,'weight_mutation_errors':mutation_metric,
                        'host_ms_per_call':timings,'captured_kernel_nodes':nodes.get('0',0),'guard_pass':True,'repeat_exact':True,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                    graph.reset();del graph,op,fixed,changed,changed_reference,pattern,edge_ref,previous,mut_ref
                    phase(f'{profile}_{waves}_passed')
    del model,guard,x,source,output,out,stream,expected,value
    return {'checks_pass':True,'rows':rows,'iterations':a.iterations,'reference_kernel_nodes':reference_nodes.get('0',0),
        'scope':'synthetic packed Head FFN reference with weight cache/quant/cubic but without separate residual epilogue fusion; host submit/wait, not whole frame or pure GPU busy time','default_promoted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dll',type=Path,required=True);p.add_argument('--quant-dll',type=Path,required=True)
    p.add_argument('--batch',type=int,choices=(1,753,768),default=1)
    p.add_argument('--iterations',type=int,choices=(1,12),default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:
        def phase(name):
            with (a.output/'phases.jsonl').open('a') as f:f.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
        atexit.register(phase,'python_atexit')
        paths=list(Path(__file__).parent.glob('native_*.py'))+[Path(__file__),a.dll,a.quant_dll]
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        result=exercise(a,phase);phase('gpu_work_complete')
        import torch
        gc.collect();torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
        result['allocated_after_release_bytes']=torch.cuda.memory_allocated();result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
        result['sources']=sources;result['sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
        result['checks_pass'] &= result['sources_unchanged'] and result['allocated_after_release_bytes']==result['reserved_after_release_bytes']==0
        phase('resources_released');save(a.output/'child.json',result)
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(a.dll.resolve()),'--quant-dll',str(a.quant_dll.resolve()),'--batch',str(a.batch),'--iterations',str(a.iterations),'--output',str(a.output.resolve())]
        raise SystemExit(0 if supervise(command,a.output,timeout=120) else 2)
