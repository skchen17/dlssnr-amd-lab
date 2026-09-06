"""Supervised Head projection/tail WMMA gate; never changes deployment defaults."""
import argparse,atexit,gc,hashlib,sys,time
from pathlib import Path
from validate_rocm_lifecycle import save,supervise


def metrics(actual,expected):
    import torch
    a,b=actual.detach().cpu(),expected.detach().cpu();diff=(a.float()-b.float()).double()
    neq=a.view(torch.int16)!=b.view(torch.int16);positions=neq.flatten().nonzero()
    return {'max_abs':diff.abs().max().item(),'rmse':diff.square().mean().sqrt().item(),
        'different_fraction':neq.double().mean().item(),'first_different_flat':int(positions[0]) if len(positions) else None,
        'bitwise_exact':not bool(neq.any())}


def exercise(a,phase):
    import torch
    from native_head_torch import RecoveredHead32
    from native_model_package import load_package
    from native_matrix_fusion import matrix_fusion
    from native_fusion_policy import fusion_policy
    from native_swin_torch import quantized_gather
    from audit_head_graph import kernel_nodes
    torch.set_num_threads(2);torch.manual_seed(109)
    if not torch.version.hip or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(2000000000/torch.cuda.get_device_properties(0).total_memory)
    records,_,_=load_package('local_models/native_single_color_v1','AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    head=RecoveredHead32(records[(70,0)]).eval().cuda().requires_grad_(False);block=head.block;block.inference_cache_enabled=True
    def wait():
        event=torch.cuda.Event();event.record();deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:
                save(a.output/'gpu_timeout.json',{'resources_retained':True})
                while True:time.sleep(1)
            time.sleep(.001)
    def reference(first,attention):
        seed=(first*block.attention_scale).half()
        product=quantized_gather(attention,block.permutation).float()@block._weight('project',fp32=True)
        projected=(seed.float()+product).half()
        left=(projected[...,:16].float()@head.tail[:16].float()).half()
        return projected,(left.float()+projected[...,16:].float()@head.tail[16:].float()).half()
    shape=(a.batch,64,32)
    first_guard=torch.full((a.batch*2048+256,),11.25,dtype=torch.float16,device='cuda')
    attention_guard=torch.full_like(first_guard,-9.5)
    first=first_guard[128:-128].reshape(shape);attention=attention_guard[128:-128].reshape(shape)
    first.copy_(torch.randn_like(first)*.1);attention.copy_(torch.randn_like(attention)*.1)
    first_source=first.clone();attention_source=attention.clone()
    output_guard=torch.full((a.batch*256+256,),7.75,dtype=torch.float16,device='cuda')
    out=output_guard[128:-128].reshape(a.batch,64,4)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());rows=[]
    with torch.no_grad(),torch.cuda.stream(stream),fusion_policy(a.quant_dll):
        expected_project,expected=reference(first,attention);wait()
        reference_graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(reference_graph,stream=stream):reference_project,reference_out=reference(first,attention)
        reference_nodes=kernel_nodes(reference_graph).get('0',0);reference_graph.reset();del reference_graph,reference_project,reference_out
        for profile in ('wmma_fp16','wmma_fp8'):
            for waves in (1,2,4):
                with matrix_fusion(a.dll,profile,modules=('head_output',),waves=waves) as op:
                    projected=op.head_project(block,first,attention);op.head_tail(head,projected,out);wait()
                    base={'project':metrics(projected,expected_project),'tail':metrics(out,expected)}
                    if not bool(torch.isfinite(projected).all()&torch.isfinite(out).all()):raise RuntimeError('nonfinite candidate')
                    if not torch.equal(first,first_source) or not torch.equal(attention,attention_source):raise RuntimeError('input corruption')
                    if not bool((first_guard[:128]==11.25).all()&(first_guard[-128:]==11.25).all()&
                        (attention_guard[:128]==-9.5).all()&(attention_guard[-128:]==-9.5).all()&
                        (output_guard[:128]==7.75).all()&(output_guard[-128:]==7.75).all()):raise RuntimeError('guard corruption')
                    graph=torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph,stream=stream):graph_project=op.head_project(block,first,attention);op.head_tail(head,graph_project,out)
                    nodes=kernel_nodes(graph).get('0',0);graph.instantiate();fixed=out.clone();timings={}
                    for mode in ('reference','candidate','graph'):
                        wait();start=time.perf_counter()
                        for _ in range(a.iterations):
                            if mode=='reference':value=reference(first,attention)[1]
                            elif mode=='candidate':projected=op.head_project(block,first,attention);op.head_tail(head,projected,out)
                            else:graph.replay()
                        wait();timings[mode]=(time.perf_counter()-start)*1000/a.iterations
                        if mode!='reference' and not torch.equal(out.view(torch.int16),fixed.view(torch.int16)):raise RuntimeError('repeat mismatch')
                    first.add_(.03125);attention.sub_(.015625)
                    changed_project=op.head_project(block,first,attention);op.head_tail(head,changed_project,out);wait();changed=out.clone()
                    graph.replay();wait()
                    if not torch.equal(changed.view(torch.int16),out.view(torch.int16)):raise RuntimeError('graph stale-input output')
                    changed_ref_project,changed_ref=reference(first,attention);changed_metrics={'project':metrics(changed_project,changed_ref_project),'tail':metrics(out,changed_ref)}
                    first.copy_(first_source);attention.copy_(attention_source)
                    old_project=block.project.clone();old_tail=head.tail.clone();block.project.add_(.001);head.tail.sub_(.001)
                    mutation_project=op.head_project(block,first,attention);op.head_tail(head,mutation_project,out);wait()
                    mut_ref_project,mut_ref=reference(first,attention);mutation={'project':metrics(mutation_project,mut_ref_project),'tail':metrics(out,mut_ref)}
                    block.project.copy_(old_project);head.tail.copy_(old_tail);wait()
                    rows.append({'profile':profile,'waves':waves,'errors':base,'changed_errors':changed_metrics,
                        'weight_mutation_errors':mutation,'candidate_kernel_nodes':nodes,'reference_kernel_nodes':reference_nodes,
                        'host_ms_per_call':timings,'guard_pass':True,'repeat_exact':True})
                    graph.reset();del graph,graph_project,fixed,changed_project,changed,changed_ref_project,changed_ref,old_project,old_tail,mutation_project,mut_ref_project,mut_ref
                    phase(f'{profile}_{waves}_passed')
    del head,block,first_guard,attention_guard,first,attention,first_source,attention_source,output_guard,out,stream,expected_project,expected,value
    return {'checks_pass':True,'rows':rows,'iterations':a.iterations,'reference_kernel_nodes':reference_nodes,
        'scope':'actual frozen Head weights; isolated attention output projection and tail; QK/softmax/PV excluded; host submit/wait','default_promoted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True);p.add_argument('--quant-dll',type=Path,required=True)
    p.add_argument('--batch',type=int,choices=(1,753,768),default=1);p.add_argument('--iterations',type=int,choices=(1,12),default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:
        def phase(name):
            with (a.output/'phases.jsonl').open('a') as f:import json;f.write(json.dumps({'phase':name})+'\n')
        atexit.register(phase,'python_atexit');paths=list(Path(__file__).parent.glob('native_*.py'))+[Path(__file__),a.dll,a.quant_dll]
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};result=exercise(a,phase);phase('gpu_work_complete')
        import torch
        gc.collect();torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
        result['allocated_after_release_bytes']=torch.cuda.memory_allocated();result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
        result['sources']=sources;result['sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
        result['checks_pass']&=result['sources_unchanged'] and result['allocated_after_release_bytes']==result['reserved_after_release_bytes']==0
        phase('resources_released');save(a.output/'child.json',result)
    else:
        a.output.mkdir(parents=True,exist_ok=False);command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(a.dll.resolve()),
            '--quant-dll',str(a.quant_dll.resolve()),'--batch',str(a.batch),'--iterations',str(a.iterations),'--output',str(a.output.resolve())]
        raise SystemExit(0 if supervise(command,a.output,timeout=120) else 2)
