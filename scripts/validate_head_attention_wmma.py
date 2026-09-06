"""Supervised QK/PV WMMA gate with the recovered Head's real position bias."""
import argparse,atexit,gc,hashlib,sys,time
from pathlib import Path
from validate_rocm_lifecycle import save,supervise


def metric(a,b):
    import torch
    x,y=a.detach().cpu(),b.detach().cpu();d=(x.float()-y.float()).double();neq=x.view(torch.int16)!=y.view(torch.int16);p=neq.flatten().nonzero()
    return {'max_abs':d.abs().max().item(),'rmse':d.square().mean().sqrt().item(),'different_fraction':neq.double().mean().item(),
        'first_different_flat':int(p[0]) if len(p) else None,'bitwise_exact':not bool(neq.any())}


def exercise(a,phase):
    import torch
    from native_head_torch import RecoveredHead32
    from native_model_package import load_package
    from native_matrix_fusion import matrix_fusion
    from native_swin_torch import quantize_e4
    from native_fusion_policy import fusion_policy
    from audit_head_graph import kernel_nodes
    torch.set_num_threads(2);torch.manual_seed(127)
    if not torch.version.hip or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(2500000000/torch.cuda.get_device_properties(0).total_memory)
    records,_,_=load_package('local_models/native_single_color_v1','AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    block=RecoveredHead32(records[(70,0)]).block.eval().cuda().requires_grad_(False)
    def wait():
        e=torch.cuda.Event();e.record();deadline=time.monotonic()+10
        while not e.query():
            if time.monotonic()>deadline:
                save(a.output/'gpu_timeout.json',{'resources_retained':True})
                while True:time.sleep(1)
            time.sleep(.001)
    def reference(q,k,p,v):
        s=(q.float()@k.float().transpose(-1,-2)+block.position_bias.float()).half()
        first=(p[...,:32].float()@v[...,:32,:].float()).half()
        return s,(first.float()+p[...,32:].float()@v[...,32:,:].float()).half()
    with torch.no_grad(),fusion_policy(a.quant_dll):
        q=quantize_e4((torch.randn(a.batch,64,32,device='cuda',dtype=torch.float16)*.1));k=quantize_e4(torch.randn_like(q)*.1);v=quantize_e4(torch.randn_like(q)*.1)
        probability=quantize_e4(torch.softmax(torch.randn(a.batch,64,64,device='cuda').half().float(),dim=-1).half())
        q0,k0,v0,p0=(x.clone() for x in (q,k,v,probability));expected_scores,expected=reference(q,k,probability,v)
        expected_probability=quantize_e4(torch.softmax(expected_scores.float(),dim=-1).half());wait()
        rows=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            reference_graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(reference_graph,stream=stream):rs,rv=reference(q,k,probability,v)
            refnodes=kernel_nodes(reference_graph).get('0',0);reference_graph.reset();del reference_graph,rs,rv
            for profile in ('wmma_fp16','wmma_fp8'):
                for waves in (1,2,4):
                    with matrix_fusion(a.dll,profile,modules=('head_attention','head_softmax'),waves=waves) as op:
                        scores=op.head_scores(block,q,k);candidate_probability=op.head_probability(scores);out=op.head_pv(probability,v);wait();base={'qk':metric(scores,expected_scores),'softmax':metric(candidate_probability,expected_probability),'pv':metric(out,expected)}
                        if not bool(torch.isfinite(scores).all()&torch.isfinite(out).all()):raise RuntimeError('nonfinite attention')
                        graph=torch.cuda.CUDAGraph(keep_graph=True)
                        with torch.cuda.graph(graph,stream=stream):gs=op.head_scores(block,q,k);go=op.head_pv(probability,v)
                        nodes=kernel_nodes(graph).get('0',0);graph.instantiate();fixed_scores=scores.clone();fixed=out.clone();timings={}
                        for mode in ('reference','candidate','graph'):
                            wait();start=time.perf_counter()
                            for _ in range(a.iterations):
                                if mode=='reference':value=reference(q,k,probability,v)
                                elif mode=='candidate':scores=op.head_scores(block,q,k);out=op.head_pv(probability,v)
                                else:graph.replay()
                            wait();timings[mode]=(time.perf_counter()-start)*1000/a.iterations
                            if mode!='reference' and (not torch.equal(scores.view(torch.int16),fixed_scores.view(torch.int16)) or not torch.equal(out.view(torch.int16),fixed.view(torch.int16))):raise RuntimeError('repeat mismatch')
                        q.copy_(quantize_e4((q+.03125).half()));probability.copy_(quantize_e4((probability+.015625).half()));scores=op.head_scores(block,q,k);out=op.head_pv(probability,v);wait();changed_scores=scores.clone();changed=out.clone();graph.replay();wait()
                        if not torch.equal(changed_scores.view(torch.int16),gs.view(torch.int16)) or not torch.equal(changed.view(torch.int16),go.view(torch.int16)):raise RuntimeError('graph stale input')
                        crs,crv=reference(q,k,probability,v);changed_probability=op.head_probability(crs);crp=quantize_e4(torch.softmax(crs.float(),dim=-1).half());changed_metric={'qk':metric(scores,crs),'softmax':metric(changed_probability,crp),'pv':metric(out,crv)}
                        old=block.position_bias.clone();block.position_bias.add_(.001);scores=op.head_scores(block,q,k);wait();mrs,_=reference(q,k,probability,v);mutation=metric(scores,mrs);block.position_bias.copy_(old)
                        q.copy_(q0);k.copy_(k0);v.copy_(v0);probability.copy_(p0);wait()
                        rows.append({'profile':profile,'waves':waves,'errors':base,'changed_errors':changed_metric,'bias_mutation_errors':mutation,
                            'reference_kernel_nodes':refnodes,'candidate_kernel_nodes':nodes,'host_ms_per_call':timings,'repeat_exact':True})
                        graph.reset();del graph,gs,go,fixed_scores,fixed,changed_scores,changed,crs,crv,changed_probability,crp,old,mrs,candidate_probability
                        phase(f'{profile}_{waves}_passed')
    del block,q,k,v,probability,q0,k0,v0,p0,expected_scores,expected_probability,expected,stream,value
    return {'checks_pass':True,'rows':rows,'reference_kernel_nodes':refnodes,'scope':'isolated Head QK+position bias and split-PV; softmax/QKV projection excluded; host submit/wait','default_promoted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True);p.add_argument('--quant-dll',type=Path,required=True);p.add_argument('--batch',type=int,choices=(1,753,768),default=1);p.add_argument('--iterations',type=int,choices=(1,12),default=1);p.add_argument('--output',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:
        def phase(name):
            with (a.output/'phases.jsonl').open('a') as f:import json;f.write(json.dumps({'phase':name})+'\n')
        atexit.register(phase,'python_atexit');paths=list(Path(__file__).parent.glob('native_*.py'))+[Path(__file__),a.dll,a.quant_dll];sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};result=exercise(a,phase);phase('gpu_work_complete')
        import torch
        gc.collect();torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();result['allocated_after_release_bytes']=torch.cuda.memory_allocated();result['reserved_after_release_bytes']=torch.cuda.memory_reserved();result['sources']=sources;result['sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items());result['checks_pass']&=result['sources_unchanged'] and result['allocated_after_release_bytes']==result['reserved_after_release_bytes']==0;phase('resources_released');save(a.output/'child.json',result)
    else:
        a.output.mkdir(parents=True,exist_ok=False);command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(a.dll.resolve()),'--quant-dll',str(a.quant_dll.resolve()),'--batch',str(a.batch),'--iterations',str(a.iterations),'--output',str(a.output.resolve())];raise SystemExit(0 if supervise(command,a.output,timeout=120) else 2)
