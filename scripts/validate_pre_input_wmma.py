"""Supervised original-weight Pre 16x32 projection WMMA gate."""
import argparse,atexit,gc,hashlib,sys,time
from pathlib import Path
from validate_rocm_lifecycle import save,supervise


def metric(a,b):
    import torch
    x,y=a.detach().cpu(),b.detach().cpu();d=(x.float()-y.float()).double();neq=x.view(torch.int16)!=y.view(torch.int16);p=neq.flatten().nonzero()
    first=int(p[0]) if len(p) else None
    return {'max_abs':d.abs().max().item(),'rmse':d.square().mean().sqrt().item(),'different_fraction':neq.double().mean().item(),'first_different_flat':first,
        'first_actual_bits':int(x.view(torch.int16).flatten()[first].item()) if first is not None else None,
        'first_expected_bits':int(y.view(torch.int16).flatten()[first].item()) if first is not None else None,'bitwise_exact':first is None}


def exercise(a,phase):
    import torch
    from native_preblock import RecoveredSingleColorPreblock,project_input_features
    from native_model_package import load_package
    from native_matrix_fusion import matrix_fusion
    from audit_head_graph import kernel_nodes
    torch.set_num_threads(2);torch.manual_seed(149)
    if not torch.version.hip or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(2500000000/torch.cuda.get_device_properties(0).total_memory)
    records,_,_=load_package('local_models/native_single_color_v1','AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    model=RecoveredSingleColorPreblock(records[(0,0)]).eval().cuda().requires_grad_(False);weight=model.input_project
    guard=torch.full((a.rows*16+256,),5.5,dtype=torch.float16,device='cuda');features=guard[128:-128].reshape(1,a.rows,16);features.copy_(torch.randn_like(features)*.1);source=features.clone()
    outguard=torch.full((a.rows*32+256,),-6.5,dtype=torch.float16,device='cuda');out=outguard[128:-128].reshape(1,a.rows,32)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    def wait():
        e=torch.cuda.Event();e.record();deadline=time.monotonic()+10
        while not e.query():
            if time.monotonic()>deadline:
                save(a.output/'gpu_timeout.json',{'resources_retained':True})
                while True:time.sleep(1)
            time.sleep(.001)
    rows=[]
    with torch.no_grad(),torch.cuda.stream(stream):
        expected=project_input_features(features,weight,rows_per_chunk=65536);wait()
        refgraph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(refgraph,stream=stream):refvalue=project_input_features(features,weight,rows_per_chunk=65536)
        refnodes=kernel_nodes(refgraph).get('0',0);refgraph.reset();del refgraph,refvalue
        for waves in (1,2,4):
            with matrix_fusion(a.dll,'wmma_fp16',modules=('pre_input',),waves=waves) as op:
                op.pre_project(features,weight,out);wait();base=metric(out,expected)
                initial=out.clone();op.pre_project(features,weight,out);wait()
                if not torch.equal(out.view(torch.int16),initial.view(torch.int16)):
                    different=int((out.view(torch.int16)!=initial.view(torch.int16)).sum().item())
                    raise RuntimeError(f'pre-graph repeat mismatch waves={waves} elements={different}')
                if not torch.equal(features,source) or not bool((guard[:128]==5.5).all()&(guard[-128:]==5.5).all()&(outguard[:128]==-6.5).all()&(outguard[-128:]==-6.5).all()):raise RuntimeError('guard/input corruption')
                graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph,stream=stream):op.pre_project(features,weight,out)
                nodes=kernel_nodes(graph).get('0',0);graph.instantiate();fixed=out.clone();timings={}
                for mode in ('reference','candidate','graph'):
                    wait();start=time.perf_counter()
                    for _ in range(a.iterations):
                        if mode=='reference':value=project_input_features(features,weight,rows_per_chunk=65536)
                        elif mode=='candidate':op.pre_project(features,weight,out)
                        else:graph.replay()
                    wait();timings[mode]=(time.perf_counter()-start)*1000/a.iterations
                    if mode!='reference' and not torch.equal(out.view(torch.int16),fixed.view(torch.int16)):
                        different=int((out.view(torch.int16)!=fixed.view(torch.int16)).sum().item())
                        raise RuntimeError(f'repeat mismatch waves={waves} mode={mode} elements={different}')
                features.add_(.03125);op.pre_project(features,weight,out);wait();changed=out.clone();graph.replay();wait()
                if not torch.equal(changed.view(torch.int16),out.view(torch.int16)):raise RuntimeError('graph stale input')
                changed_ref=project_input_features(features,weight,rows_per_chunk=65536);changed_metric=metric(out,changed_ref)
                features.copy_(source);old=weight.clone();weight.add_(.001);op.pre_project(features,weight,out);wait();mutref=project_input_features(features,weight,rows_per_chunk=65536);mutation=metric(out,mutref);weight.copy_(old);wait()
                rows.append({'waves':waves,'errors':base,'changed_errors':changed_metric,'weight_mutation_errors':mutation,'reference_kernel_nodes':refnodes,'candidate_kernel_nodes':nodes,'host_ms_per_call':timings,'guard_pass':True,'repeat_exact':True})
                graph.reset();del graph,initial,fixed,changed,changed_ref,old,mutref;phase(f'waves_{waves}_passed')
    del model,weight,guard,features,source,outguard,out,stream,expected,value
    return {'checks_pass':True,'rows':rows,'input_rows':a.rows,'scope':'actual frozen Pre input weight, isolated 16x32 projection; host submit/wait','default_promoted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True);p.add_argument('--rows',type=int,choices=(16,65536),default=16);p.add_argument('--iterations',type=int,choices=(1,12),default=1);p.add_argument('--output',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:
        def phase(name):
            with (a.output/'phases.jsonl').open('a') as f:import json;f.write(json.dumps({'phase':name})+'\n')
        atexit.register(phase,'python_atexit');paths=list(Path(__file__).parent.glob('native_*.py'))+[Path(__file__),a.dll];sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};result=exercise(a,phase);phase('gpu_work_complete')
        import torch
        gc.collect();torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();result['allocated_after_release_bytes']=torch.cuda.memory_allocated();result['reserved_after_release_bytes']=torch.cuda.memory_reserved();result['sources']=sources;result['sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items());result['checks_pass']&=result['sources_unchanged'] and result['allocated_after_release_bytes']==result['reserved_after_release_bytes']==0;phase('resources_released');save(a.output/'child.json',result)
    else:
        a.output.mkdir(parents=True,exist_ok=False);command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(a.dll.resolve()),'--rows',str(a.rows),'--iterations',str(a.iterations),'--output',str(a.output.resolve())];raise SystemExit(0 if supervise(command,a.output,timeout=120) else 2)
