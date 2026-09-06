"""Supervised isolated fusion gate, no model/default/game changes."""
import argparse,atexit,gc,hashlib,json,math,os,sys,time
from pathlib import Path
from validate_rocm_lifecycle import supervise,save,configure_fault_logging


def exercise(a,phase):
    import torch
    from native_fusion_quantize import QuantizeE4,CubicQuantizeE4
    from native_grouped_ffn import cubic_silu
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
    if 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('unreviewed GPU')
    torch.cuda.set_per_process_memory_fraction(5000000000/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2);phase('gpu_initialized')
    op=CubicQuantizeE4(a.dll) if a.operation=='cubic' else QuantizeE4(a.dll)
    def original(x):
        if a.operation=='cubic':x=cubic_silu(x)
        return x.clamp(-448,448).to(torch.float8_e4m3fn).half()
    def wait():
        event=torch.cuda.Event();event.record();deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:raise TimeoutError('GPU timeout; no retry or cancel')
            time.sleep(.001)
    host=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
    source=host.cuda();expected=original(source);wait()
    expected_cpu=expected.cpu();valid=~torch.isnan(expected_cpu)
    for i in range(a.iterations):
        actual=op(source);wait();cpu=actual.cpu()
        if not torch.equal(cpu.view(torch.int16)[valid],expected_cpu.view(torch.int16)[valid]) or not torch.isnan(cpu[~valid]).all():
            raise RuntimeError('exhaustive GPU correctness failed')
        phase(f'exhaustive{i}_passed')
    # Throughput probe distinct from exhaustive acceptance; only finite half inputs.
    source=(torch.arange(1048576,dtype=torch.int32)%31744).to(torch.int16).view(torch.float16).cuda();wait()
    timing=[];reference=None
    for kind in ('torch_three_ops','fused_one_launch'):
        for i in range(3):
            wait();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin=time.perf_counter();start.record()
            actual=original(source) if kind=='torch_three_ops' else op(source)
            end.record();wait();wall=(time.perf_counter()-begin)*1000;event_ms=start.elapsed_time(end)
            data=actual.cpu().numpy().tobytes();digest=hashlib.sha256(data).hexdigest()
            if reference is None:reference=digest
            if digest!=reference:raise RuntimeError('large tensor exact gate failed')
            timing.append({'kind':kind,'warm':i>0,'host_submit_wait_ms':wall,'gpu_event_raw_ms':event_ms,
                           'gpu_event_plausible':math.isfinite(event_ms) and 0<event_ms<=wall*1.25,
                           'authored_fused_launches':1 if kind=='fused_one_launch' else None})
    phase('gpu_work_complete')
    return {'checks_pass':True,'finite_or_inf_exact_cases':int(valid.sum()),'nan_cases':int((~valid).sum()),
            'iterations':a.iterations,'operation':a.operation,'timing':timing,'dll_sha256':hashlib.sha256(a.dll.read_bytes()).hexdigest(),
            'network_accepted':False,'game_runtime_ready':False}


def child(a):
    trace=(a.output/'traceback.log').open('x');configure_fault_logging(trace);globals()['_trace']=trace
    def phase(name):
        with (a.output/'phases.jsonl').open('a') as f:f.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
    atexit.register(phase,'python_atexit');r=exercise(a,phase)
    import torch
    gc.collect();torch.cuda.empty_cache();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
    r['allocated_after_release_bytes']=torch.cuda.memory_allocated();r['reserved_after_release_bytes']=torch.cuda.memory_reserved()
    r['checks_pass'] &= r['allocated_after_release_bytes']==r['reserved_after_release_bytes']==0
    phase('resources_released');save(a.output/'child.json',r);return r['checks_pass']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--iterations',type=int,choices=(1,12),default=1);p.add_argument('--operation',choices=('quantize','cubic'),default='quantize');p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:raise SystemExit(0 if child(a) else 2)
    a.output.mkdir(parents=True,exist_ok=False)
    raise SystemExit(0 if supervise([sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(a.dll.resolve()),'--output',str(a.output.resolve()),'--iterations',str(a.iterations),'--operation',a.operation],a.output,timeout=90) else 2)
