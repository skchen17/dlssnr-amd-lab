"""Bounded non-RGP gate for a two-kernel E4M3 resident activation chain."""
import argparse,ctypes as ct,hashlib,json,math,statistics,time
from pathlib import Path
from gpu_safety import require_gpu_tests_enabled


def run(a):
    import torch
    require_gpu_tests_enabled('resident FP8 activation chain')
    if not torch.version.hip or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError('reviewed gfx1201 ROCm device required')
    lib=ct.CDLL(str(a.dll.resolve()))
    for name in ('nr_fp8_prepack_weights','nr_fp8_chain_fp16_boundary','nr_fp8_chain_resident'):
        fn=getattr(lib,name);fn.restype=ct.c_int
    lib.nr_fp8_prepack_weights.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p]
    for name in ('nr_fp8_chain_fp16_boundary','nr_fp8_chain_resident'):
        getattr(lib,name).argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p]
    stream=torch.cuda.current_stream();sp=ct.c_void_p(stream.cuda_stream);n=a.matrices*256
    x=(torch.arange(n,device='cuda',dtype=torch.float32).remainder(257).sub(128).div(31)).half()
    w=(torch.arange(n,device='cuda',dtype=torch.float32).mul(17).remainder(251).sub(125).div(29)).half()
    packed_w=torch.empty(n,device='cuda',dtype=torch.uint8)
    half_boundary=torch.empty(n,device='cuda',dtype=torch.float16);byte_boundary=torch.empty(n,device='cuda',dtype=torch.uint8)
    ref=torch.empty(n,device='cuda',dtype=torch.float16);got=torch.empty_like(ref)
    code=lib.nr_fp8_prepack_weights(w.data_ptr(),packed_w.data_ptr(),n,sp)
    if code:raise RuntimeError(f'weight prepack launch failed: HIP error {code}')
    def wait(event):
        deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:raise TimeoutError('GPU uncertain; no cancel/retry')
            time.sleep(.001)
    done=torch.cuda.Event();done.record();wait(done)
    calls=[('fp16_boundary',lib.nr_fp8_chain_fp16_boundary,half_boundary,ref),
           ('resident_fp8',lib.nr_fp8_chain_resident,byte_boundary,got)]
    times={name:[] for name,_,_,_ in calls}
    order=calls if a.iterations==1 else calls+list(reversed(calls))
    for _ in range(a.iterations):
        for name,fn,boundary,out in order:
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            code=fn(x.data_ptr(),packed_w.data_ptr(),boundary.data_ptr(),out.data_ptr(),a.matrices,sp)
            if code:raise RuntimeError(f'{name} launch failed: {code}')
            end.record();wait(end);times[name].append(begin.elapsed_time(end))
    exact=torch.equal(ref,got);finite=bool(torch.isfinite(got).all())
    max_error=float((ref.float()-got.float()).abs().max())
    output_sha=hashlib.sha256(got.cpu().numpy().tobytes()).hexdigest().upper()
    timing_valid=all(math.isfinite(v) and v>0 for values in times.values() for v in values)
    result={'schema':1,'checks_pass':exact and finite,'matrices':a.matrices,'iterations':a.iterations,
            'boundary':'confirmed E4M3 only; residual/FP16 boundaries are unchanged','wmma':'RDNA4 FP8 builtin in both consumers',
            'fp16_boundary_bytes':half_boundary.numel()*half_boundary.element_size(),
            'resident_fp8_boundary_bytes':byte_boundary.numel()*byte_boundary.element_size(),
            'activation_byte_reduction_percent':50.0,'dispatches_per_chain':2,'conversion_dispatch_reduction':0,
            'consumer_repack_removed':True,'bitwise_exact':exact,'max_abs_error':max_error,'finite':finite,
            'output_sha256':output_sha,'gpu_event_timing_valid':timing_valid,
            'gpu_event_ms':({k:{'samples':v,'median':statistics.median(v),
                'p95':sorted(v)[max(0,int(len(v)*.95+.999)-1)]} for k,v in times.items()} if timing_valid else None),
            'gpu_event_raw_samples':times,
            'scope':'isolated producer-consumer prototype; does not claim whole-frame promotion'}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result['checks_pass']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--matrices',type=int,choices=(1,64,1024),default=64)
    p.add_argument('--iterations',type=int,choices=(1,12),default=1)
    raise SystemExit(0 if run(p.parse_args()) else 2)
