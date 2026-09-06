"""Isolate large FP32 preprojection; small CPU row oracle is diagnostic only."""
import argparse
import atexit
import gc
import json
from pathlib import Path
import sys
import time
from validate_rocm_lifecycle import supervise,save,configure_fault_logging
from profile_native_boundary_batch import case_spec,sha,compare_bytes,fingerprints


def exercise(args,phase):
    import torch
    from native_preblock import single_color_features
    from native_model_package import load_package
    from native_whole_frame import feature_geometry
    torch.set_num_threads(2)
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('unreviewed GPU')
    def wait():
        e=torch.cuda.Event();e.record();deadline=time.monotonic()+10
        while not e.query():
            if time.monotonic()>deadline:raise TimeoutError('GPU timeout; no cancellation/retry')
            time.sleep(.001)
    path,(w,h),expected,kind=case_spec(args.case);raw=path.read_bytes()
    if sha(raw)!=expected:raise ValueError('input changed')
    records,origins,settings=load_package(Path('local_models/native_single_color_v1'),
          'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    from native_preblock import RecoveredSingleColorPreblock
    model=RecoveredSingleColorPreblock(records[(0,0)]).eval().cuda().requires_grad_(False)
    color=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(h,w,4).cuda()
    pw,ph=feature_geometry(w,h,4000000)
    features=single_color_features(color,pw,ph,0,**settings).float()
    weight=model.input_project.float();wait()
    feature_hash=sha(features.cpu().numpy().tobytes());weight_hash=sha(weight.cpu().numpy().tobytes())
    phase('gpu_initialized')
    # Independently evaluate distributed rows on CPU as a diagnostic oracle.
    rows=torch.linspace(0,pw*ph-1,1024,device=features.device).long()
    sample=features.reshape(-1,16)[rows].cpu()
    reference=(sample@weight.cpu()).half().numpy().tobytes()
    modes=['original_3d','row_chunk_fp32'] if args.mode=='both' else [args.mode]
    cases=[]
    for mode in modes:
        previous=None
        for repeat in range(2):
            start=time.perf_counter()
            if mode=='original_3d':out=(features@weight).half()
            else:
                flat=features.reshape(-1,16)
                out=torch.cat([(part@weight).half() for part in flat.split(65536)]).reshape(ph,pw,32)
            wait();ms=(time.perf_counter()-start)*1000
            actual=out.cpu().numpy().tobytes()
            sample_out=out.reshape(-1,32)[rows].cpu().numpy().tobytes()
            item={'mode':mode,'repeat':repeat,'host_ms':ms,'sha256':sha(actual),
                  'same_input':sha(features.cpu().numpy().tobytes())==feature_hash,
                  'same_weight':sha(weight.cpu().numpy().tobytes())==weight_hash,
                  'repeat_comparison':compare_bytes(previous,actual) if previous else None,
                  'cpu_1024_rows_comparison':compare_bytes(reference,sample_out)}
            cases.append(item);previous=actual
            with (args.output/'cases.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
            phase(f'{mode}_{repeat}_complete')
    phase('gpu_work_complete')
    return {'checks_pass':all(c['same_input'] and c['same_weight'] and
                  c['cpu_1024_rows_comparison']['bitwise_exact'] and
                  (c['repeat_comparison'] is None or c['repeat_comparison']['bitwise_exact']) for c in cases),
            'cases':cases,'input_geometry':[w,h],'feature_shape':list(features.shape),
            'feature_sha256':feature_hash,'weight_sha256':weight_hash,'input_source':kind,
            'cpu_neural_fallback':False,'cpu_oracle_only':True,'game_runtime_ready':False,
            'tf32_allowed':torch.backends.cuda.matmul.allow_tf32}


def child(args):
    initial=fingerprints();save(args.output/'sources_at_start.json',initial)
    trace=(args.output/'traceback.log').open('x');configure_fault_logging(trace);globals()['_trace']=trace
    def phase(name):
        with (args.output/'phases.jsonl').open('a') as stream:stream.write(json.dumps({'phase':name})+'\n')
    atexit.register(phase,'python_atexit')
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
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=('128','640','2342'),required=True)
    p.add_argument('--mode',choices=('original_3d','row_chunk_fp32','both'),default='original_3d')
    p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:return child(a)
    a.output.mkdir(parents=True,exist_ok=False)
    return supervise([sys.executable,str(Path(__file__).resolve()),'--child','--case',a.case,'--mode',a.mode,
                      '--output',str(a.output.resolve())],a.output,timeout=90)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
