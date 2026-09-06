"""Strict future A/B gate for bounded-LDS C32 attention.

The repository safety halt is checked before a child process or HIP runtime is
started.  Merely generating or compiling this validator does not access a GPU.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from validate_rocm_lifecycle import save,supervise

BASE_MODULES=('c32_ffn','c32_attention')


def digest(tensor):return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest().upper()


def compare(reference,candidate):
    import torch
    delta=candidate.float()-reference.float()
    different=int(torch.count_nonzero(candidate.view(torch.int16)!=reference.view(torch.int16)).item())
    rmse=float(torch.sqrt(torch.mean(delta.square())).item());den=float(torch.sqrt(torch.mean(reference.float().square())).item())
    return {'bitwise_exact':different==0,'different_components':different,
            'max_absolute_error':float(delta.abs().max().item()),'rmse':rmse,
            'nrmse':rmse/den if den else (0.0 if rmse==0 else None)}


def exercise(args):
    import torch
    from native_execution_policy import execution_policy
    from native_fusion_policy import fusion_policy
    from native_matrix_fusion import matrix_fusion
    from native_model_package import load_package
    from native_packed_swin import RecoveredPackedSwin
    torch.set_num_threads(2);torch.manual_seed(163)
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('ROCm required')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('gfx1201 required')
    records,_,_=load_package('local_models/native_single_color_v1','AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    block=RecoveredPackedSwin(records[(2,0)],record_kind='swin32').eval().cuda().requires_grad_(False)
    ramp=torch.linspace(-.25,.25,1<<16,device='cuda',dtype=torch.float16)
    raw=ramp.repeat((args.windows*2048+len(ramp)-1)//len(ramp))[:args.windows*2048].reshape(args.windows,2048).contiguous()
    stream=torch.cuda.current_stream()
    def run(modules):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),execution_policy('native_fp16'),fusion_policy(args.quant_dll),matrix_fusion(
                args.matrix_dll,'wmma_fp16',modules=modules,waves=args.waves) as matrix:
            for _ in range(2):output=block(raw)
            torch.cuda.synchronize();times=[]
            for _ in range(args.iterations):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record(stream);output=block(raw);end.record(stream);end.synchronize();times.append(begin.elapsed_time(end))
            if not bool(torch.isfinite(output).all()):raise RuntimeError('nonfinite C32 output')
            result=output.clone();calls=args.iterations+2
            row={'modules':list(modules),'gpu_event_ms':times,'median_gpu_event_ms':statistics.median(times),
                 'matrix_launches_per_call':matrix.launches/calls,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'sha256':digest(result)}
        return result,row
    candidate_family={'bounded':'c32_attention_bounded','staged':'c32_attention_staged','core':'c32_attention_core'}[args.candidate]
    candidate_modules=('c32_ffn',candidate_family)
    baseline,base_row=run(BASE_MODULES);candidate,candidate_row=run(candidate_modules);equality=compare(baseline,candidate)
    scope={'bounded':'same C32 FFN; six global attention intermediates versus one bounded per-window LDS attention',
        'staged':'same C32 FFN; six global attention stages versus QKV/norm + 8 KiB attention core + projection',
        'core':'same C32 FFN; established QKV/norm + 8 KiB fused QK/softmax/PV + projection'}[args.candidate]
    return {'checks_pass':equality['bitwise_exact'],'scope':scope,'candidate_strategy':args.candidate,
            'windows':args.windows,'waves':args.waves,'baseline':base_row,'candidate':candidate_row,'comparison':equality,
            'speedup':base_row['median_gpu_event_ms']/candidate_row['median_gpu_event_ms'],
            'device':props.name,'architecture':props.gcnArchName,'weights_modified':False,'default_promoted':False}


def child(args):
    def phase(name):
        with (args.output/'phases.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps({'phase':name,'pid':os.getpid(),'monotonic':time.monotonic()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter')
    import ctypes as ct
    dll=ct.CDLL(str(args.matrix_dll.resolve()))
    symbols={'bounded':('nr_c32_ffn_wmma','nr_c32_attention_window_fused'),
        'staged':('nr_c32_ffn_wmma','nr_c32_qkv_norm_staged','nr_c32_attention_core_staged','nr_c32_project_wmma'),
        'core':('nr_c32_ffn_wmma','nr_head_qkv_project_wmma','nr_c32_qkv_norm','nr_c32_attention_core_staged','nr_c32_project_wmma')}[args.candidate]
    for symbol in symbols:
        if not hasattr(dll,symbol):raise ValueError(f'missing required symbol {symbol}')
    phase('abi_verified');result=exercise(args);phase('gpu_work_complete')
    import torch
    gc.collect();torch.cuda.empty_cache()
    if hasattr(torch._C,'_cuda_clearCublasWorkspaces'):torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();result['allocated_after_release_bytes']=torch.cuda.memory_allocated();result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
    result['checks_pass']&=result['allocated_after_release_bytes']==result['reserved_after_release_bytes']==0
    phase('resources_released');save(args.output/'child.json',result);return result['checks_pass']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--quant-dll',type=Path,required=True)
    parser.add_argument('--matrix-dll',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',choices=('bounded','staged','core'),default='bounded')
    parser.add_argument('--windows',type=int,choices=(1,16,256,768),required=True);parser.add_argument('--waves',type=int,choices=(1,2,4),default=2)
    parser.add_argument('--iterations',type=int,choices=(1,12),default=12);parser.add_argument('--child',action='store_true');args=parser.parse_args()
    require_gpu_tests_enabled('C32 bounded-attention GPU gate')
    if args.child:return child(args)
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--quant-dll',str(args.quant_dll.resolve()),
             '--matrix-dll',str(args.matrix_dll.resolve()),'--output',str(args.output.resolve()),
             '--candidate',args.candidate,'--windows',str(args.windows),'--waves',str(args.waves),'--iterations',str(args.iterations)]
    return supervise(command,args.output,timeout=120)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
