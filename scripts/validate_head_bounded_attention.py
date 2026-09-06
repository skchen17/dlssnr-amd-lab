"""Strict A/B gate for the bounded-LDS whole-grid Output Head attention."""
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

from validate_rocm_lifecycle import save,supervise

BASE_MODULES=("head_ffn","head_attention","head_softmax","head_output")
CANDIDATE_MODULES=("head_ffn","head_attention_bounded","head_output")


def digest(tensor):return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest().upper()


def compare(reference,candidate):
    import torch
    delta=candidate.float()-reference.float();different=int(torch.count_nonzero(candidate.view(torch.int16)!=reference.view(torch.int16)).item())
    rmse=float(torch.sqrt(torch.mean(delta.square())).item());den=float(torch.sqrt(torch.mean(reference.float().square())).item())
    return {"bitwise_exact":different==0,"different_components":different,
        "max_absolute_error":float(delta.abs().max().item()),"rmse":rmse,
        "nrmse":rmse/den if den else (0.0 if rmse==0 else None)}


def exercise(args):
    import torch
    from native_execution_policy import execution_policy
    from native_head_fusion import head_input_fusion
    from native_head_torch import RecoveredHead32
    from native_matrix_fusion import matrix_fusion
    from native_model_package import load_package
    torch.set_num_threads(2);torch.manual_seed(107)
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError("ROCm required")
    props=torch.cuda.get_device_properties(0)
    if "gfx1201" not in props.gcnArchName:raise RuntimeError("gfx1201 required")
    torch.cuda.set_per_process_memory_fraction(args.allocator_budget_mb*1_000_000/props.total_memory)
    records,_,_=load_package("local_models/native_single_color_v1","AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3")
    head=RecoveredHead32(records[(70,0)]).eval().cuda().requires_grad_(False)
    pw,ph=args.width,args.height
    if pw%8 or ph%8:raise ValueError("padded dimensions must be divisible by eight")
    count=((pw+11)//8)*((ph+11)//8);ramp=torch.linspace(-.25,.25,1<<16,device="cuda",dtype=torch.float16)
    def fixture(elements,offset):return ramp.roll(offset).repeat((elements+len(ramp)-1)//len(ramp))[:elements].contiguous()
    main,skip=fixture(pw*ph*8,13),fixture(pw*ph*32,29);stream=torch.cuda.current_stream()

    def run(modules):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),execution_policy("native_fp16"),head_input_fusion(
                args.quant_dll,epilogue=True,qkv=True,whole_grid=True) as fusion,matrix_fusion(
                args.matrix_dll,"wmma_fp16",modules,waves=args.waves) as matrix:
            def once():return head.forward_range(main,skip,0,count,pw,ph)
            for _ in range(2):output=once()
            torch.cuda.synchronize();times=[]
            for _ in range(args.iterations):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record(stream);output=once();end.record(stream);end.synchronize();times.append(begin.elapsed_time(end))
            if not bool(torch.isfinite(output).all()):raise RuntimeError("nonfinite Head output")
            result=output.clone();calls=args.iterations+2
            launches=fusion.launches+fusion.qkv_launches+fusion.add_launches+matrix.launches
            row={"modules":modules,"gpu_event_ms":times,"median_gpu_event_ms":statistics.median(times),
                "observed_native_launches_per_call":launches/calls,"head_input_launches":fusion.launches,
                "head_qkv_norm_launches":fusion.qkv_launches,"matrix_launches":matrix.launches,
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "sha256":digest(result)}
        return result,row
    baseline,baseline_row=run(BASE_MODULES);candidate,candidate_row=run(CANDIDATE_MODULES)
    equality=compare(baseline,candidate)
    return {"checks_pass":equality["bitwise_exact"],"scope":"same native WMMA math; global intermediates versus bounded per-window LDS attention",
        "geometry":[pw,ph],"cta_count":count,"waves":args.waves,"baseline":baseline_row,"candidate":candidate_row,
        "comparison":equality,"speedup":baseline_row["median_gpu_event_ms"]/candidate_row["median_gpu_event_ms"],
        "device":props.name,"architecture":props.gcnArchName,"weights_modified":False,"default_promoted":False}


def child(args):
    def phase(name):
        with (args.output/"phases.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps({"phase":name,"pid":os.getpid(),"monotonic":time.monotonic()})+"\n")
    atexit.register(phase,"python_atexit");phase("child_enter")
    import ctypes as ct
    for library,symbols in ((args.quant_dll,("native_fusion_head_input","native_fusion_head_qkv")),
            (args.matrix_dll,("nr_head_ffn_wmma","nr_head_attention_window_fused","nr_head_tail_exact"))):
        dll=ct.CDLL(str(library.resolve()))
        for symbol in symbols:
            if not hasattr(dll,symbol):raise ValueError(f"missing required symbol {symbol}")
    phase("abi_verified");result=exercise(args);phase("gpu_work_complete")
    import torch
    gc.collect();torch.cuda.empty_cache()
    if hasattr(torch._C,"_cuda_clearCublasWorkspaces"):torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();result["allocated_after_release_bytes"]=torch.cuda.memory_allocated();result["reserved_after_release_bytes"]=torch.cuda.memory_reserved()
    result["checks_pass"]&=result["allocated_after_release_bytes"]==result["reserved_after_release_bytes"]==0
    phase("resources_released");save(args.output/"child.json",result);return result["checks_pass"]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--quant-dll",type=Path,required=True)
    parser.add_argument("--matrix-dll",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--width",type=int,required=True);parser.add_argument("--height",type=int,required=True)
    parser.add_argument("--waves",type=int,choices=(1,2,4),default=2);parser.add_argument("--iterations",type=int,choices=(1,12),default=12)
    parser.add_argument("--allocator-budget-mb",type=int,choices=(4000,5000),default=5000);parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    if min(args.width,args.height)<=0:parser.error("positive geometry required")
    if args.child:return child(args)
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),"--child","--quant-dll",str(args.quant_dll.resolve()),"--matrix-dll",str(args.matrix_dll.resolve()),
        "--output",str(args.output.resolve()),"--width",str(args.width),"--height",str(args.height),"--waves",str(args.waves),
        "--iterations",str(args.iterations),"--allocator-budget-mb",str(args.allocator_budget_mb)]
    return supervise(command,args.output,timeout=180)


if __name__=="__main__":raise SystemExit(0 if main() else 2)
