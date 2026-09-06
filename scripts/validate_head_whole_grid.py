"""Correctness/performance gate for the opt-in whole-grid Output Head schedule.

Both sides execute the same native kernels, weights and precision policy.  The
only changed variable is Python CTA batching versus one complete CTA range.
This is a scheduling gate, not evidence that the WMMA approximation matches an
RTX teacher or the tensor reference.
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

from validate_rocm_lifecycle import save, supervise


MATRIX_MODULES=("head_ffn","head_attention","head_softmax","head_output")


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest().upper()


def compare(reference, candidate):
    import torch
    delta=candidate.float()-reference.float()
    different=int(torch.count_nonzero(candidate.view(torch.int16)!=reference.view(torch.int16)).item())
    denominator=float(torch.sqrt(torch.mean(reference.float().square())).item())
    rmse=float(torch.sqrt(torch.mean(delta.square())).item())
    return {
        "bitwise_exact":different==0,
        "different_components":different,
        "max_absolute_error":float(delta.abs().max().item()),
        "rmse":rmse,
        "nrmse":rmse/denominator if denominator else (0.0 if rmse==0 else None),
    }


def exercise(args):
    import torch
    from native_head_torch import RecoveredHead32
    from native_model_package import load_package
    from native_head_fusion import head_input_fusion
    from native_matrix_fusion import matrix_fusion
    from native_execution_policy import execution_policy

    torch.set_num_threads(2);torch.manual_seed(97)
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError("ROCm required")
    props=torch.cuda.get_device_properties(0)
    if "gfx1201" not in props.gcnArchName:raise RuntimeError("this gate is restricted to gfx1201")
    torch.cuda.set_per_process_memory_fraction(args.allocator_budget_mb*1_000_000/props.total_memory)
    records,_,_=load_package("local_models/native_single_color_v1","AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3")
    head=RecoveredHead32(records[(70,0)]).eval().cuda().requires_grad_(False)
    pw,ph=args.width,args.height
    if pw%8 or ph%8:raise ValueError("padded dimensions must be divisible by eight")
    count=((pw+11)//8)*((ph+11)//8)
    # A repeated finite ramp avoids allocating a temporary FP32 random tensor
    # larger than the actual fixture and is deterministic across processes.
    ramp=torch.linspace(-.25,.25,1<<16,device="cuda",dtype=torch.float16)
    def fixture(elements,offset):
        repeats=(elements+len(ramp)-1)//len(ramp)
        return (ramp.roll(offset).repeat(repeats)[:elements]).contiguous()
    main=fixture(pw*ph*8,13);skip=fixture(pw*ph*32,29)
    stream=torch.cuda.current_stream()
    def run(whole_grid):
        torch.cuda.reset_peak_memory_stats();fusion=None;matrix=None
        with torch.no_grad(),execution_policy("native_fp16"),head_input_fusion(
                args.quant_dll,epilogue=True,qkv=True,whole_grid=whole_grid) as fusion,matrix_fusion(
                args.matrix_dll,"wmma_fp16",MATRIX_MODULES,waves=args.waves) as matrix:
            def once():
                if whole_grid:return head.forward_range(main,skip,0,count,pw,ph)
                return torch.cat([head.forward_range(main,skip,start,min(args.batch,count-start),pw,ph)
                                  for start in range(0,count,args.batch)])
            for _ in range(2):output=once()
            torch.cuda.synchronize();times=[]
            for _ in range(args.iterations):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record(stream);output=once();end.record(stream);end.synchronize();times.append(begin.elapsed_time(end))
            if not bool(torch.isfinite(output).all()):raise RuntimeError("nonfinite Head output")
            result=output.clone()
            row={"whole_grid":whole_grid,"gpu_event_ms":times,"median_gpu_event_ms":statistics.median(times),
                 "head_input_launches":fusion.launches,"matrix_launches":matrix.launches,
                 "expected_native_dispatches_per_whole_grid_call":9,
                 "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"sha256":digest(result)}
            # The matrix path is FFN + QKV + QK + softmax + PV + project + tail
            # (7), plus Head input (1).  Use observed launch deltas for totals;
            # do not infer fusion from the Python call count above.
            calls=args.iterations+2
            native_launches=fusion.launches+fusion.qkv_launches+fusion.add_launches+matrix.launches
            row["observed_native_launches_per_call"]=native_launches/calls
            del output
        return result,row
    baseline,base_row=run(False)
    candidate,candidate_row=run(True)
    equality=compare(baseline,candidate)
    return {"checks_pass":equality["bitwise_exact"],"scope":"same native math; schedule-only batched versus whole-grid A/B",
            "geometry":[pw,ph],"cta_count":count,"reference_batch":args.batch,"waves":args.waves,
            "rows":[base_row,candidate_row],"comparison":equality,
            "speedup":base_row["median_gpu_event_ms"]/candidate_row["median_gpu_event_ms"],
            "device":props.name,"architecture":props.gcnArchName,"weights_modified":False,
            "rtx_quality_accepted":False,"game_runtime_ready":False}


def child(args):
    def phase(name):
        with (args.output/"phases.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps({"phase":name,"pid":os.getpid(),"monotonic":time.monotonic()})+"\n")
    atexit.register(phase,"python_atexit");phase("child_enter")
    # Fail before HIP initialization when an accidentally supplied older probe
    # library does not carry the complete Head ABI.
    import ctypes as ct
    quant=ct.CDLL(str(args.quant_dll.resolve()))
    for symbol in ("native_fusion_head_input","native_fusion_head_qkv"):
        if not hasattr(quant,symbol):raise ValueError(f"quant DLL is missing required symbol {symbol}")
    matrix=ct.CDLL(str(args.matrix_dll.resolve()))
    for symbol in ("nr_head_ffn_wmma","nr_head_qkv_project_wmma","nr_head_qk_wmma",
                   "nr_head_softmax_e4","nr_head_pv_wmma","nr_head_project_wmma","nr_head_tail_exact"):
        if not hasattr(matrix,symbol):raise ValueError(f"matrix DLL is missing required symbol {symbol}")
    phase("abi_verified")
    result=exercise(args);phase("gpu_work_complete")
    import torch
    gc.collect();torch.cuda.empty_cache()
    if hasattr(torch._C,"_cuda_clearCublasWorkspaces"):torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    result["allocated_after_release_bytes"]=torch.cuda.memory_allocated()
    result["reserved_after_release_bytes"]=torch.cuda.memory_reserved()
    result["checks_pass"] &= result["allocated_after_release_bytes"]==result["reserved_after_release_bytes"]==0
    phase("resources_released");save(args.output/"child.json",result)
    return result["checks_pass"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-dll",type=Path,required=True);parser.add_argument("--matrix-dll",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--width",type=int,required=True)
    parser.add_argument("--height",type=int,required=True);parser.add_argument("--batch",type=int,default=96)
    parser.add_argument("--waves",type=int,choices=(1,2,4),default=2);parser.add_argument("--iterations",type=int,choices=(1,12),default=12)
    parser.add_argument("--allocator-budget-mb",type=int,choices=(4000,5000),default=5000);parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    if min(args.width,args.height,args.batch)<=0:parser.error("positive geometry and batch required")
    if args.child:return child(args)
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),"--child","--quant-dll",str(args.quant_dll.resolve()),
             "--matrix-dll",str(args.matrix_dll.resolve()),"--output",str(args.output.resolve()),"--width",str(args.width),
             "--height",str(args.height),"--batch",str(args.batch),"--waves",str(args.waves),"--iterations",str(args.iterations),
             "--allocator-budget-mb",str(args.allocator_budget_mb)]
    return supervise(command,args.output,timeout=180)


if __name__=="__main__":raise SystemExit(0 if main() else 2)
