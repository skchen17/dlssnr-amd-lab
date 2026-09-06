"""Strict gate for fused Pre feature construction, projection and C32 packing."""
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


def digest(tensor):return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest().upper()


def comparison(reference,candidate):
    import torch
    delta=candidate.float()-reference.float();different=int(torch.count_nonzero(reference.view(torch.int16)!=candidate.view(torch.int16)).item())
    rmse=float(torch.sqrt(torch.mean(delta.square())).item());den=float(torch.sqrt(torch.mean(reference.float().square())).item())
    return {"bitwise_exact":different==0,"different_components":different,"max_absolute_error":float(delta.abs().max().item()),
        "rmse":rmse,"nrmse":rmse/den if den else (0.0 if rmse==0 else None)}


def exercise(args):
    import torch
    from native_compact_layout import compact_layout
    from native_execution_policy import execution_policy
    from native_matrix_fusion import matrix_fusion
    from native_model_package import load_package
    from native_multiscale import pack_image
    from native_pre_fusion import pre_features_fusion
    from native_preblock import RecoveredSingleColorPreblock,positional_noise,project_input_features,single_color_features
    if not torch.version.hip or "gfx1201" not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError("gfx1201 ROCm required")
    torch.set_num_threads(2);torch.manual_seed(113);props=torch.cuda.get_device_properties(0)
    records,_,_=load_package("local_models/native_single_color_v1","AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3")
    pre=RecoveredSingleColorPreblock(records[(0,0)]).eval().cuda().requires_grad_(False)
    h,w=args.height,args.width;color=(torch.rand(h,w,4,device="cuda",dtype=torch.float16)*.8+.1).half()
    pw,ph=(w+127)//128*128,(h+127)//128*128;noise=positional_noise(pw,ph,71,color.device)
    conditioning=(0.0,1.0,-0.5,0.25,2.0);scale=1.25;stream=torch.cuda.current_stream()

    def run(fused):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),execution_policy("native_fp16"),compact_layout(True),pre_features_fusion(
                args.quant_dll,project_pack=fused) as op,matrix_fusion(
                args.matrix_dll,"wmma_fp16",modules=("pre_input",),waves=2):
            def once():
                if fused:return op.project_pack(color,noise,pre.input_project,pw,ph,scale,conditioning)
                features=single_color_features(color,pw,ph,71,color_scale=scale,conditioning=conditioning)
                return pack_image(project_input_features(features,pre.input_project))
            for _ in range(2):output=once()
            torch.cuda.synchronize();times=[]
            for _ in range(args.iterations):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record(stream)
                output=once();end.record(stream);end.synchronize();times.append(begin.elapsed_time(end))
            if not bool(torch.isfinite(output).all()):raise RuntimeError("nonfinite Pre packed projection")
            result=output.clone();row={"fused":fused,"gpu_event_ms":times,"median_gpu_event_ms":statistics.median(times),
                "project_pack_launches":op.project_pack_launches,"feature_launches":op.launches,
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "sha256":digest(result)}
        return result,row
    baseline,base_row=run(False);candidate,candidate_row=run(True);error=comparison(baseline,candidate)
    return {"checks_pass":error["bitwise_exact"],"scope":"same supplied positional noise; feature construction + FP16 16x32 projection + compact C32 packing",
        "geometry":[w,h],"padded_geometry":[pw,ph],"baseline":base_row,"candidate":candidate_row,"comparison":error,
        "speedup":base_row["median_gpu_event_ms"]/candidate_row["median_gpu_event_ms"],"device":props.name,
        "noise_generator_changed":False,"weights_modified":False,"default_promoted":False}


def child(args):
    def phase(name):
        with (args.output/"phases.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps({"phase":name,"pid":os.getpid(),"monotonic":time.monotonic()})+"\n")
    atexit.register(phase,"python_atexit");phase("child_enter")
    import ctypes as ct
    dll=ct.CDLL(str(args.quant_dll.resolve()))
    for symbol in ("native_fusion_pre_features","native_fusion_pre_project_pack"):
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
    parser.add_argument("--iterations",type=int,choices=(1,12),default=12);parser.add_argument("--child",action="store_true");args=parser.parse_args()
    if min(args.width,args.height)<=0:parser.error("positive geometry required")
    if args.child:return child(args)
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),"--child","--quant-dll",str(args.quant_dll.resolve()),"--matrix-dll",str(args.matrix_dll.resolve()),
        "--output",str(args.output.resolve()),"--width",str(args.width),"--height",str(args.height),"--iterations",str(args.iterations)]
    return supervise(command,args.output,timeout=180)


if __name__=="__main__":raise SystemExit(0 if main() else 2)
