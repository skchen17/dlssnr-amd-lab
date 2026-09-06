"""Finite Output Head target whose measured region contains only one schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


MODULES=("head_ffn","head_attention","head_softmax","head_output")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-dll",type=Path,required=True);parser.add_argument("--matrix-dll",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--schedule",choices=("batched","whole_grid","bounded"),required=True)
    parser.add_argument("--width",type=int,default=640);parser.add_argument("--height",type=int,default=384)
    parser.add_argument("--batch",type=int,default=96);parser.add_argument("--waves",type=int,choices=(1,2,4),default=2)
    parser.add_argument("--attach-seconds",type=float,default=10);parser.add_argument("--post-seconds",type=float,default=20)
    parser.add_argument("--iterations",type=int,default=12);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if min(args.width,args.height,args.batch,args.iterations)<=0 or args.width%8 or args.height%8:
        raise ValueError("positive eight-aligned geometry and bounded schedule required")
    if not 0<=args.attach_seconds<=30 or not 0<=args.post_seconds<=30:raise ValueError("bounded profiler waits required")

    import torch
    from native_head_torch import RecoveredHead32
    from native_model_package import load_package
    from native_head_fusion import head_input_fusion
    from native_matrix_fusion import matrix_fusion
    from native_execution_policy import execution_policy
    if not torch.version.hip or "gfx1201" not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError("gfx1201 ROCm required")
    torch.set_num_threads(2);torch.manual_seed(97)
    records,_,_=load_package("local_models/native_single_color_v1","AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3")
    head=RecoveredHead32(records[(70,0)]).eval().cuda().requires_grad_(False)
    pw,ph=args.width,args.height;count=((pw+11)//8)*((ph+11)//8)
    ramp=torch.linspace(-.25,.25,1<<16,device="cuda",dtype=torch.float16)
    def fixture(elements,offset):return ramp.roll(offset).repeat((elements+len(ramp)-1)//len(ramp))[:elements].contiguous()
    main,skip=fixture(pw*ph*8,13),fixture(pw*ph*32,29)
    whole=args.schedule in ("whole_grid","bounded")
    modules=("head_ffn","head_attention_bounded","head_output") if args.schedule=="bounded" else MODULES
    with torch.no_grad(),execution_policy("native_fp16"),head_input_fusion(
            args.quant_dll,epilogue=True,qkv=True,whole_grid=whole),matrix_fusion(
            args.matrix_dll,"wmma_fp16",modules,waves=args.waves):
        def once():
            if whole:return head.forward_range(main,skip,0,count,pw,ph)
            return torch.cat([head.forward_range(main,skip,start,min(args.batch,count-start),pw,ph)
                              for start in range(0,count,args.batch)])
        for _ in range(2):output=once()
        torch.cuda.synchronize();print("RGP_HEAD_READY",flush=True);time.sleep(args.attach_seconds)
        times=[]
        for _ in range(args.iterations):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();output=once();end.record();end.synchronize();times.append(begin.elapsed_time(end))
        finite=bool(torch.isfinite(output).all())
        raw=output.cpu().numpy().tobytes()
    report={"checks_pass":finite,"schedule":args.schedule,"geometry":[pw,ph],"cta_count":count,
            "batch":args.batch,"waves":args.waves,"iterations":args.iterations,"gpu_event_ms":times,
            "median_gpu_event_ms":statistics.median(times),"sha256":hashlib.sha256(raw).hexdigest().upper(),
            "scope":"RGP target; same full native Head math, one selected scheduling path"}
    with (args.output/"target.json").open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True);time.sleep(args.post_seconds)
    return 0 if finite else 2


if __name__=="__main__":raise SystemExit(main())
