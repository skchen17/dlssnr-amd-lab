"""Prepare Head intermediates, then expose exactly one named native dispatch to RGP."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


STAGES=("input_pack","ffn","qkv_project","qkv_norm","qk","softmax","pv","projection","tail")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--quant-dll",type=Path,required=True)
    parser.add_argument("--matrix-dll",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=STAGES,required=True);parser.add_argument("--arm-seconds",type=float,default=2)
    parser.add_argument("--post-seconds",type=float,default=5);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import torch
    from native_execution_policy import execution_policy
    from native_head_fusion import head_input_fusion
    from native_head_torch import RecoveredHead32
    from native_matrix_fusion import matrix_fusion
    from native_model_package import load_package
    if not torch.version.hip or "gfx1201" not in torch.cuda.get_device_properties(0).gcnArchName:raise RuntimeError("gfx1201 ROCm required")
    torch.set_num_threads(2);torch.manual_seed(109)
    records,_,_=load_package("local_models/native_single_color_v1","AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3")
    head=RecoveredHead32(records[(70,0)]).eval().cuda().requires_grad_(False);pw,ph=640,384
    windows=((pw+11)//8)*((ph+11)//8);ramp=torch.linspace(-.25,.25,1<<16,device="cuda",dtype=torch.float16)
    def fixture(elements,offset):return ramp.roll(offset).repeat((elements+len(ramp)-1)//len(ramp))[:elements].contiguous()
    main,skip=fixture(pw*ph*8,13),fixture(pw*ph*32,29)
    modules=("head_ffn","head_attention","head_softmax","head_output")
    with torch.no_grad(),execution_policy("native_fp16"),head_input_fusion(
            args.quant_dll,epilogue=True,qkv=True,whole_grid=True) as fusion,matrix_fusion(
            args.matrix_dll,"wmma_fp16",modules,waves=2) as matrix:
        # All allocations, weight transforms and prerequisite math finish before
        # the ready marker. The post-marker call therefore has one native launch.
        raw=fusion(main,skip,head.main_scale,head.skip_scale,0,windows,pw,ph)
        first=matrix.head_ffn(head.block,raw)
        projection=matrix.head_qkv(head.block,first)
        q,k,v=fusion.qkv(projection,head.block.q_scale)
        scores=matrix.head_scores(head.block,q,k)
        probability=matrix.head_probability(scores)
        value=matrix.head_pv(probability,v)
        post=matrix.head_project(head.block,first,value)
        tail=matrix.head_tail(head,post)
        torch.cuda.synchronize();print("RGP_DISPATCH_READY",flush=True);time.sleep(args.arm_seconds)
        if args.stage=="input_pack":result=fusion(main,skip,head.main_scale,head.skip_scale,0,windows,pw,ph)
        elif args.stage=="ffn":result=matrix.head_ffn(head.block,raw)
        elif args.stage=="qkv_project":result=matrix.head_qkv(head.block,first)
        elif args.stage=="qkv_norm":result=fusion.qkv(projection,head.block.q_scale)[0]
        elif args.stage=="qk":result=matrix.head_scores(head.block,q,k)
        elif args.stage=="softmax":result=matrix.head_probability(scores)
        elif args.stage=="pv":result=matrix.head_pv(probability,v)
        elif args.stage=="projection":result=matrix.head_project(head.block,first,value)
        else:result=matrix.head_tail(head,post)
        torch.cuda.synchronize();finite=bool(torch.isfinite(result).all())
    report={"checks_pass":finite,"stage":args.stage,"geometry":[pw,ph],"cta_count":windows,
        "native_dispatches_after_ready":1,"scope":"prerequisites synchronized before ready marker; one selected native call after profiler arm delay"}
    (args.output/"target.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True);time.sleep(args.post_seconds);return 0 if finite else 2


if __name__=="__main__":raise SystemExit(main())
