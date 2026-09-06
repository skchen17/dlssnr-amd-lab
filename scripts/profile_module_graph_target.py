"""Finite, handshaken RGP target for one recovered module graph replay.

Correctness is checked before announcing readiness.  After readiness the only
GPU work is one already-instantiated graph replay, so a dispatch-count capture
cannot accidentally consume validation/reduction kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=('pre','c32','c64','c128','c256','c512','vit'),required=True)
    parser.add_argument('--batch',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quant-dll',type=Path,required=True);parser.add_argument('--matrix-dll',type=Path)
    parser.add_argument('--matrix-profile',choices=('reference','wmma_fp16','wmma_fp8'),default='reference')
    parser.add_argument('--matrix-modules',default='head_ffn');parser.add_argument('--matrix-waves',type=int,choices=(1,2,4),default=2)
    parser.add_argument('--attach-seconds',type=float,default=8);parser.add_argument('--post-seconds',type=float,default=5)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.batch<=0 or not 0<=args.attach_seconds<=30 or not 0<=args.post_seconds<=30:raise ValueError('bounded target settings required')
    if args.matrix_profile!='reference' and args.matrix_dll is None:raise ValueError('matrix candidate requires its DLL')

    import torch
    from native_model_package import load_package
    from native_packed_swin import RecoveredPackedSwin,logical_indices
    from native_preblock import RecoveredSingleColorPreblock
    from native_split_swin512 import RecoveredSplitSwin512,RECORD_SIZES as SPLIT_PARTS
    from native_vit1024 import RecoveredVit1024,RECORD_SIZES as VIT_PARTS
    from native_execution_policy import execution_policy
    from native_fusion_policy import fusion_policy
    from native_matrix_fusion import matrix_fusion
    from audit_head_graph import kernel_nodes
    if not torch.version.hip or not torch.cuda.is_available() or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('gfx1201 ROCm required')
    torch.set_num_threads(2);torch.manual_seed(151)
    records,_,_=load_package('local_models/native_single_color_v1','AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    channels={'pre':32,'c32':32,'c64':64,'c128':128,'c256':256,'c512':512,'vit':1024}[args.family]
    if args.family=='pre':module=RecoveredSingleColorPreblock(records[(0,0)]).swin
    elif args.family=='c512':
        module=RecoveredSplitSwin512(**{key:records[(24,index)] for index,key in enumerate(SPLIT_PARTS)})
        ai,ri=logical_indices(512);module.register_buffer('profile_a',ai);module.register_buffer('profile_r',ri)
    elif args.family=='vit':module=RecoveredVit1024(**{key:records[(31,index)] for index,key in enumerate(VIT_PARTS)})
    else:
        block={'c32':2,'c64':6,'c128':10,'c256':16}[args.family]
        module=RecoveredPackedSwin(records[(block,0)],record_kind=f'swin{channels}')
    module=module.eval().cuda().requires_grad_(False)
    if args.family=='c512':run=lambda x:module.unquantized(x[:,module.profile_a],x[:,module.profile_r])
    elif args.family=='vit':run=lambda x:module(x,query_chunk=1024)
    else:run=module
    shape=(1,args.batch,1024) if args.family=='vit' else (args.batch,64*channels)
    x=(torch.randn(*shape,dtype=torch.float16)*.1).cuda()
    modules=tuple(filter(None,args.matrix_modules.split(',')))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(),torch.cuda.stream(stream),execution_policy('native_fp16'),fusion_policy(args.quant_dll),matrix_fusion(
            args.matrix_dll,args.matrix_profile,modules=modules,waves=args.matrix_waves):
        for _ in range(2):expected=run(x)
        stream.synchronize()
        if not bool(torch.isfinite(expected).all()):raise RuntimeError('nonfinite warmup output')
        reference=hashlib.sha256(expected.cpu().numpy().tobytes()).hexdigest().upper()
        graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):actual=run(x)
        graph.instantiate();graph.replay();stream.synchronize()
        if hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest().upper()!=reference:raise RuntimeError('graph output differs')
        nodes=kernel_nodes(graph);dispatches=nodes.get('0',0)
        if dispatches<=0:raise RuntimeError('graph has no kernel nodes')
        report={'checks_pass':True,'family':args.family,'batch':args.batch,'matrix_profile':args.matrix_profile,
                'matrix_modules':list(modules),'matrix_waves':args.matrix_waves,'kernel_nodes':dispatches,
                'graph_node_types':nodes,'pre_capture_sha256':reference,
                'scope':'one full module-core HIP graph replay; excludes gather/scatter and enclosing full-frame stages'}
        print(f'RGP_MODULE_READY dispatches={dispatches}',flush=True);time.sleep(args.attach_seconds)
        graph.replay();stream.synchronize()
        # No GPU validation dispatch follows the profiled replay.  Validation
        # above used the identical immutable graph and input.
        with (args.output/'target.json').open('x',encoding='utf-8') as output:json.dump(report,output,indent=2)
        print(json.dumps(report),flush=True);time.sleep(args.post_seconds)
    return 0


if __name__=='__main__':raise SystemExit(main())
