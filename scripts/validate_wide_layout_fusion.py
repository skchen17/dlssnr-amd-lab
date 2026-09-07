"""Supervised bitwise gate for C64/C128/C256/C512 whole-grid layouts."""
from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from validate_rocm_lifecycle import save, supervise


def digest(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest().upper()


def exercise(args):
    import torch
    from native_c32_layout_fusion import c32_layout_fusion
    from native_packed_swin import gather_packed, scatter_packed

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm required')
    props=torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(1_000_000_000/props.total_memory)
    torch.manual_seed(9182)
    rows=[]
    for channels in (64,128,256,512):
        for ox,oy in ((0,0),(-4,-4)):
            source=(torch.randn(args.width*args.height*channels,device='cuda',dtype=torch.float16)*.1).contiguous()
            expected,mapping=gather_packed(source,args.width,args.height,channels,ox,oy)
            expected_scatter=scatter_packed(expected.reshape(-1,64,channels),mapping,args.width,args.height)
            with c32_layout_fusion(args.dll,(64,128,256,512)) as operator:
                actual,native_mapping=gather_packed(source,args.width,args.height,channels,ox,oy)
                actual_scatter=scatter_packed(actual.reshape(-1,64,channels),native_mapping,args.width,args.height)
                fixed=actual.clone()
                repeated,_=gather_packed(source,args.width,args.height,channels,ox,oy)
                calls=operator.calls_by_channel[channels].copy()
            event=torch.cuda.Event();event.record();deadline=time.monotonic()+10
            while not event.query():
                if time.monotonic()>=deadline:
                    save(args.output/'gpu_timeout.json',{'resources_retained':True})
                    while True:time.sleep(1)
                time.sleep(.001)
            gather_exact=bool(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
            scatter_exact=bool(torch.equal(actual_scatter.view(torch.int16),expected_scatter.view(torch.int16)))
            repeat_exact=bool(torch.equal(repeated.view(torch.int16),fixed.view(torch.int16)))
            rows.append({'channels':channels,'offset':[ox,oy],'gather_bitwise_exact':gather_exact,
                         'scatter_bitwise_exact':scatter_exact,'repeat_bitwise_exact':repeat_exact,
                         'finite':bool(torch.isfinite(actual).all()),'calls':calls,
                         'output_sha256':digest(actual),'scatter_sha256':digest(actual_scatter)})
            del source,expected,mapping,expected_scatter,actual,native_mapping,actual_scatter,fixed,repeated,event
    return {'checks_pass':all(all(row[key] for key in ('gather_bitwise_exact','scatter_bitwise_exact','repeat_bitwise_exact','finite')) for row in rows),
            'geometry':[args.width,args.height],'rows':rows,'kernel_launches_per_conversion':1,
            'scope':'value-preserving whole-grid packed-image/window layout; no numeric operation',
            'device':props.name,'architecture':props.gcnArchName,'default_promoted':False}


def child(args):
    def phase(name):
        with (args.output/'phases.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter')
    result=exercise(args);phase('gpu_work_complete')
    import torch
    gc.collect();torch.cuda.empty_cache()
    if hasattr(torch._C,'_cuda_clearCublasWorkspaces'):torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    result['allocated_after_release_bytes']=torch.cuda.memory_allocated()
    result['reserved_after_release_bytes']=torch.cuda.memory_reserved()
    result['checks_pass'] &= result['allocated_after_release_bytes']==result['reserved_after_release_bytes']==0
    phase('resources_released');save(args.output/'child.json',result);return result['checks_pass']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll',type=Path,required=True)
    parser.add_argument('--width',type=int,choices=(32,128),default=32)
    parser.add_argument('--height',type=int,choices=(32,128),default=32)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--child',action='store_true');args=parser.parse_args()
    require_gpu_tests_enabled('wide whole-grid layout gate')
    if args.child:return child(args)
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--child','--dll',str(args.dll.resolve()),
             '--width',str(args.width),'--height',str(args.height),'--output',str(args.output.resolve())]
    return supervise(command,args.output,timeout=120)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
