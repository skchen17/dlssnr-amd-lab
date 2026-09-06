"""Bounded gfx1201 gate for the C++-owned captured-kernel graph bridge."""
from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import time
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from validate_rocm_lifecycle import save,supervise


def digest(value):return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest().upper()


def child(args):
    import torch
    from native_cpp_nr_plan import CapturedNRPlan
    def phase(name):
        with (args.output/'phases.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'phase':name,'pid':os.getpid()})+'\n')
    atexit.register(phase,'python_atexit');phase('child_enter')
    if not torch.version.hip or not torch.cuda.is_available() or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('gfx1201 ROCm required')
    def wait(stream):
        event=torch.cuda.Event()
        with torch.cuda.stream(stream):event.record()
        deadline=time.monotonic()+10
        while not event.query():
            if time.monotonic()>deadline:raise TimeoutError('GPU uncertain; no cancellation or retry')
            time.sleep(.001)
    torch.manual_seed(71);source=torch.randn((32,48,4),device='cuda',dtype=torch.float16)
    def forward(value):return ((value.float()*1.5+0.25).half().roll(1,1)).contiguous()
    expected=forward(source);wait(torch.cuda.current_stream())
    plan=CapturedNRPlan(args.dll,forward,source,wait=wait)
    first=plan.submit(source);wait(plan.stream)
    exact1=bool(torch.equal(first,expected));hash1=digest(first)
    changed=source.clone().add_(.03125);expected2=forward(changed);wait(torch.cuda.current_stream())
    second=torch.empty_like(first);plan.submit(changed,second,resource_generation=2);wait(plan.stream)
    exact2=bool(torch.equal(second,expected2));hash2=digest(second)
    sequence=plan.sequence;plan.close(wait);del plan,source,changed,expected,expected2,first,second
    phase('gpu_work_complete');gc.collect();torch.cuda.empty_cache()
    if hasattr(torch._C,'_cuda_clearCublasWorkspaces'):torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
    result={'checks_pass':exact1 and exact2 and allocated==reserved==0,'classification':'CPP_OWNED_CAPTURED_KERNEL_GRAPH',
            'graph_replays':sequence,'first_exact':exact1,'changed_input_exact':exact2,'first_sha256':hash1,
            'changed_sha256':hash2,'python_operations_per_submit':0,'dynamic_allocation_per_submit':False,
            'allocated_after_release_bytes':allocated,'reserved_after_release_bytes':reserved}
    phase('resources_released');save(args.output/'child.json',result);return result['checks_pass']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dll',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--child',action='store_true');args=parser.parse_args()
    require_gpu_tests_enabled('C++ NRPlan captured graph gate')
    if args.child:return child(args)
    args.output.mkdir(parents=True,exist_ok=False)
    return supervise([os.fspath(Path(os.sys.executable)),os.fspath(Path(__file__).resolve()),'--child','--dll',
                      os.fspath(args.dll.resolve()),'--output',os.fspath(args.output.resolve())],args.output,timeout=60)


if __name__=='__main__':raise SystemExit(0 if main() else 2)
