"""Explicit small-device gate. Never imported or enabled by model execution.

On timeout/error stop; GPU submission cannot be cancelled by CPU timeout.
Do not retry automatically. Run in an independently supervised process.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import torch
from native_fusion_quantize import QuantizeE4,CubicQuantizeE4
from native_grouped_ffn import cubic_silu
from native_swin_torch import quantize_e4


def wait(event):
    deadline=time.monotonic()+10
    while not event.query():
        if time.monotonic()>deadline: raise TimeoutError('GPU deadline exceeded; do not retry or assume cancellation')
        time.sleep(.005)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dll',required=True)
    p.add_argument('--iterations',type=int,choices=(1,12),default=1)
    p.add_argument('--operation',choices=('quantize','cubic_quantize'),default='quantize')
    a=p.parse_args()
    if not torch.version.hip: raise RuntimeError('ROCm required')
    op=(CubicQuantizeE4 if a.operation=='cubic_quantize' else QuantizeE4)(a.dll)
    bits=torch.arange(65536,dtype=torch.int32).to(torch.uint16)
    source=bits.view(torch.float16).cuda()
    reference=quantize_e4(cubic_silu(source)) if a.operation=='cubic_quantize' else quantize_e4(source)
    end=torch.cuda.Event();end.record();wait(end)
    records=[]
    for i in range(a.iterations):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();output=op(source);end.record();wait(end)
        actual=output.cpu();expected=reference.cpu();valid=~torch.isnan(expected)
        assert torch.equal(actual.view(torch.int16)[valid],expected.view(torch.int16)[valid])
        assert torch.isnan(actual[~valid]).all()
        records.append({'iteration':i,'gpu_event_ms':start.elapsed_time(end),'explicit_fused_dispatches':1,'bit_mismatches':0})
    print(json.dumps({'pass':True,'operation':a.operation,'device':torch.cuda.get_device_name(),'records':records,'scope':'isolated_exhaustive_FP16_conversion_not_network_acceptance'}))


if __name__=='__main__':main()
