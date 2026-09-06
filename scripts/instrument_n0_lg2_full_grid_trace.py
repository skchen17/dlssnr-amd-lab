#!/usr/bin/env python3
"""Capture the first Box-Muller lg2 input/output pair for every N0 sample."""

from __future__ import annotations

import argparse, hashlib, json, re
from pathlib import Path


GRID_X=80;GRID_Y=48;SAMPLES_PER_CTA=64;SAMPLES=GRID_X*GRID_Y*SAMPLES_PER_CTA
TRACE_OFFSET=384*640*32;BYTES_PER_SAMPLE=8;TRACE_BYTES=SAMPLES*BYTES_PER_SAMPLE
LG2=re.compile(r"lg2\.approx\.ftz\.f32\s+%r218\s*,\s*%r181\s*;")


def instrument(text:str)->tuple[str,int]:
    prefix='__n0_lg2_grid_trace'
    block='\n'.join([
        '{',f'.reg .b32 %{prefix}_ctax, %{prefix}_ctay, %{prefix}_linear, %{prefix}_sample, %{prefix}_offset;',
        f'.reg .b64 %{prefix}_scratch, %{prefix}_wide, %{prefix}_address;',
        f'mov.u32 %{prefix}_ctax, %ctaid.x;',f'mov.u32 %{prefix}_ctay, %ctaid.y;',
        f'mad.lo.u32 %{prefix}_linear, %{prefix}_ctay, {GRID_X}, %{prefix}_ctax;',
        f'mad.lo.u32 %{prefix}_sample, %{prefix}_linear, {SAMPLES_PER_CTA}, %r5556;',
        f'mad.lo.u32 %{prefix}_offset, %{prefix}_sample, {BYTES_PER_SAMPLE}, {TRACE_OFFSET};',
        f'ld.param.b64 %{prefix}_scratch, [%rd16+216];',f'cvt.u64.u32 %{prefix}_wide, %{prefix}_offset;',
        f'add.s64 %{prefix}_address, %{prefix}_scratch, %{prefix}_wide;',
        f'st.global.b32 [%{prefix}_address], %r181;',f'st.global.b32 [%{prefix}_address+4], %r218;','}'
    ])
    return LG2.subn(lambda m:m.group(0)+'\n'+block,text)


def main()->int:
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('report',type=Path);a=p.parse_args();source=a.input.read_bytes();text,count=instrument(source.decode('utf-8'));output=text.encode('utf-8');passed=count==1
    report={'schema':1,'experiment':'n0_lg2_full_grid_trace_instrumentation','status':'PASS'if passed else'FAIL','classification':'N0_NVIDIA_LG2_APPROXIMATION_CURVE_DATASET','source_sha256':hashlib.sha256(source).hexdigest().upper(),'output_sha256':hashlib.sha256(output).hexdigest().upper(),'grid':[GRID_X,GRID_Y,1],'block':[32,1,1],'sample_count':SAMPLES,'samples_per_cta':SAMPLES_PER_CTA,'bytes_per_sample':BYTES_PER_SAMPLE,'trace_offset':TRACE_OFFSET,'trace_bytes':TRACE_BYTES,'scratch_extra_bytes':TRACE_BYTES,'insertion_count':count}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(output);a.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,separators=(',',':')));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
