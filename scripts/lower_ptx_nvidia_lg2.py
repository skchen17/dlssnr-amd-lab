#!/usr/bin/env python3
"""Lower the two N0 Box-Muller lg2 sites to an empirical NVIDIA SM120 curve."""

from __future__ import annotations

import argparse, hashlib, json, re, struct
from pathlib import Path


LG2=re.compile(r"lg2\.approx\.ftz\.f32\s+(?P<dst>%r(?:218|222))\s*,\s*(?P<src>%r(?:181|205))\s*;")

def lower(text:str,model:dict)->tuple[str,int]:
    rows=model.get('coefficients_u32')
    segment_bits=model.get('segment_bits');segment_bits=segment_bits if isinstance(segment_bits,int) else (len(rows).bit_length()-1 if isinstance(rows,list)and len(rows)>0 else 0);local_bits=model.get('local_mantissa_bits',23-segment_bits);segments=1<<segment_bits
    degree=model.get('degree',len(rows[0])-1 if isinstance(rows,list)and rows else 0);coeff_count=degree+1
    if not isinstance(rows,list)or len(rows)!=segments or any(not isinstance(r,list)or len(r)!=coeff_count for r in rows):raise ValueError('model coefficient layout does not match segment_bits/degree')
    if local_bits!=23-segment_bits or not 4<=segment_bits<=10:raise ValueError('invalid model mantissa partition')
    local_mask=(1<<local_bits)-1;scale_bits=struct.unpack('<I',struct.pack('<f',2.0**-local_bits))[0]
    words=[value for row in rows for value in row];declaration=f'.const .align 4 .u32 __n0_nvidia_lg2_coeffs[{len(words)}] = {{\n  '+',\n  '.join(', '.join(f'0x{words[i+j]:08X}'for j in range(coeff_count))for i in range(0,len(words),coeff_count))+'\n};\n\n'
    marker='.visible .entry '
    position=text.find(marker)
    if position<0:raise ValueError('PTX entry declaration not found')
    text=text[:position]+declaration+text[position:]
    count=0
    def replace(m:re.Match[str])->str:
        nonlocal count
        i=count;count+=1;p=f'__n0_lg2_{i}';dst=m.group('dst');src=m.group('src')
        cb=[f'%{p}_c{k}b'for k in range(coeff_count)];cf=[f'%{p}_c{k}'for k in range(coeff_count)]
        lines=['{',f'.reg .b32 %{p}_bits, %{p}_expbits, %{p}_expint, %{p}_mant, %{p}_seg, %{p}_local, '+', '.join(cb)+';',f'.reg .f32 %{p}_t, %{p}_ef, %{p}_q, '+', '.join(cf)+';',f'.reg .b64 %{p}_base, %{p}_offset, %{p}_address;',f'mov.b32 %{p}_bits, {src};',f'shr.u32 %{p}_expbits, %{p}_bits, 23;',f'and.b32 %{p}_expbits, %{p}_expbits, 255;',f'add.s32 %{p}_expint, %{p}_expbits, -127;',f'cvt.rn.f32.s32 %{p}_ef, %{p}_expint;',f'and.b32 %{p}_mant, %{p}_bits, 8388607;',f'shr.u32 %{p}_seg, %{p}_mant, {local_bits};',f'and.b32 %{p}_local, %{p}_mant, {local_mask};',f'cvt.rn.f32.u32 %{p}_t, %{p}_local;',f'mul.rn.f32 %{p}_t, %{p}_t, 0f{scale_bits:08X};',f'mov.u64 %{p}_base, __n0_nvidia_lg2_coeffs;',f'mul.wide.u32 %{p}_offset, %{p}_seg, {coeff_count*4};',f'add.u64 %{p}_address, %{p}_base, %{p}_offset;']
        for k in range(coeff_count):lines.extend((f'ld.const.b32 {cb[k]}, [%{p}_address'+(f'+{k*4}'if k else '')+'];',f'mov.b32 {cf[k]}, {cb[k]};'))
        lines.append(f'mov.f32 %{p}_q, {cf[-1]};')
        for k in range(degree-1,-1,-1):lines.append(f'fma.rn.f32 %{p}_q, %{p}_q, %{p}_t, {cf[k]};')
        lines.extend((f'add.rn.f32 {dst}, %{p}_q, %{p}_ef;','}'));return '\n'.join(lines)
    return LG2.sub(replace,text),count

def main()->int:
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('report',type=Path);p.add_argument('--model',type=Path,required=True);a=p.parse_args();source=a.input.read_bytes();model=json.loads(a.model.read_text(encoding='utf-8'));text,count=lower(source.decode('utf-8'),model);output=text.encode('utf-8');passed=count==2
    report={'schema':1,'experiment':'ptx_n0_nvidia_lg2_curve_lowering','status':'PASS'if passed else'FAIL','classification':'RTX5070_PRODUCTION_DOMAIN_EMPIRICAL_SFU_LOWERING','source_sha256':hashlib.sha256(source).hexdigest().upper(),'output_sha256':hashlib.sha256(output).hexdigest().upper(),'model':str(a.model),'model_experiment':model.get('experiment'),'lg2_sites_lowered':count,'remaining_target_lg2_sites':len(LG2.findall(text))}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(output);a.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,separators=(',',':')));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
