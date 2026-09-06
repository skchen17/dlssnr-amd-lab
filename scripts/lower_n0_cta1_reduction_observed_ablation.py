#!/usr/bin/env python3
"""Causal A/B ablations for observed CTA(1,0) reduction differences."""

from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path

A_TARGET=re.compile(r"\{add\.f16x2 %r2321,%r2266,%r2267;\s*\}")
B_TARGET=re.compile(r"\{add\.f16x2 %r2713,%r2656,%r2657;\s*\}")

def block(tag,register,lane_lo,lane_hi,old,new):
 p=f"__{tag}"
 return "\n".join(("{",f".reg .pred %{p}_px, %{p}_py, %{p}_lo, %{p}_hi, %{p}_cta, %{p}_lanes, %{p}_value, %{p}_apply;",f".reg .u32 %{p}_x, %{p}_y, %{p}_lane;",f"mov.u32 %{p}_x, %ctaid.x;",f"mov.u32 %{p}_y, %ctaid.y;",f"mov.u32 %{p}_lane, %laneid;",f"setp.eq.u32 %{p}_px, %{p}_x, 1;",f"setp.eq.u32 %{p}_py, %{p}_y, 0;",f"and.pred %{p}_cta, %{p}_px, %{p}_py;",f"setp.ge.u32 %{p}_lo, %{p}_lane, {lane_lo};",f"setp.le.u32 %{p}_hi, %{p}_lane, {lane_hi};",f"and.pred %{p}_lanes, %{p}_lo, %{p}_hi;",f"setp.eq.u32 %{p}_value, %{register}, 0x{old:08X};",f"and.pred %{p}_apply, %{p}_cta, %{p}_lanes;",f"and.pred %{p}_apply, %{p}_apply, %{p}_value;",f"@%{p}_apply mov.b32 %{register}, 0x{new:08X};","}"))

def lower(text,policy):
 if len(A_TARGET.findall(text))!=1 or len(B_TARGET.findall(text))!=1:raise ValueError('expected one A and B target')
 if policy in ('a','both'):
  suffix="\n"+"\n".join((block('r2321_g12','r2321',12,15,0x3D453D45,0x3D443D44),block('r2321_g16','r2321',16,19,0x3D323D32,0x3D313D31),block('r2321_g28','r2321',28,31,0x3DCC3DCC,0x3DCD3DCD)))
  text=A_TARGET.sub(lambda m:m.group(0)+suffix,text,count=1)
 if policy in ('b','both'):
  text=B_TARGET.sub(lambda m:m.group(0)+"\n"+block('r2713_g12','r2713',12,15,0x3C633C63,0x3C623C62),text,count=1)
 return text,(3 if policy=='a' else 1 if policy=='b' else 4)

def main():
 p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('report',type=Path);p.add_argument('--policy',choices=('a','b','both'),required=True);a=p.parse_args();source=a.input.read_bytes();out_text,count=lower(source.decode(),a.policy);out=out_text.encode();r={'schema':1,'experiment':'n0_cta1_observed_reduction_ablation','status':'PASS','classification':'CAUSAL_LOCALIZATION_NOT_GENERAL_LOWERING','counts_as_s7':False,'policy':a.policy,'source_sha256':hashlib.sha256(source).hexdigest().upper(),'output_sha256':hashlib.sha256(out).hexdigest().upper(),'correction_count':count,'scope':'CTA (1,0), observed lane groups and exact reduction words only'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(out);a.report.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,separators=(',',':')));return 0
if __name__=='__main__':raise SystemExit(main())
