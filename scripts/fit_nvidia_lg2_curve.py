#!/usr/bin/env python3
"""Fit a segmented quadratic approximation to an RTX N0 lg2 trace."""

from __future__ import annotations

import argparse, json, struct
from pathlib import Path
import numpy as np


def u32(value:float)->int:return struct.unpack('<I',struct.pack('<f',value))[0]

def evaluate(cf:np.ndarray,seg:np.ndarray,tf:np.ndarray,exp:np.ndarray)->np.ndarray:
    c=cf[seg];q=c[:,-1].copy()
    for index in range(c.shape[1]-2,-1,-1):q=np.float32(np.float32(q*tf)+c[:,index])
    return np.float32(q+exp.astype('<f4'))

def fit(rtx_path:Path,amd_path:Path,segment_bits:int=6,degree:int=2)->dict:
    if not 4<=segment_bits<=10:raise ValueError('segment_bits must be in [4,10]')
    if not 2<=degree<=4:raise ValueError('degree must be in [2,4]')
    segments=1<<segment_bits;local_bits=23-segment_bits;local_mask=(1<<local_bits)-1
    r=np.fromfile(rtx_path,dtype='<u4').reshape(-1,2);a=np.fromfile(amd_path,dtype='<u4').reshape(-1,2)
    if r.shape!=(245760,2)or a.shape!=r.shape:raise ValueError('expected 245760 input/output pairs')
    if not np.array_equal(r[:,0],a[:,0]):raise ValueError('RTX/RX lg2 inputs are not bitwise identical')
    xb=r[:,0];yb=r[:,1];y=yb.view('<f4');exp=((xb>>23)&255).astype(np.int32)-127;mant=xb&0x7fffff;seg=(mant>>local_bits).astype(np.int32);t=(mant&local_mask).astype(np.float64)/(1<<local_bits);train=exp==-1
    coefficients=[]
    for s in range(segments):
        m=train&(seg==s);c=np.polynomial.polynomial.polyfit(t[m],y[m].astype(np.float64)+1,degree).astype('<f4');coefficients.append([u32(float(v))for v in c])
    cf=np.asarray([[struct.unpack('<f',struct.pack('<I',v))[0]for v in row]for row in coefficients],dtype='<f4');tf=t.astype('<f4')
    # Quantized coefficient fitting can land one or two ULP from a half-rounding
    # boundary. Tune each intercept locally against every observed exponent while
    # retaining squared FP32 error as the tie-breaker.
    for s in range(segments):
        m=seg==s;target_half=y[m].astype('<f2').view('<u2');base=cf[s].copy();bits=base[0].view('<u4').item();best=None
        for delta in range(-128,129):
            candidate=base.copy();candidate[0]=np.asarray(bits+delta,dtype='<u4').view('<f4')
            q=candidate[-1]
            for index in range(len(candidate)-2,-1,-1):q=np.float32(np.float32(q*tf[m])+candidate[index])
            p=np.float32(q+exp[m].astype('<f4'))
            score=(int(np.sum(p.astype('<f2').view('<u2')!=target_half)),float(np.sum((p.astype(np.float64)-y[m].astype(np.float64))**2)),abs(delta))
            if best is None or score<best[0]:best=(score,candidate[0])
        cf[s,0]=best[1]
    coefficients=[[u32(float(v))for v in row]for row in cf]
    pred=evaluate(cf,seg,tf,exp);pb=pred.view('<u4');half_mismatch=int(np.sum(pred.astype('<f2').view('<u2')!=y.astype('<f2').view('<u2')))
    return {'schema':1,'experiment':f'nvidia_sm120_lg2_{segments}_segment_degree{degree}_fit','status':'PASS','classification':'RTX5070_PRODUCTION_DOMAIN_EMPIRICAL_SFU_MODEL','segments':segments,'degree':degree,'segment_bits':segment_bits,'local_mantissa_bits':local_bits,'intercept_search_ulps':128,'training_exponent':-1,'training_samples':int(np.sum(train)),'validation_samples':len(y),'fp32_exact_samples':int(np.sum(pb==yb)),'fp32_exact_fraction':float(np.mean(pb==yb)),'fp16_mismatch_samples':half_mismatch,'coefficients_u32':coefficients}

def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--rtx',type=Path,required=True);p.add_argument('--amd',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--segment-bits',type=int,default=6);p.add_argument('--degree',type=int,default=2);a=p.parse_args();model=fit(a.rtx,a.amd,a.segment_bits,a.degree);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(model,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:model[k]for k in ('status','segments','degree','training_samples','validation_samples','fp32_exact_fraction','fp16_mismatch_samples')},separators=(',',':')));return 0

if __name__=='__main__':raise SystemExit(main())
