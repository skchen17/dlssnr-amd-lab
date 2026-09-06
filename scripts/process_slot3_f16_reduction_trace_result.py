#!/usr/bin/env python3
"""Validate and compare the returned slot-3 packed-FP16 reduction trace."""
from __future__ import annotations
import argparse, hashlib, json, shutil, struct, sys, zipfile
from datetime import datetime
from pathlib import Path
try:
    from scripts.instrument_slot3_f16_reduction_trace import LANES, REGISTERS
    from scripts.process_slot3_mma_trace_result import archive_sha256, normalized_member
except ModuleNotFoundError:
    from instrument_slot3_f16_reduction_trace import LANES, REGISTERS
    from process_slot3_mma_trace_result import archive_sha256, normalized_member

TRACE_BYTES=LANES*len(REGISTERS)*4;EXPECTED_EXPERIMENT='rtx5070_slot3_f16_reduction_register_trace'
UNINSTRUMENTED_OUTPUT_SHA256='33FE600487C7CF89F8D8F238999D8E0C4602A865E33802CAB09C5D1DC50BD4F9'
def sha256(data:bytes)->str:return hashlib.sha256(data).hexdigest().upper()
def inspect_archive(zf):
    infos=zf.infolist()
    if not infos or len(infos)>32:raise ValueError(f'unexpected ZIP member count: {len(infos)}')
    if sum(x.file_size for x in infos)>256*1024*1024:raise ValueError('ZIP uncompressed payload exceeds safety limit')
    seen=set();files=[]
    for info in infos:
        path=normalized_member(info.filename);key=path.as_posix().casefold()
        if key in seen:raise ValueError(f'duplicate ZIP member rejected: {info.filename!r}')
        seen.add(key)
        if not info.is_dir():files.append((path,info))
    ms=[x for x in files if x[0].name=='manifest.json'];ts=[x for x in files if x[0].name=='f16_reduction_trace.raw']
    if len(ms)!=1 or len(ts)!=1:raise ValueError('archive must contain exactly one manifest and reduction trace')
    if ms[0][0].parent!=ts[0][0].parent or len(ms[0][0].parts)!=2:raise ValueError('manifest and trace must share one top-level directory')
    if ts[0][1].file_size!=TRACE_BYTES:raise ValueError(f'reduction trace must be {TRACE_BYTES} bytes')
    return ms[0][1],ts[0][1]
def require_manifest(m,t):
    c={'schema':m.get('schema')==1,'experiment':m.get('experiment')==EXPECTED_EXPERIMENT,'status':m.get('status')=='PASS','payload_integrity':m.get('payload_integrity') is True,'probe_exit':m.get('probe_exit')==0,'probe_pass':m.get('probe_pass') is True,'device_name':isinstance(m.get('device_name'),str) and 'NVIDIA' in m['device_name'].upper(),'cta':m.get('cta')==[2,0,0],'registers':m.get('registers')==REGISTERS,'kernel_launched':m.get('kernel_launched') is True,'checkpoint_bytes':m.get('checkpoint_bytes')==TRACE_BYTES,'checkpoint_nonzero_bytes':m.get('checkpoint_nonzero_bytes')==sum(x!=0 for x in t),'checkpoint_sha256':str(m.get('checkpoint_sha256','')).upper()==sha256(t)}
    failed=[k for k,v in c.items() if not v]
    if failed:raise ValueError('result manifest validation failed: '+', '.join(failed))
def compare(rtx,amd):
    if len(rtx)!=TRACE_BYTES or len(amd)!=TRACE_BYTES:raise ValueError(f'trace size must be {TRACE_BYTES}')
    rw=struct.unpack(f'<{TRACE_BYTES//4}I',rtx);aw=struct.unpack(f'<{TRACE_BYTES//4}I',amd);reports=[];first=None;total=0
    for i,r in enumerate(REGISTERS):
        examples=[]
        for lane in range(LANES):
            o=lane*len(REGISTERS)+i
            if rw[o]!=aw[o]:
                total+=1;item={'lane':lane,'rtx_u32':f'0x{rw[o]:08X}','amd_u32':f'0x{aw[o]:08X}'};examples.append(item)
                if first is None:first={'register':r,'register_index':i,**item}
        reports.append({'register':r,'mismatch_lanes':len(examples),'examples':examples[:8]})
    return {'schema':1,'experiment':'slot3_f16_reduction_rtx5070_vs_rx9070xt','status':'PASS','classification':'FIRST_PACKED_F16_REDUCTION_DIVERGENCE','rtx_sha256':sha256(rtx),'amd_sha256':sha256(amd),'bitwise_equal':total==0,'word_mismatches':total,'first_mismatch':first,'per_register':reports}
def process_archive(archive,amd_trace,output):
    archive=archive.resolve(strict=True);amd_trace=amd_trace.resolve(strict=True);amd=amd_trace.read_bytes()
    if len(amd)!=TRACE_BYTES:raise ValueError(f'AMD trace must be {TRACE_BYTES} bytes')
    if output.exists():raise FileExistsError(f'output already exists: {output}')
    with zipfile.ZipFile(archive) as zf:mi,ti=inspect_archive(zf);m=json.loads(zf.read(mi).decode('utf-8-sig'));rtx=zf.read(ti)
    require_manifest(m,rtx);output.mkdir(parents=True);(output/'rtx_f16_reduction_trace.raw').write_bytes(rtx);shutil.copyfile(amd_trace,output/'amd_f16_reduction_trace.raw');(output/'rtx_manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8');c=compare(rtx,amd)
    observed_output=str(m.get('output_sha256','')).upper();preserved=observed_output==UNINSTRUMENTED_OUTPUT_SHA256
    c['classification']='UNINSTRUMENTED_REDUCTION_ORACLE' if preserved else 'INSTRUMENTATION_PERTURBATION_OBSERVED';c['reference_output_preserved']=preserved;c['admissible_as_uninstrumented_oracle']=preserved;(output/'comparison.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    receipt={'schema':1,'experiment':'slot3_f16_reduction_trace_result_ingestion','status':'PASS','source_archive':str(archive),'source_archive_sha256':archive_sha256(archive),'rtx_device_name':m['device_name'],'cta':m['cta'],'rtx_trace_sha256':sha256(rtx),'amd_trace_sha256':sha256(amd),'bitwise_equal':c['bitwise_equal'],'word_mismatches':c['word_mismatches'],'first_mismatch':c['first_mismatch'],'uninstrumented_output_sha256':UNINSTRUMENTED_OUTPUT_SHA256,'instrumented_output_sha256':observed_output or None,'reference_output_preserved':preserved,'instrumentation_perturbed':not preserved,'admissible_as_uninstrumented_oracle':preserved};(output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt
def main():
    repo=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--amd-trace',type=Path,default=repo/'results/20260901_080000_slot3_cta2_f16_reduction_trace/amd/f16_reduction_trace.raw');p.add_argument('--output',type=Path);a=p.parse_args();out=a.output or repo/'results'/f'{datetime.now():%Y%m%d_%H%M%S}_slot3_f16_reduction_cross_vendor'
    try:r=process_archive(a.archive,a.amd_trace,out)
    except (FileNotFoundError,FileExistsError,ValueError,zipfile.BadZipFile,json.JSONDecodeError) as e:print(f'ERROR: {e}',file=sys.stderr);return 2
    print(json.dumps(r,indent=2));print(f'Result: {out.resolve()}');return 0
if __name__=='__main__':raise SystemExit(main())
