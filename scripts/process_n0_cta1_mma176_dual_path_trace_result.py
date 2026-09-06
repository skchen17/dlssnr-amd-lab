#!/usr/bin/env python3
"""Ingest and compare the RTX CTA(1,0) MMA-176 dual-path trace."""

from __future__ import annotations
import argparse,json,shutil,sys,zipfile
from datetime import datetime
from pathlib import Path,PurePosixPath
try:
 from scripts.analyze_n0_norm_path_trace import analyze,sha256
 from scripts.instrument_n0_cta1_mma176_dual_path_trace import STAGES,TRACE_BYTES
 from scripts.process_slot3_mma_trace_result import archive_sha256,normalized_member
except ModuleNotFoundError:
 from analyze_n0_norm_path_trace import analyze,sha256
 from instrument_n0_cta1_mma176_dual_path_trace import STAGES,TRACE_BYTES
 from process_slot3_mma_trace_result import archive_sha256,normalized_member

def inspect(zf):
 infos=zf.infolist()
 if not infos or len(infos)>32 or sum(i.file_size for i in infos)>8*1024*1024: raise ValueError('invalid ZIP size/count')
 seen=set();files=[]
 for i in infos:
  p=normalized_member(i.filename);k=p.as_posix().casefold()
  if k in seen: raise ValueError('duplicate ZIP member')
  seen.add(k)
  if not i.is_dir():files.append((p,i))
 ms=[x for x in files if x[0].name=='manifest.json'];ts=[x for x in files if x[0].name=='dual_path_trace.raw']
 if len(ms)!=1 or len(ts)!=1 or ms[0][0].parent!=ts[0][0].parent or len(ms[0][0].parts)!=2: raise ValueError('manifest/trace layout invalid')
 if ts[0][1].file_size!=TRACE_BYTES:return (_ for _ in ()).throw(ValueError('trace size invalid'))
 return ms[0][1],ts[0][1]

def require(m,t):
 expected=[{'index':i,'name':n,'register':r} for i,(n,_,r) in enumerate(STAGES)]
 checks={'schema':m.get('schema')==1,'experiment':m.get('experiment')=='rtx_n0_cta1_mma176_dual_path_trace','status':m.get('status')=='PASS','integrity':m.get('payload_integrity')is True,'probe':m.get('probe_exit')==0 and m.get('probe_pass')is True,'baseline':m.get('baseline_probe_exit')==0 and m.get('baseline_probe_pass')is True,'device':isinstance(m.get('device_name'),str)and'NVIDIA'in m['device_name'].upper(),'launch':m.get('grid')==[80,48,1]and m.get('block')==[32,1,1]and m.get('target_cta')==[1,0,0]and m.get('kernel_launched')is True,'stages':m.get('stage_count')==len(STAGES)and m.get('stages')==expected,'bytes':m.get('checkpoint_bytes')==TRACE_BYTES,'nonzero':m.get('checkpoint_nonzero_bytes')==sum(x!=0 for x in t),'hash':str(m.get('checkpoint_sha256','')).upper()==sha256(t),'extra':m.get('scratch_extra_bytes')==TRACE_BYTES,'input':m.get('input_variant')=='zero_rgba16f_full_graph','preserved':m.get('reference_output_preserved')is True and m.get('instrumentation_perturbed')is False}
 failed=[k for k,v in checks.items() if not v]
 if failed:raise ValueError('manifest validation failed: '+', '.join(failed))

def process_archive(archive,amd_trace,output):
 archive=archive.resolve(strict=True);amd_trace=amd_trace.resolve(strict=True)
 if amd_trace.stat().st_size!=TRACE_BYTES:raise ValueError('AMD trace size invalid')
 if output.exists():raise FileExistsError(output)
 with zipfile.ZipFile(archive)as z:m_info,t_info=inspect(z);m=json.loads(z.read(m_info).decode('utf-8-sig'));t=z.read(t_info)
 require(m,t);output.mkdir(parents=True);rp=output/'rtx_dual_path_trace.raw';ap=output/'amd_dual_path_trace.raw';rp.write_bytes(t);shutil.copyfile(amd_trace,ap);(output/'rtx_manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
 c=analyze(rp,ap,stages_layout=STAGES,trace_bytes=TRACE_BYTES);c['experiment']='n0_cta1_mma176_dual_path_rtx_vs_rx9070xt';c['reference_output_preserved']=True;c['instrumentation_perturbed']=False;(output/'comparison.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
 receipt={'schema':1,'experiment':'n0_cta1_mma176_dual_path_result_ingestion','status':'PASS','source_archive':str(archive),'source_archive_sha256':archive_sha256(archive),'rtx_device_name':m['device_name'],'rtx_trace_sha256':sha256(t),'amd_trace_sha256':c['amd_sha256'],'reference_output_preserved':True,'instrumentation_perturbed':False,'first_divergent_stage':c['first_divergent_stage'],'first_divergent_stage_name':c['first_divergent_stage_name']};(output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt

def main():
 repo=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--amd-trace',type=Path,default=repo/'results/20260901_203000_n0_cta1_mma176_dual_path_trace/amd/dual_path_trace.raw');p.add_argument('--output',type=Path);a=p.parse_args();out=a.output or repo/'results'/f'{datetime.now():%Y%m%d_%H%M%S}_n0_cta1_mma176_dual_path_cross_vendor'
 try:r=process_archive(a.archive,a.amd_trace,out)
 except(FileNotFoundError,FileExistsError,ValueError,zipfile.BadZipFile,json.JSONDecodeError)as e:print(f'ERROR: {e}',file=sys.stderr);return 2
 print(json.dumps(r,indent=2));print(f'Result: {out.resolve()}');return 0
if __name__=='__main__':raise SystemExit(main())
