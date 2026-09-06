#!/usr/bin/env python3
"""Validate and ingest a returned RTX selected-CTA Box-Muller trace."""

from __future__ import annotations

import argparse, json, shutil, sys, zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.analyze_n0_box_muller_cta_trace import analyze_bytes, sha256
    from scripts.instrument_n0_box_muller_cta_trace import SAMPLES, STAGES as BASIC_STAGES
    from scripts.instrument_n0_box_muller_full_cta_trace import STAGES as FULL_STAGES
    from scripts.process_slot3_mma_trace_result import archive_sha256, normalized_member
except ModuleNotFoundError:
    from analyze_n0_box_muller_cta_trace import analyze_bytes, sha256
    from instrument_n0_box_muller_cta_trace import SAMPLES, STAGES as BASIC_STAGES
    from instrument_n0_box_muller_full_cta_trace import STAGES as FULL_STAGES
    from process_slot3_mma_trace_result import archive_sha256, normalized_member


def process(archive: Path, amd_path: Path, output: Path, target_cta: list[int]) -> dict:
    archive=archive.resolve(strict=True);amd=amd_path.resolve(strict=True).read_bytes()
    if output.exists(): raise FileExistsError(f"output already exists: {output}")
    with zipfile.ZipFile(archive) as zf:
        files=[(normalized_member(i.filename),i) for i in zf.infolist() if not i.is_dir()]
        ms=[i for p,i in files if p.name=='manifest.json'];ts=[i for p,i in files if p.name=='box_muller_trace.raw']
        if len(ms)!=1 or len(ts)!=1: raise ValueError('archive trace layout invalid')
        manifest=json.loads(zf.read(ms[0]).decode('utf-8-sig'));rtx=zf.read(ts[0])
    stage_names=manifest.get('stages')
    variants=[BASIC_STAGES,FULL_STAGES]
    stages=next((candidate for candidate in variants if stage_names==[name for name,_,_,_ in candidate]),None)
    if stages is None: raise ValueError('unsupported Box-Muller stage schema')
    trace_bytes=SAMPLES*len(stages)*4
    if len(amd)!=trace_bytes or len(rtx)!=trace_bytes: raise ValueError(f'trace size must be {trace_bytes} bytes')
    expected=[name for name,_,_,_ in stages]
    checks={'experiment':manifest.get('experiment')=='rtx_n0_selected_cta_box_muller_trace','status':manifest.get('status')=='PASS','integrity':manifest.get('payload_integrity')is True,'device':isinstance(manifest.get('device_name'),str)and'NVIDIA'in manifest['device_name'].upper(),'launch':manifest.get('target_cta')==target_cta and manifest.get('grid')==[80,48,1],'shape':manifest.get('sample_count')==SAMPLES and manifest.get('stage_count')==len(stages)and manifest.get('stages')==expected and manifest.get('checkpoint_bytes')==trace_bytes,'hash':str(manifest.get('checkpoint_sha256','')).upper()==sha256(rtx),'scope':manifest.get('classification')=='CONTROLLED_IMMEDIATE_POST_DEFINITION_APPROX_MATH_TRACE'}
    failed=[k for k,v in checks.items() if not v]
    if failed: raise ValueError('manifest validation failed: '+', '.join(failed))
    output.mkdir(parents=True);(output/'rtx_box_muller_trace.raw').write_bytes(rtx);shutil.copyfile(amd_path,output/'amd_box_muller_trace.raw');(output/'rtx_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    comparison=analyze_bytes(rtx,amd,stages);(output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='utf-8')
    receipt={'schema':1,'experiment':'n0_box_muller_cta_trace_result_ingestion','status':'PASS','source_archive':str(archive),'source_archive_sha256':archive_sha256(archive),'rtx_device_name':manifest['device_name'],'target_cta':target_cta,'first_mismatch_stage':comparison['first_mismatch_stage'],'reference_output_preserved':manifest.get('reference_output_preserved'),'admissible_scope':manifest.get('admissible_scope')}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt


def main()->int:
    repo=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--amd-trace',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--target-x',type=int,default=8);p.add_argument('--target-y',type=int,default=0);a=p.parse_args();out=a.output or repo/'results'/f'{datetime.now():%Y%m%d_%H%M%S}_n0_box_muller_cross_vendor'
    target_cta=[a.target_x,a.target_y,0]
    if not (0 <= a.target_x < 80 and 0 <= a.target_y < 48):
        p.error('target CTA must be inside the 80x48 N0 grid')
    try:r=process(a.archive,a.amd_trace,out,target_cta)
    except (OSError,ValueError,zipfile.BadZipFile,json.JSONDecodeError) as e:print(f'ERROR: {e}',file=sys.stderr);return 2
    print(json.dumps(r,indent=2));print(f'Result: {out.resolve()}');return 0


if __name__=='__main__':raise SystemExit(main())
