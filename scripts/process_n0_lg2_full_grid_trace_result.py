#!/usr/bin/env python3
"""Validate and ingest the RTX full-grid N0 lg2 approximation dataset."""

from __future__ import annotations

import argparse, hashlib, json, shutil, sys, zipfile
from datetime import datetime
from pathlib import Path

try:
    from scripts.instrument_n0_lg2_full_grid_trace import SAMPLES, TRACE_BYTES
    from scripts.process_slot3_mma_trace_result import archive_sha256, normalized_member
except ModuleNotFoundError:
    from instrument_n0_lg2_full_grid_trace import SAMPLES, TRACE_BYTES
    from process_slot3_mma_trace_result import archive_sha256, normalized_member


def sha256(data:bytes)->str:return hashlib.sha256(data).hexdigest().upper()


def process(archive:Path,amd_path:Path,output:Path)->dict:
    archive=archive.resolve(strict=True);amd_path=amd_path.resolve(strict=True);amd=amd_path.read_bytes()
    if len(amd)!=TRACE_BYTES:raise ValueError(f'AMD trace must be {TRACE_BYTES} bytes')
    if output.exists():raise FileExistsError(f'output already exists: {output}')
    with zipfile.ZipFile(archive)as zf:
        files=[(normalized_member(i.filename),i)for i in zf.infolist()if not i.is_dir()]
        ms=[i for p,i in files if p.name=='manifest.json'];ts=[i for p,i in files if p.name=='lg2_trace.raw']
        if len(ms)!=1 or len(ts)!=1 or ts[0].file_size!=TRACE_BYTES:raise ValueError('archive trace layout invalid')
        manifest=json.loads(zf.read(ms[0]).decode('utf-8-sig'));rtx=zf.read(ts[0])
    checks={'experiment':manifest.get('experiment')=='rtx_n0_lg2_full_grid_trace','status':manifest.get('status')=='PASS','integrity':manifest.get('payload_integrity')is True,'device':isinstance(manifest.get('device_name'),str)and'NVIDIA'in manifest['device_name'].upper(),'launch':manifest.get('grid')==[80,48,1]and manifest.get('block')==[32,1,1]and manifest.get('kernel_launched')is True,'shape':manifest.get('sample_count')==SAMPLES and manifest.get('bytes_per_sample')==8 and manifest.get('checkpoint_bytes')==TRACE_BYTES,'hash':str(manifest.get('checkpoint_sha256','')).upper()==sha256(rtx),'preserved':manifest.get('reference_output_preserved')is True and manifest.get('instrumentation_perturbed')is False}
    failed=[k for k,v in checks.items()if not v]
    if failed:raise ValueError('manifest validation failed: '+', '.join(failed))
    input_mismatch=output_mismatch=0;first_output=None
    for index in range(SAMPLES):
        offset=index*8
        if rtx[offset:offset+4]!=amd[offset:offset+4]:input_mismatch+=1
        if rtx[offset+4:offset+8]!=amd[offset+4:offset+8]:
            output_mismatch+=1
            if first_output is None:first_output=index
    output.mkdir(parents=True);(output/'rtx_lg2_trace.raw').write_bytes(rtx);shutil.copyfile(amd_path,output/'amd_lg2_trace.raw');(output/'rtx_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    comparison={'schema':1,'experiment':'n0_lg2_full_grid_rtx_vs_rx9070xt','sample_count':SAMPLES,'input_mismatch_samples':input_mismatch,'output_mismatch_samples':output_mismatch,'output_exact_fraction':1-output_mismatch/SAMPLES,'first_output_mismatch_sample':first_output,'rtx_sha256':sha256(rtx),'amd_sha256':sha256(amd)};(output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='utf-8')
    receipt={'schema':1,'experiment':'n0_lg2_full_grid_trace_result_ingestion','status':'PASS','source_archive':str(archive),'source_archive_sha256':archive_sha256(archive),'rtx_device_name':manifest['device_name'],'reference_output_preserved':True,**comparison};(output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt


def main()->int:
    repo=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--amd-trace',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();out=a.output or repo/'results'/f'{datetime.now():%Y%m%d_%H%M%S}_n0_lg2_full_grid_cross_vendor'
    try:r=process(a.archive,a.amd_trace,out)
    except(OSError,ValueError,zipfile.BadZipFile,json.JSONDecodeError)as e:print(f'ERROR: {e}',file=sys.stderr);return 2
    print(json.dumps(r,indent=2));print(f'Result: {out.resolve()}');return 0


if __name__=='__main__':raise SystemExit(main())
