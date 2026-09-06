"""Summarize four bounded full-frame runs without hiding clock drift."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(directory: Path, label: str) -> dict:
    probe=json.loads((directory/'child.json').read_text(encoding='utf-8-sig'))
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8-sig'))
    if not probe.get('checks_pass') or not manifest.get('normal_exit'):
        raise ValueError(f'{label}: lifecycle did not pass')
    if probe.get('allocated_after_release_bytes') or probe.get('reserved_after_release_bytes'):
        raise ValueError(f'{label}: resources not released')
    warm=[float(row['gpu_event_raw_ms']) for row in probe['runs'] if row['warm']]
    if len(warm)!=2:raise ValueError(f'{label}: expected exactly two warm samples')
    hashes={row['output_sha256'] for row in probe['runs']}
    if len(hashes)!=1:raise ValueError(f'{label}: unstable output hash')
    return {'label':label,'directory':str(directory),'warm_gpu_event_ms':warm,
            'output_sha256':hashes.pop(),'peak_allocated_bytes':max(row['peak_allocated_bytes'] for row in probe['runs']),
            'matrix_modules':probe['matrix_modules'],'matrix_profile':probe['matrix_profile']}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs=4,required=True,metavar=('B1','A1','B2','A2'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    labels=('candidate_b1','reference_a1','candidate_b2','reference_a2')
    rows=[load(path,label) for path,label in zip(args.runs,labels)]
    hashes={row['output_sha256'] for row in rows}
    if len(hashes)!=1:raise ValueError('A/B outputs are not hash-identical')
    candidate=rows[0]['warm_gpu_event_ms']+rows[2]['warm_gpu_event_ms']
    reference=rows[1]['warm_gpu_event_ms']+rows[3]['warm_gpu_event_ms']
    cm,rm=statistics.median(candidate),statistics.median(reference)
    report={'schema':1,'status':'PASS','order':'B-A-B-A','timing_scope':'full-forward HIP event span, not kernel busy sum',
            'runs':rows,'candidate_warm_gpu_event_ms':candidate,'reference_warm_gpu_event_ms':reference,
            'candidate_median_ms':cm,'reference_median_ms':rm,'speedup':rm/cm,
            'reduction_percent':100*(rm-cm)/rm,'bitwise_hash_equal':True,'output_sha256':hashes.pop(),
            'rgp_counters_used':False,'four_k_used':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
