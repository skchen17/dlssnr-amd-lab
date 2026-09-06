"""Aggregate bounded module-fusion evidence, keeping hardware dispatch unknown."""
import argparse,hashlib,json,statistics
from pathlib import Path

def summarize(name,block):
    root=Path('results')/name
    r=json.loads((root/'child.json').read_bytes())
    if not r['checks_pass'] or not r['sources_unchanged'] or r['allocated_after_release_bytes'] or r['reserved_after_release_bytes']:
        raise ValueError(f'unaccepted run {name}')
    runs=r['runs']
    if len(runs)!=12 or len({x['output_sha256'] for x in runs})!=1:raise ValueError('twelve exact runs required')
    sha=hashlib.sha256((root/'output.rgba16f').read_bytes()).hexdigest().upper()
    if sha!=runs[0]['output_sha256']:raise ValueError('output hash changed')
    file='head_times.json' if block==70 else f'module{block}_times.json'
    m=json.loads((root/file).read_bytes())['runs']
    counts=json.loads((root/'aten_counts.json').read_bytes())
    return dict(evidence=str(root),frame_host_ms=statistics.median(x['host_forward_submit_wait_ms'] for x in runs[1:]),
        module_host_ms=statistics.median(x['host_submit_wait_ms'] for x in m),
        module_aten_invocations_NOT_dispatches=sum(counts['blocks'][str(block)].values()),
        module_peak_live_bytes=max(x['allocator_peak_live_bytes'] for x in m),
        frame_peak_live_bytes=max(x['peak_allocated_bytes'] for x in runs),
        frame_peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in runs),
        authored_frame_launches={k:v for k,v in runs[0].items() if 'launches_cumulative' in k},
        gpu_total_dispatches=None,pure_gpu_module_ms=None,output_sha256=sha,
        evidence_sha256=hashlib.sha256((root/'child.json').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    specs={'head_before':('20260906_head2160_baseline_twelve_v1',70),
        'head_after':('20260906_head2160_twelve_v2',70),
        'head_gather_experimental':('20260906_head_gather2160_twelve_v1',70),
        'pre_before':('20260906_head_only_precontrol2160_twelve_v1',0),
        'pre_after':('20260906_head_pre2160_twelve_v1',0),
        'c32_before':('20260906_head_c32control2160_twelve_v1',2),
        'c32_after':('20260906_head_c322160_twelve_v1',2)}
    rows={k:summarize(*v) for k,v in specs.items()}
    if len({v['output_sha256'] for v in rows.values()})!=1:raise ValueError('cross-mode output differs')
    result=dict(scope='4K SYNTHETIC FROZEN CANDIDATE, separate module passes, host submit/wait timings',
        rows=rows,all_outputs_bitwise_equal=True,maximum_output_error=0,
        complete_module_fusion=False,gpu_dispatch_trace_available=False)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
