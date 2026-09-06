"""Strict two-series module/frame promotion gate, never game promotion."""
import argparse,json,statistics
from pathlib import Path


def evaluate(rows):
    if set(rows)!={'a1','b1','b2','a2','a3','b3','b4','a4'}:raise ValueError('two ABBA series required')
    if len({r['output_sha256'] for r in rows.values()})!=1:raise ValueError('strict output gate failed')
    series=[]
    for labels in [('a1','b1','b2','a2'),('a3','b3','b4','a4')]:
        averages={g:{k:statistics.mean(rows[t][k] for t in labels if t.startswith(g))
                     for k in ('frame_ms','module_ms')} for g in ('a','b')}
        reduction=1-averages['b']['module_ms']/averages['a']['module_ms']
        regression=averages['b']['frame_ms']/averages['a']['frame_ms']-1
        series.append({'labels':labels,'averages':averages,'module_reduction':reduction,'frame_regression':regression,
                       'performance_gate':reduction>=.05 and regression<=.02})
    return {'rows':rows,'series':series,'two_series_performance_gate':all(s['performance_gate'] for s in series),
        'strict_same_fixture_output':True,'game_promoted':False,'all_geometry_correctness_gate':False,
        'pure_gpu_busy_ms':None,'scope':'4K fixed synthetic fixture; host submit/wait only. Performance gate alone cannot promote a candidate.'}


def read(prefix,label,module_file='head_times.json'):
    root=Path(f'{prefix}_{label}');data=json.loads((root/'child.json').read_bytes())
    manifest=json.loads((root/'manifest.json').read_bytes())
    if not manifest['pass'] or not data['checks_pass'] or not data['sources_unchanged']:raise ValueError('failed lifecycle')
    runs=data['runs']
    if len(runs)!=12 or len({r['output_sha256'] for r in runs})!=1:raise ValueError('twelve repeated outputs required')
    if any(r['peak_reserved_bytes']>5000000000 or r['device_used_bytes_sample']>6000000000 for r in runs):raise ValueError('resource gate')
    module=json.loads((root/module_file).read_bytes())['runs']
    return {'frame_ms':statistics.median(r['host_forward_submit_wait_ms'] for r in runs[1:]),
            'module_ms':statistics.median(r['host_submit_wait_ms'] for r in module),
            'output_sha256':runs[0]['output_sha256'],
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in runs),
            'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in runs)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',required=True);p.add_argument('--module-file',default='head_times.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=evaluate({s:read(a.prefix,s,a.module_file) for s in ('a1','b1','b2','a2','a3','b3','b4','a4')})
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result['series'],indent=2))
