"""Head cache/epilogue A-B-B-A evidence; timestamps are not GPU busy totals."""
import argparse,hashlib,json,statistics
from pathlib import Path

def load(path):return json.loads(path.read_bytes())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest().upper()

def sample(label,prefix):
    root=Path(f'{prefix}_{label}')
    r=load(root/'child.json');manifest=load(root/'manifest.json')
    if not manifest['pass'] or not r['checks_pass'] or not r['sources_unchanged']:
        raise ValueError(f'failed evidence {label}')
    runs=r['runs'];sha=digest(root/'output.rgba16f')
    if len(runs)!=12 or any(x['output_sha256']!=sha for x in runs):raise ValueError('twelve exact outputs required')
    head=load(root/'head_times.json')['runs'];counts=load(root/'aten_counts.json')['blocks']['70']
    return {'label':label,'frame_host_ms':statistics.median(x['host_forward_submit_wait_ms'] for x in runs[1:]),
        'head_host_ms':statistics.median(x['host_submit_wait_ms'] for x in head),
        'head_aten_calls_NOT_dispatches':sum(counts.values()),
        'reserved_bytes':max(x['peak_reserved_bytes'] for x in runs),
        'frame_peak_live_bytes':max(x['peak_allocated_bytes'] for x in runs),
        'head_peak_live_bytes':max(x['allocator_peak_live_bytes'] for x in head),
        'authored_epilogue_launches_per_frame':runs[1]['head_epilogue_launches_cumulative']-runs[0]['head_epilogue_launches_cumulative'],
        'authored_qkv_launches_per_frame':runs[1].get('head_qkv_launches_cumulative',0)-runs[0].get('head_qkv_launches_cumulative',0),
        'authored_quantize_launches_per_frame':runs[1]['authored_fusions_cumulative']['quantize_launches']-runs[0]['authored_fusions_cumulative']['quantize_launches'],
        'output_sha256':sha,'source_report_sha256':digest(root/'child.json')}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix',default='results/20260906_head_epilogue_abba')
    p.add_argument('--scope',default='4K synthetic; A=head input/compose; B=A plus frozen Head weight cache and residual epilogues');a=p.parse_args()
    rows=[sample(label,a.prefix) for label in ('a1','b1','b2','a2')]
    if len({r['output_sha256'] for r in rows})!=1:raise ValueError('A/B output differs')
    averages={}
    for group in ('a','b'):
        selected=[r for r in rows if r['label'].startswith(group)]
        averages[group]={key:statistics.mean(r[key] for r in selected) for key in ('frame_host_ms','head_host_ms')}
    result={'scope':a.scope,
        'runs':rows,'mean_of_run_medians':averages,'maximum_output_error':0,
        'frame_latency_reduction_percent':100*(1-averages['b']['frame_host_ms']/averages['a']['frame_host_ms']),
        'head_latency_reduction_percent':100*(1-averages['b']['head_host_ms']/averages['a']['head_host_ms']),
        'gpu_total_dispatches':None,'pure_gpu_busy_ms':None,
        'limitations':'Only four sequential process runs; does not establish cross-scene performance or eliminate clock/load drift.'}
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
