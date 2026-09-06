"""Summarize four independent 1080p Python/NRPlan benchmark processes."""
import argparse,json,statistics
from pathlib import Path


def load(root):
    root=Path(root);child=json.loads((root/'child.json').read_bytes())
    manifest=json.loads((root/'manifest.json').read_bytes())
    warm=[row for row in child['runs'] if row['warm']]
    if len(warm)<2:raise ValueError(f'{root}: at least two warm rows required')
    if not child['checks_pass'] or not manifest['pass'] or not manifest['normal_exit']:
        raise ValueError(f'{root}: failed or abnormal process')
    if child['allocated_after_release_bytes'] or child['reserved_after_release_bytes']:
        raise ValueError(f'{root}: resources not released')
    return child,{
        'directory':str(root),'host_median_ms':statistics.median(r['host_forward_submit_wait_ms'] for r in warm),
        'event_median_ms':statistics.median(r['gpu_stream_elapsed_ms'] for r in warm),
        'output_sha256':warm[0]['output_sha256'],'cpp_nr_plan':warm[0]['cpp_nr_plan'],
        'kernel_nodes':child.get('gpu_dispatch_count'),'graph_nodes':child.get('gpu_graph_nodes'),
        'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in warm),
        'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in warm),
        'device_used_bytes':max(r['device_used_bytes_sample'] for r in warm)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--a1',type=Path,required=True);p.add_argument('--b1',type=Path,required=True)
    p.add_argument('--b2',type=Path,required=True);p.add_argument('--a2',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();loaded=[load(x) for x in (a.a1,a.b1,a.b2,a.a2)];children=[x[0] for x in loaded];rows=[x[1] for x in loaded]
    if [r['cpp_nr_plan'] for r in rows]!=[False,True,True,False]:raise ValueError('required order is Python, NRPlan, NRPlan, Python')
    hashes={r['output_sha256'] for r in rows};inputs={json.dumps(c['input'],sort_keys=True) for c in children}
    if len(hashes)!=1 or len(inputs)!=1:raise ValueError('input or output mismatch')
    a_ms=statistics.mean((rows[0]['host_median_ms'],rows[3]['host_median_ms']))
    b_ms=statistics.mean((rows[1]['host_median_ms'],rows[2]['host_median_ms']))
    report={'schema':1,'checks_pass':True,'order':['A_PYTHON','B_NRPLAN','B_NRPLAN','A_PYTHON'],'rows':rows,
            'a_host_mean_of_medians_ms':a_ms,'b_host_mean_of_medians_ms':b_ms,
            'nrplan_host_change_percent':(b_ms/a_ms-1)*100,'output_sha256':rows[0]['output_sha256'],
            'event_scope_warning':'Python event spans may exclude host-side gaps/blocking; compare host submit/wait for the runtime A/B. NRPlan event spans are stream elapsed, not kernel-busy sums.',
            'gpu_profiler_used':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
