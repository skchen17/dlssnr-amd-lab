"""Summarize isolated stage and core graph diagnostics without extrapolated FPS."""
import argparse
import json
import math
import statistics
from pathlib import Path


def family(block):
    if not 0 <= block <= 70:
        raise ValueError('invalid stage')
    for name, spans in [('pre',[(0,0)]),('head',[(70,70)]),
                        ('c32',[(1,4),(66,69)]),('c64',[(5,8),(62,65)]),
                        ('c128',[(9,14),(56,61)]),('c256',[(15,22),(48,55)]),
                        ('c512',[(23,30),(40,47)]),('vit',[(31,38)]),
                        ('decoder_projection',[(39,39)])]:
        if any(lo <= block <= hi for lo,hi in spans):
            return name
    raise AssertionError(block)


def summarize(drained, probes):
    if drained.get('all_outputs_exact') is not True:
        raise ValueError('stage correctness gate missing')
    stages=[]
    for b in range(71):
        rows=[r for r in drained['runs'] if r['block']==b]
        if len(rows)!=3 or {r['repeat'] for r in rows}!={0,1,2}:
            raise ValueError('three distinct stage repeats required')
        values=[r['host_submit_wait_ms'] for r in rows]
        if not all(math.isfinite(v) and v>0 for v in values):
            raise ValueError('invalid stage time')
        stages.append({'block':b,'family':family(b),'median_isolated_ms':statistics.median(values)})
    cores=[]
    for p in probes:
        if p.get('checks_pass') is not True or p.get('allocated_after_release_bytes')!=0:
            raise ValueError('core correctness/lifecycle gate missing')
        for row in p['rows']:
            if not row.get('changed_input_exact'):
                raise ValueError('changed-input gate missing')
            eager,graph=(row['bulk'][mode] for mode in ('eager','graph'))
            if min(eager['calls'],graph['calls'])<12:
                raise ValueError('twelve calls required for performance summary')
            times=[eager['host_per_call_ms'],graph['host_per_call_ms']]
            if not all(math.isfinite(t) and t>0 for t in times):
                raise ValueError('invalid core time')
            cores.append({'family':p['family'],'batch':row['batch'],
                'eager_ms':times[0],'graph_ms':times[1],'observed_core_ratio':times[0]/times[1],
                'kernel_nodes_unchanged_by_replay':row['kernel_nodes'],
                'tenfold_in_this_microprobe_only':times[0]/times[1]>=10,
                'full_stage_speedup_verified':False})
    return {'stages':sorted(stages,key=lambda r:r['median_isolated_ms'],reverse=True),'cores':cores,
            'scope':'Drained stage times are not additive forward time. Synthetic core graph ratios exclude image adapters and are not fusion speedups.',
            'gpu_busy_time_measured':False,'whole_frame_speedup_verified':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--drained',type=Path,required=True)
    p.add_argument('--cores',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=summarize(json.loads(a.drained.read_bytes()),[json.loads(f.read_bytes()) for f in a.cores])
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
