"""Aggregate independent transition A-B-B-A processes without hiding failures."""
import argparse,json,math,re,statistics
from collections import defaultdict
from pathlib import Path


def aggregate(paths):
    groups=defaultdict(list)
    for path in paths:
        report=json.loads((path/'child.json').read_text(encoding='utf-8'))
        match=re.match(r'^(\S+) (encoder|decoder) transition A-B-B-A',report.get('scope',''))
        if not match or report.get('checks_pass') is not True or report.get('allocated_after_release_bytes')!=0 or report.get('reserved_after_release_bytes')!=0:
            raise ValueError(f'invalid or failed benchmark {path}')
        groups[match.groups()].append((path,report))
    output={}
    for (size,kind),runs in sorted(groups.items()):
        if len(runs)!=3:raise ValueError(f'{size} {kind} needs exactly three independent processes')
        by_transition=defaultdict(lambda:{'reference':[],'native':[],'reference_peaks':[],'native_peaks':[],'process_medians':[]})
        for path,report in runs:
            for row in report['rows']:
                if not all(value['bitwise_exact'] for value in row['correctness'].values()):raise ValueError(f'non-exact output in {path}')
                item=by_transition[row['transition']]
                medians={}
                for implementation in ('reference','native'):
                    samples=[value for phase in row['phases'] if phase['implementation']==implementation for value in phase['gpu_event_ms']]
                    if len(samples)!=4 or not all(math.isfinite(v) and v>0 for v in samples):raise ValueError('invalid A-B-B-A sample set')
                    peaks=[phase['incremental_peak_allocated_bytes'] for phase in row['phases'] if phase['implementation']==implementation]
                    if len(peaks)!=4 or not all(type(v) is int and v>=0 for v in peaks):raise ValueError('invalid allocator peak set')
                    item[implementation].extend(samples);item[f'{implementation}_peaks'].extend(peaks);medians[implementation]=statistics.median(samples)
                item['process_medians'].append({'source':str(path).replace('\\','/'),**medians})
        rows=[]
        for transition,item in by_transition.items():
            if len(item['reference'])!=12 or len(item['native'])!=12:raise ValueError('expected 12 samples per implementation')
            reference=statistics.median(item['reference']);native=statistics.median(item['native'])
            rows.append({'transition':transition,'reference_median_ms':reference,'native_median_ms':native,
                'speedup':reference/native,'saved_ms':reference-native,'reference_samples':item['reference'],
                'native_samples':item['native'],'reference_max_incremental_peak_bytes':max(item['reference_peaks']),
                'native_max_incremental_peak_bytes':max(item['native_peaks']),
                'process_medians':item['process_medians'],'bitwise_exact':True})
        total_reference=sum(row['reference_median_ms'] for row in rows);total_native=sum(row['native_median_ms'] for row in rows)
        output[f'{size}_{kind}']={'processes':3,'samples_per_implementation_per_transition':12,'rows':rows,
            'sum_of_transition_medians_reference_ms':total_reference,'sum_of_transition_medians_native_ms':total_native,
            'sum_speedup':total_reference/total_native,'sum_saved_ms':total_reference-total_native,
            'sum_scope':'Sum of isolated transition medians; not whole-frame time or kernel busy sum.'}
    return {'schema':1,'groups':output,'all_outputs_bitwise_exact':True,'rgp_used':False,
        'scope':'Three independent A-B-B-A processes; 12 HIP-event samples per implementation and transition.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report=aggregate(a.inputs)
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
