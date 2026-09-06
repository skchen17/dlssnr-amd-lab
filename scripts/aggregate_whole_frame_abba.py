"""Aggregate two-run A-B-B-A whole-frame results with exact-output gating."""
import argparse,json,math,statistics
from pathlib import Path


def aggregate(groups):
    output={}
    for size,paths in groups.items():
        if len(paths)!=4:raise ValueError('whole-frame group must be A-B-B-A')
        labels=('reference','native','native','reference');samples={'reference':[],'native':[]};hashes=set();rows=[]
        for label,path in zip(labels,paths):
            report=json.loads((path/'child.json').read_text(encoding='utf-8'))
            if report.get('checks_pass') is not True or report.get('allocated_after_release_bytes')!=0 or report.get('reserved_after_release_bytes')!=0:
                raise ValueError(f'failed whole-frame run {path}')
            warm=[row['host_forward_submit_wait_ms'] for row in report['runs'] if row['warm']]
            if len(warm)!=2 or not all(math.isfinite(v) and v>0 for v in warm):raise ValueError('invalid warm sample set')
            run_hashes={row['output_sha256'] for row in report['runs']}
            if len(run_hashes)!=1:raise ValueError('output changed within process')
            hashes|=run_hashes;samples[label].extend(warm)
            rows.append({'implementation':label,'source':str(path).replace('\\','/'),'warm_samples_ms':warm,
                'peak_allocated_bytes':max(row['peak_allocated_bytes'] for row in report['runs']),
                'peak_reserved_bytes':max(row['peak_reserved_bytes'] for row in report['runs']),
                'device_used_bytes_sample':max(row['device_used_bytes_sample'] for row in report['runs'])})
        if len(hashes)!=1:raise ValueError('reference/native whole-frame hashes differ')
        reference=statistics.median(samples['reference']);native=statistics.median(samples['native'])
        output[size]={'reference_median_ms':reference,'native_median_ms':native,'saved_ms':reference-native,
            'speedup':reference/native,'percent_reduction':100*(reference-native)/reference,
            'samples':samples,'rows':rows,'output_sha256':next(iter(hashes)),'bitwise_exact':True}
    return {'schema':1,'scope':'Two independent warm-frame samples per process in A-B-B-A process order; host submit/wait, not kernel busy sum.',
        'groups':output,'all_outputs_bitwise_exact':True,'rgp_used':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for size in ('1080p','1440p'):p.add_argument(f'--{size}',type=Path,nargs=4,required=True,metavar=('A','B','B','A'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();report=aggregate({'1080p':getattr(a,'1080p'),'1440p':getattr(a,'1440p')})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
