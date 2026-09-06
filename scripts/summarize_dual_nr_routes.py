"""Aggregate hash-bound offline component evidence; never label it game/e2e quality."""
import argparse,hashlib,json,statistics
from pathlib import Path


def load(path):return json.loads(path.read_bytes())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def nr(root):
    r=load(root/'child.json')
    if not r['checks_pass'] or not r['sources_unchanged'] or r['allocated_after_release_bytes'] or r['reserved_after_release_bytes']:
        raise ValueError('unaccepted NR evidence')
    if len(r['runs'])!=12 or len({x['output_sha256'] for x in r['runs']})!=1:raise ValueError('12 exact NR repetitions required')
    actual=digest(root/'output.rgba16f')
    if actual!=r['runs'][0]['output_sha256']:raise ValueError('NR output changed')
    hot=r['runs'][1:];events=[x['gpu_stream_elapsed_ms'] for x in hot]
    return {'evidence':str(root),'host_warm_median_ms':statistics.median(x['host_forward_submit_wait_ms'] for x in hot),
            'gpu_stream_median_ms':statistics.median(events) if all(x is not None for x in events) else None,
            'allocator_peak_live_bytes':max(x['peak_allocated_bytes'] for x in r['runs']),
            'allocator_peak_reserved_bytes':max(x['peak_reserved_bytes'] for x in r['runs']),
            'device_used_peak_sample_bytes':max(x['device_used_bytes_sample'] for x in r['runs']),
            'output_sha256':actual,'input_sha256':r['input']['sha256'],
            'first_frame_authored_fusion_launches':r['runs'][0]['authored_fusions_cumulative'],
            'gpu_total_dispatches':None,'gpu_busy_kernel_sum_ms':None}


def main(output):
    rows={};sources={}
    for size in ('1080','1440','2160'):
        before=nr(Path(f'results/20260906_nr_resolution{size}_twelve_v2'))
        after=nr(Path(f'results/20260906_nr_fused{size}_twelve_v2'))
        if before['output_sha256']!=after['output_sha256'] or before['input_sha256']!=after['input_sha256']:raise ValueError('fusion changed reference')
        rows[size]={'before':before,'after':after,'same_input_output_exact':True}
        rows[size]['latency_reduction_percent']=100*(1-after['host_warm_median_ms']/before['host_warm_median_ms'])
        if size!='2160':
            root=Path(f'results/20260906_nr_fsr{size}_twelve_v1');r=load(root/'report.json');p=load(root/'provenance.json')
            if not r['pass'] or r['provider']!='3.1.0' or p['exit_code'] or p['input_sha256'].upper()!=after['output_sha256']:
                raise ValueError('unaccepted or mismatched FSR component')
            if p['output_sha256']!=digest(root/'output.rgba16f'):raise ValueError('FSR output changed')
            fsr=statistics.median(r['gpu_ms_including_warmup'][1:])
            rows[size]['fsr']={'gpu_dispatch_interval_median_ms':fsr,'sampled_process_local_bytes':r['process_local_usage_peak_sampled'],
                'reset_every_frame':r['reset_every_frame'],'persistent_context':r['persistent_context'],'output_sha256':p['output_sha256']}
            rows[size]['component_gpu_interval_sum_ms_NOT_e2e']=after['gpu_stream_median_ms']+fsr if after['gpu_stream_median_ms'] is not None else None
        for kind in ('resolution','fused'):
            path=Path(f'results/20260906_nr_{kind}{size}_twelve_v2/child.json');sources[str(path)]=digest(path)
    counts={}
    for kind in ('resolution','fused'):
        c=load(Path(f'results/20260906_nr_{kind}2160_v1/aten_counts.json'))
        counts[kind]={'aten_invocations':c['total'],'not_gpu_dispatches':True}
    stages={}
    for kind in ('resolution','fused'):
        label='baseline' if kind=='resolution' else 'fused'
        path=Path(f'results/20260906_nr_stage_clock_{label}_v3/stage_times.json')
        r=load(path);sources[str(path)]=digest(path)
        groups={}
        for name,lo,hi in [('pre',0,0),('encoder_C32_C64',1,8),('decoder_C64_C32',62,69),('head',70,70),('vit',31,38)]:
            groups[name]=sum(x['host_stage_submit_ms'] for x in r['stages'] if lo<=x['block']<=hi)
        stages[kind]={'host_submit_groups_ms_NOT_gpu_busy':groups,
            'stage_clock_valid':r['valid'],'partition_coherent':r['partition_coherent'],
            'minimum_inter_stage_gap_ms':min(x['gpu_stream_gap_ms'] for x in r['inter_stage_gaps']),
            'accepted_stage_gpu_comparison':False}
    quality=load(Path('results/20260906_nr_fsr_static_comparison_v1/report.json'))
    for key,item in quality['sources'].items():
        if digest(Path(item['path'])).lower()!=item['sha256']:raise ValueError('quality input changed')
    result={'scope':'SYNTHETIC_STATIC_COMPONENT_BENCHMARK_NOT_GAME_END_TO_END',
        'timing_caveat':'NR GPU event spans include stream idle gaps; FSR is D3D12 timestamp dispatch interval. No kernel busy sum or complete dispatch trace available.',
        'memory_caveat':'NR and FSR measured in separate processes; simultaneous pipeline peak not measured. No summing/maxing these as observed combined VRAM.',
        'rows':rows,'aten_counts_4k':counts,'stage_clock_audit':stages,
        'static_candidate_agreement':quality['comparisons'],'true_game_temporal_quality_accepted':False,
        'full_nvidia_granularity_fusion_complete':False,'sources':sources}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);main(p.parse_args().output)
