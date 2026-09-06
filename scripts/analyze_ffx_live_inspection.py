"""Validate bounded live COM descriptors and queue correlation, not GPU pixels/state."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def analyze_events(events, expected):
    if type(expected) is not int or not 1 <= expected <= 32:
        raise ValueError('invalid expected sample count')
    failures = []
    resources = [e for e in events if e.get('event') == 'resources']
    returns = [e for e in events if e.get('event') == 'dispatch_return']
    submits = [e for e in events if e.get('event') == 'execute']
    ids = list(range(1, expected + 1))
    if sorted(e.get('sample_id', -1) for e in resources) != ids:
        failures.append('missing/duplicate resource samples')
    if sorted(e.get('sample_id', -1) for e in returns) != ids or any(e.get('status') != 0 for e in returns):
        failures.append('missing/duplicate/failed dispatch returns')
    if any(e.get('event') in ('inspection_failed', 'protection_restore_failed') for e in events):
        failures.append('native inspection/protection failure')
    hooks = [e for e in events if e.get('event') == 'queue_hook']
    if len(hooks) != 1 or hooks[0].get('installed') is not True:
        failures.append('queue observation not installed')
    matches = Counter(i for e in submits for i in e.get('sample_ids', []))
    if matches != Counter(ids):
        failures.append('missing/duplicate/unexpected submission correlation')
    layouts = {}
    for e in resources:
        if e.get('all_descriptors_match') is not True or e.get('gpu_copies') != 0 or e.get('resource_states_verified') is not False:
            failures.append('resource observation contract')
        entries = e.get('resources', [])
        roles = [r.get('role') for r in entries]
        if sorted(roles) != sorted(('color','depth','motion','exposure','reactive','transparency','output')):
            failures.append('resource role set')
        total = 0
        for r in entries:
            if not r.get('present'):
                if r.get('role') in ('color','depth','motion','output'):
                    failures.append('missing required texture')
                continue
            if r.get('same_device') is not True or r.get('descriptor_matches') is not True:
                failures.append('COM/FFX descriptor mismatch')
            if any(r.get(k) != v for k,v in dict(dimension=3,array_size=1,mips=1,samples=1).items()):
                failures.append('unsupported texture shape')
            w,h,pitch,row,foot = (r.get(k,0) for k in ('width','height','row_pitch','row_bytes','footprint_bytes'))
            bpp = {10:8,34:4,40:4,41:4,61:1}.get(r.get('dxgi_format'))
            if not all(type(x) is int and x > 0 for x in (w,h,pitch,row,foot)) or bpp is None or row != w*bpp or pitch < row or pitch%256 or foot != pitch*(h-1)+row:
                failures.append('unverified footprint')
            total += foot
            fields = ('width','height','dxgi_format','ffx_format','d3d12_flags','declared_ffx_state','row_pitch','footprint_bytes')
            layouts.setdefault(r['role'], set()).add(tuple(r.get(k) for k in fields))
        if total != e.get('total_footprint_bytes'):
            failures.append('footprint total mismatch')
    queues = {e.get('queue') for e in submits if e.get('sample_ids')}
    if any(e.get('queue_type') != 0 or not e.get('queue') or e.get('queue') == '0x0' for e in submits if e.get('sample_ids')):
        failures.append('non-direct/invalid correlated queue')
    return dict(status='LIVE_INSPECTION_PASS' if not failures else 'LIVE_INSPECTION_FAIL',
                failures=sorted(set(failures)), sampled_dispatches=len(resources), correlated_dispatches=sum(matches.values()),
                queue_correlation_verified=matches==Counter(ids) and len(hooks)==1 and hooks[0].get('installed') is True,
                complete_descriptor_contract_verified=bool(resources) and all(e.get('all_descriptors_match') is True for e in resources),
                correlated_queues=sorted(queues), resource_layout_fields=list(fields) if resources and layouts else [],
                resource_layouts={k: sorted(v) for k,v in layouts.items()},
                total_footprint_bytes=sorted({e.get('total_footprint_bytes',0) for e in resources}),
                render_sizes=sorted({tuple(e['render_size']) for e in resources if 'render_size' in e}),
                raw_upscale_sizes=sorted({tuple(e['upscale_size']) for e in resources if 'upscale_size' in e}),
                motion_scales=sorted({tuple(e['motion_scale']) for e in resources if 'motion_scale' in e}),
                current_context_maximum_verified=False,
                resource_states_verified=False, gpu_pixels_captured=False, game_runtime_ready=False, dlss_nr_verified=False)


def read_log(path):
    raw = path.read_bytes()
    if len(raw) > 2*1024*1024 or not raw.endswith(b'\n'):
        raise ValueError('incomplete/oversized live log')
    return [json.loads(line) for line in raw.splitlines()]


def analyze_pair(base, observed):
    from analyze_ffx_dispatch_probe import analyze_provider
    a,b = analyze_provider(base), analyze_provider(observed)
    if a['output_sha256'] != b['output_sha256']:
        raise ValueError('inspection changed output')
    for name in ('requested_provider','requested_provider_id','adapter_vendor','adapter_device'):
        ma,mb = (json.loads((p/'manifest.json').read_text()) for p in (base,observed))
        if ma.get(name) is None or ma[name] != mb.get(name):
            raise ValueError('on/off provider/device mismatch')
    for name in ('depth_r32f.raw','motion_rg16f.raw','exposure_r32f.raw',*[f'input_{i}.raw' for i in range(4)]):
        if (base/name).read_bytes() != (observed/name).read_bytes():
            raise ValueError('on/off inputs differ')
    report = analyze_events(read_log(observed/'live.jsonl'), 4)
    report.update(output_sha256=b['output_sha256'], on_off_output_equal=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path',type=Path);p.add_argument('--expected',type=int,default=16);p.add_argument('--pairs',action='store_true');p.add_argument('--suffix',choices=('_inspect','_contract'),default='_inspect')
    args=p.parse_args()
    if args.pairs:
        reports={n:analyze_pair(args.path/n,args.path/(n+args.suffix)) for n in ('fsr3','fsr4')}
        report=dict(status='LIVE_INSPECTION_PASS' if all(r['status']=='LIVE_INSPECTION_PASS' for r in reports.values()) else 'LIVE_INSPECTION_FAIL',providers=reports)
    else:
        report=analyze_events(read_log(args.path/'live.jsonl'),args.expected)
        report['log_sha256']=hashlib.sha256((args.path/'live.jsonl').read_bytes()).hexdigest()
    (args.path/'live_analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2));return 0 if report['status']=='LIVE_INSPECTION_PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
