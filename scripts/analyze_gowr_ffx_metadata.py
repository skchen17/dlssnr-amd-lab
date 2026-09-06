"""Snapshot and summarize live game FFX metadata, without reading GPU resources."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def analyze_events(events):
    creates = [e for e in events if e.get('event') == 'ffxCreateContext' and e.get('body_decoded')]
    dispatches = [e for e in events if e.get('event') == 'ffxDispatch']
    decoded = [e for e in dispatches if e.get('body_decoded')]
    contexts = {e.get('context_after'): e for e in creates if e.get('return_code') == 0}
    issues = Counter()
    layouts = defaultdict(set)
    render_sizes, effective_sizes, flags, scales = set(), set(), set(), set()
    byte_estimates = set()
    for e in decoded:
        if not e.get('bool_encoding_valid'):
            issues['invalid_bool'] += 1
        if e.get('extension_types') or e.get('extension_chain') != 'complete':
            issues['unknown_dispatch_extension'] += 1
        context = contexts.get(e.get('context_before'))
        if context is None:
            issues['missing_context'] += 1
            continue
        flags.add(context['create_flags'])
        render, upscale = e.get('render_size', []), e.get('upscale_size', [])
        effective = context['max_upscale'] if upscale == [0, 0] else upscale
        if len(render) != 2 or len(effective) != 2 or any(type(x) is not int or x < 1 or x > 8192 for x in render + effective):
            issues['invalid_dimensions'] += 1
            continue
        render_sizes.add(tuple(render)); effective_sizes.add(tuple(effective))
        scales.add(tuple(e.get('motion_scale', [])))
        for key in ('jitter','motion_scale'):
            if len(e.get(key, [])) != 2 or not all(isinstance(x, (float,int)) and math.isfinite(x) for x in e[key]):
                issues['invalid_motion_or_jitter'] += 1
        resources = e.get('resources', {})
        for role in ('color','depth','motion','output'):
            if not resources.get(role, {}).get('pointer') or resources[role]['pointer'] == '0x0':
                issues['missing_required_resource'] += 1
        total = 0
        for role, r in resources.items():
            if r.get('pointer') == '0x0':
                layouts[role].add(('absent',))
                continue
            fields = tuple(r.get(k) for k in ('width','height','format','ffx_state','usage','type','mips','depth'))
            layouts[role].add(fields)
            if r.get('type') != 2 or r.get('mips') != 1 or r.get('depth') != 1:
                issues['unsupported_resource_shape'] += 1
            bpp = {4:8,28:4,18:4,25:1}.get(r.get('format'))
            if bpp is None:
                issues['unknown_format_estimate'] += 1
            elif type(r.get('width')) is int and type(r.get('height')) is int:
                total += r['width'] * r['height'] * bpp
        byte_estimates.add(total)
    failures = Counter((e.get('event'),e.get('return_code')) for e in events if e.get('return_code') != 0)
    if len(decoded) != len(dispatches): issues['undecoded_dispatch'] += len(dispatches)-len(decoded)
    coherent = bool(decoded) and not issues and not failures
    return dict(status='METADATA_COHERENT' if coherent else 'METADATA_INCOMPLETE_OR_INCONSISTENT',
        event_count=len(events), event_counts=dict(Counter(e.get('event') for e in events)),
        decoded_dispatches=len(decoded), distinct_dispatch_threads=len({e.get('thread_id') for e in decoded}),
        distinct_command_lists=len({e.get('command_list') for e in decoded}),
        create_flags=sorted(flags), render_sizes=sorted(render_sizes), effective_upscale_sizes=sorted(effective_sizes),
        dispatch_upscale_size_default_count=sum(e.get('upscale_size') == [0,0] for e in decoded),
        reset_count=sum(e.get('reset') == 1 for e in decoded), motion_scales=sorted(scales),
        resource_layout_field_order=['width','height','ffx_format','ffx_state','usage','type','mips','depth'],
        resource_layouts={k: sorted(v, key=str) for k,v in layouts.items()},
        estimated_unpadded_capture_bytes_per_dispatch=sorted(byte_estimates),
        issues=dict(issues), failures=[dict(event=k[0],status=k[1],count=v) for k,v in failures.items()],
        gpu_resource_descriptors_verified=False, game_texture_capture_ready=False,
        active_provider_version_verified=False, dlss_nr_verified=False,
        scope='Live game descriptor metadata only; no COM resource inspection, pixels or GPU queue proof.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('output must be new')
    raw = (args.run / 'events.jsonl').read_bytes()
    if len(raw) > 64*1024*1024: raise ValueError('metadata log size cap exceeded')
    end = raw.rfind(b'\n') + 1
    complete = raw[:end]  # A writer may still be producing the final line.
    def reject_constant(value): raise ValueError('nonstandard JSON constant: ' + value)
    events = [json.loads(line, parse_constant=reject_constant) for line in complete.splitlines()]
    report = analyze_events(events)
    report.update(snapshot_sha256=hashlib.sha256(complete).hexdigest(), snapshot_bytes=len(complete),
                  ignored_partial_tail_bytes=len(raw)-end, source_run=str(args.run.resolve()))
    args.output.mkdir(parents=True)
    (args.output / 'events.jsonl').write_bytes(complete)
    (args.output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'METADATA_COHERENT' else 1


if __name__ == '__main__':
    raise SystemExit(main())
