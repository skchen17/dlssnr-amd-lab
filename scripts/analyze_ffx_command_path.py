"""Validate bounded command-path counters without inferring unobserved GPU work."""
import argparse
import json
from pathlib import Path

from analyze_ffx_state_trace import read_events, summarize, list_key, validate_observe_pair

COUNTERS = ('gpu_dispatch_calls', 'execute_indirect_calls', 'legacy_barrier_calls',
            'legacy_barriers', 'enhanced_barrier_calls', 'enhanced_groups', 'other_interface_calls')


def analyze(events):
    state = summarize(events)
    hooks = [e for e in events if e.get('event') == 'command_path_hooks']
    if len(hooks) != 1 or hooks[0].get('installed') is not True:
        raise ValueError('missing/failed command path hooks')
    paths = [e for e in events if e.get('event') == 'command_path']
    candidates = [e for e in events if e.get('event') == 'candidate']
    keys = [(list_key(e), e.get('sample')) for e in paths]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate command path')
    for path, key in zip(paths, keys):
        matches = [e for e in candidates if (list_key(e), e.get('sample')) == key]
        if len(matches) != 1:
            raise ValueError('unmatched command path identity')
        if any(type(path.get(k)) is not int or not 0 <= path[k] <= 1000000 for k in COUNTERS):
            raise ValueError('invalid command counter')
        if path.get('capture_authorized') is not False:
            raise ValueError('unexpected capture authorization')
        if type(path.get('list_table_matches')) is not bool or type(path.get('original_status')) is not int:
            raise ValueError('invalid path coverage/status')
        total = sum(path[k] for k in ('gpu_dispatch_calls', 'execute_indirect_calls', 'legacy_barrier_calls', 'enhanced_barrier_calls'))
        if path['other_interface_calls'] > total:
            raise ValueError('inconsistent other-interface counter')
    return dict(status='COMMAND_PATH_EVIDENCE', calls=len(paths), candidates=state['candidates'],
                totals={k: sum(p[k] for p in paths) for k in COUNTERS},
                zero_observed_command_calls=sum(not any(p[k] for k in ('gpu_dispatch_calls', 'execute_indirect_calls', 'legacy_barrier_calls', 'enhanced_barrier_calls')) for p in paths),
                table_mismatches=sum(p['list_table_matches'] is not True for p in paths),
                nonzero_return_statuses=sum(p['original_status'] != 0 for p in paths),
                unpaired_candidates=len(candidates) - len(paths), enhanced_hook_supported=hooks[0].get('enhanced_supported') is True,
                trace_truncated=state['trace_truncated'], failures=state['failures'],
                scope='Synchronous same-thread callbacks through selected tables; wrapper reentry may double-count',
                capture_authorized=False, gpu_execution_verified=False, game_frame_capture_verified=False)


def controls(root):
    result = {}
    for name in ('fsr3', 'fsr4'):
        records = {}
        for suffix in ('_observe', '_window'):
            window = suffix == '_window'
            validate_observe_pair(root / name, root / (name + suffix), window)
            events = read_events(root / (name + suffix) / 'session.jsonl')
            report = analyze(events)
            if (report['calls'] != (2 if window else 4) or report['unpaired_candidates'] or
                    report['table_mismatches'] or report['nonzero_return_statuses'] or report['failures']):
                raise ValueError('incomplete path control')
            paths = [e for e in events if e.get('event') == 'command_path']
            if any(p['gpu_dispatch_calls'] <= 0 or p['legacy_barrier_calls'] <= 0 for p in paths):
                raise ValueError('missing positive compute/legacy control')
            if window:
                manifest = json.loads((root / (name + suffix) / 'manifest.json').read_text())
                enhanced = [e for e in events if e.get('kind') == 'enhanced_state_unknown']
                if (manifest.get('window_enhanced_global_barrier_probe') is not True or
                        [e.get('window') for e in enhanced] != [1, 2] or
                        any(e.get('group_count') != 1 for e in enhanced) or not report['enhanced_hook_supported']):
                    raise ValueError('missing positive enhanced control')
                # The fixture inserts its global barrier before, NOT inside FFX.
                if any(p['enhanced_barrier_calls'] != 0 for p in paths):
                    raise ValueError('control changed: enhanced calls inside FFX need separate attribution')
            records[suffix[1:]] = report
        result[name] = records
    return dict(status='COMMAND_PATH_CONTROL_PASS', providers=result,
                on_off_output_equal=True, enhanced_positive_control=True, game_frame_capture_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite report')
    report = controls(args.path) if args.controls else analyze(read_events(args.path))
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))
