"""Audit bounded FFX state evidence; never certify GPU state from CPU log order."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_ffx_dispatch_probe import analyze_provider


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def list_key(event):
    return event.get('window', 0), event.get('list'), event.get('generation')


def correlate_batches(events):
    groups = {}
    for e in events:
        if e.get('event') == 'state_trace' and 'batch_id' in e:
            groups.setdefault((e.get('window', 0), e['batch_id']), []).append(e)
    complete, incomplete, invalid = [], [], []
    for (window, batch), group in groups.items():
        begins = [e for e in group if e.get('kind') == 'submit_begin']
        ends = [e for e in group if e.get('kind') == 'submit_end']
        if len(begins) != 1 or len(ends) != 1:
            incomplete.append(dict(window=window, batch_id=batch))
            continue
        b, end = begins[0], ends[0]
        members = b.get('lists', [])
        valid = (0 < b.get('batch_count', 0) <= 64 and len(members) == b['batch_count']
                 and [m.get('index') for m in members] == list(range(len(members)))
                 and b['sequence'] < end['sequence'] and b.get('queue') == end.get('queue'))
        returns = [e for e in group if e.get('kind') == 'submit_return']
        valid &= len({e.get('batch_index') for e in returns}) == len(returns)
        for e in returns:
            index = e.get('batch_index', -1)
            if not 0 <= index < len(members):
                valid = False
                continue
            m = members[index]
            valid &= (m.get('tracked') is True and e.get('list') == m.get('list')
                      and e.get('generation') == m.get('generation')
                      and e.get('queue') == b.get('queue') and e.get('batch_count') == len(members)
                      and b['sequence'] < e['sequence'] < end['sequence'])
        target = complete if valid else invalid
        target.append(dict(window=window, batch_id=batch, queue=b.get('queue'),
                           lists=members, batch_count=b.get('batch_count')))
    return dict(complete_batches=complete, incomplete_batches=incomplete,
                invalid_batches=invalid, gpu_execution_order_verified=False)


def correlate_dispatch_windows(events):
    """Correlate CPU recording evidence only; no cross-list state propagation."""
    trace = [e for e in events if e.get('event') == 'state_trace']
    candidates = [e for e in events if e.get('event') == 'candidate']
    windows = []
    for begin in (e for e in trace if e.get('kind') == 'dispatch_begin'):
        key = list_key(begin)
        related = [e for e in trace if list_key(e) == key]
        ends = [e for e in related if e.get('kind') == 'dispatch_end' and e['sequence'] > begin['sequence']]
        matches = [e for e in candidates if list_key(e) == key and e.get('sample') == begin.get('sample')]
        if len(ends) != 1 or len(matches) != 1:
            continue
        end, candidate = ends[0], matches[0]
        roles = []
        for r in candidate.get('resources', []):
            resource = r.get('resource')
            if not resource:
                continue
            transitions = [e for e in trace if e.get('window', 0) == key[0] and e.get('kind') == 'transition' and e.get('resource') == resource]
            local = [e for e in transitions if list_key(e) == key]
            inside = [e for e in local if begin['sequence'] < e['sequence'] < end['sequence']]
            after = [e for e in local if e['sequence'] > end['sequence']]
            external = [e for e in transitions if e.get('list') != begin.get('list')]
            roles.append(dict(role=r.get('role'), resource=resource,
                              local_state_at_boundary=r.get('plane0_recorded_state'),
                              observed_transitions_inside_call=len(inside),
                              first_local_transition_after=after[0] if after else None,
                              other_lists_with_transitions=sorted({e['list'] for e in external})))
        windows.append(dict(window=key[0], sample=begin.get('sample'), list=key[1], generation=key[2],
                            begin_sequence=begin['sequence'], end_sequence=end['sequence'],
                            original_status=end.get('original_status'), roles=roles))
    return dict(complete_observed_call_windows=len(windows), windows=windows,
                scope='CPU recording evidence for watched resources; not global GPU state',
                cross_list_gpu_state_verified=False)


def summarize(events):
    candidates = [e for e in events if e.get('event') == 'candidate']
    contexts = {}
    for e in events:
        if e.get('event') == 'context_create':
            contexts[e['context']] = e
        elif e.get('event') == 'context_destroy_begin':
            contexts.pop(e.get('context'), None)
    last = candidates[-1] if candidates else {}
    context = contexts.get(last.get('context'))
    known = bool(context and last.get('context_known') is True and last.get('epoch') == context.get('epoch'))
    issues = []
    for r in last.get('resources', []):
        state, expected = r.get('plane0_recorded_state', -1), r.get('expected_state', -1)
        if state < 0 or expected < 0 or state != expected:
            issues.append(dict(role=r.get('role'), recorded=state, declared=expected,
                               reason='unknown' if state < 0 or expected < 0 else 'mismatch'))
    trace = [e for e in events if e.get('event') == 'state_trace']
    identifiers = [e.get('window', 0) for e in trace]
    if any(type(w) is not int or not 0 <= w <= 8 for w in identifiers) or identifiers != sorted(identifiers):
        raise ValueError('observation window identity/order')
    for window in set(identifiers):
        part = [e for e in trace if e.get('window', 0) == window]
        if len(part) > 1024 or [e.get('sequence') for e in part] != list(range(1, len(part)+1)):
            raise ValueError('trace sequence/budget')
    failures = [e for e in events if e.get('event') == 'failure']
    return dict(status='OBSERVATION_ONLY', candidates=len(candidates),
                latest_context_verified_in_log=known, latest_context=context if known else None,
                latest_resource_state_issues=issues,
                ready_candidates=sum(e.get('recording_gate_ready') is True for e in candidates),
                failures=failures, trace_events=len(trace),
                trace_kinds=dict(Counter(e.get('kind') for e in trace)),
                trace_truncated=any(e.get('event') == 'state_trace_limit' for e in events),
                recorded_input_copy_events=sum(e.get('event') == 'capture_recorded_inputs' for e in events),
                dispatch_boundary_evidence=correlate_dispatch_windows(events),
                submission_batch_evidence=correlate_batches(events),
                cross_list_gpu_state_verified=False, game_frame_capture_verified=False,
                dlss_nr_verified=False)


def validate_observe_pair(base, observed, window_mode=False):
    a, b = analyze_provider(base), analyze_provider(observed)
    if a['output_sha256'] != b['output_sha256']:
        raise ValueError('observer changed output')
    m = json.loads((observed/'manifest.json').read_text())
    bm = json.loads((base/'manifest.json').read_text())
    if m.get('session_observe_only') is not True or m.get('session_poll_status') != 2:
        raise ValueError('observe-only native status')
    for key in ('requested_provider_id', 'adapter_vendor', 'adapter_device'):
        if m.get(key) != bm.get(key):
            raise ValueError('provider/adapter mismatch')
    if (observed/'capture').exists():
        raise ValueError('unexpected capture artifacts')
    events = read_events(observed/'session.jsonl')
    if any(e.get('event', '').startswith('capture_') for e in events):
        raise ValueError('unexpected capture lifecycle')
    report = summarize(events)
    if report['failures'] or report['trace_truncated'] or report['candidates'] != (2 if window_mode else 4):
        raise ValueError('incomplete observation')
    if window_mode:
        if (m.get('session_window_control') is not True or m.get('window_idle_frames') != 2
                or m.get('window_triggers_with_frames') != 2 or m.get('window_cap_rejected') is not True
                or m.get('window_busy_rejected') is not True or m.get('window_split_producer_lists') is not True):
            raise ValueError('native window controls')
        begins = [e for e in events if e.get('event') == 'observation_begin']
        stops = [e for e in events if e.get('event') == 'observation_stop']
        if [e.get('window') for e in begins] != list(range(1, 9)) or [e.get('window') for e in stops] != list(range(1, 9)):
            raise ValueError('window lifecycle/budget')
        if any(e.get('capture_armed') is not False for e in begins+stops):
            raise ValueError('window unexpectedly armed')
        if [e.get('window') for e in events if e.get('event') == 'candidate'] != [1, 2]:
            raise ValueError('candidate recorded outside selected windows')
        active_window = None
        for e in events:
            if e.get('event') == 'observation_begin':
                if active_window is not None:
                    raise ValueError('overlapping observation windows')
                active_window = e.get('window')
            elif e.get('event') == 'observation_stop':
                if active_window != e.get('window'):
                    raise ValueError('unmatched observation stop')
                active_window = None
            elif e.get('event') in ('candidate', 'state_trace') and e.get('window') != active_window:
                raise ValueError('recording outside active observation')
    trace = [e for e in events if e.get('event') == 'state_trace']
    for kind in ('transition', 'close', 'submit_return', 'dispatch_begin', 'dispatch_end'):
        if not any(e['kind'] == kind for e in trace):
            raise ValueError('missing trace kind: '+kind)
    # Validate association by list generation, not log adjacency across CPU threads.
    for begin in (e for e in trace if e['kind'] == 'dispatch_begin'):
        related = [e for e in trace if list_key(e) == list_key(begin)]
        positions = []
        for kind in ('dispatch_begin', 'dispatch_end', 'close', 'submit_return'):
            group = [e for e in related if e['kind'] == kind]
            if len(group) != 1:
                raise ValueError('ambiguous/missing list lifecycle')
            positions.append(group[0]['sequence'])
        if positions != sorted(positions):
            raise ValueError('list lifecycle ordering')
    batches = report['submission_batch_evidence']
    if batches['invalid_batches'] or batches['incomplete_batches'] or len(batches['complete_batches']) != (2 if window_mode else 4):
        raise ValueError('submission batch association')
    if window_mode:
        for batch in batches['complete_batches']:
            if batch['batch_count'] != 2 or any(m.get('tracked') is not True for m in batch['lists']):
                raise ValueError('two-list batch not observed')
        for candidate in (e for e in events if e.get('event') == 'candidate'):
            if candidate.get('recording_gate_ready') is not False:
                raise ValueError('split-list gate must remain conservative')
            for r in candidate['resources']:
                if r['role'] in ('depth', 'motion') and r['plane0_recorded_state'] != -1:
                    raise ValueError('external state incorrectly propagated')
    return dict(status='WINDOW_CONTROL_PASS' if window_mode else 'STATE_TRACE_CONTROL_PASS', on_off_output_equal=True,
                trace_events=len(trace), capture_enabled=False,
                split_producer_lists_verified=window_mode,
                game_frame_capture_verified=False, dlss_nr_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--window-pairs', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite report')
    if args.paired and args.window_pairs:
        raise SystemExit('Select one pair mode')
    if args.paired or args.window_pairs:
        report = {name: validate_observe_pair(args.path/name, args.path/(name+('_window' if args.window_pairs else '_observe')), args.window_pairs)
                  for name in ('fsr3', 'fsr4')}
        report['status'] = 'PASS'
    else:
        report = summarize(read_events(args.path))
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))
