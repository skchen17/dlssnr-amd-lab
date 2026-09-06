"""Find narrow post-FFX output boundaries in logs; these never authorize live copies."""
import argparse
import json
from pathlib import Path
from analyze_ffx_resource_path import analyze as resource_summary, key
from analyze_ffx_state_trace import summarize, read_events, list_key


def analyze(events):
    resources = resource_summary(events)
    states = summarize(events)
    calls = states['dispatch_boundary_evidence']['windows']
    trace = [e for e in events if e.get('event') == 'state_trace']
    rows = []
    for detail in resources['calls']:
        identity = key(detail)
        candidates = [e for e in events if e.get('event') == 'candidate' and key(e) == identity]
        boundaries = [c for c in calls if key(c) == identity]
        if len(candidates) != 1 or len(boundaries) != 1:
            rows.append(dict(window=identity[0], sample=identity[3], reasons=['missing unique FFX boundary']))
            continue
        candidate, call = candidates[0], boundaries[0]
        local = [e for e in trace if list_key(e) == list_key(call)]
        path = [e for e in events if e.get('event') == 'command_path' and key(e) == identity][0]
        snapshots = [e for e in events if e.get('event') == 'resource_path' and
                     e.get('kind') == 'resources_before' and key(e) == identity]
        output = next(r['info'] for r in snapshots[0]['roles'] if r['role'] == 'output')
        pointer = output['resource']
        reasons = []
        if not candidate.get('reset_observed'):
            reasons.append('list reset not observed')
        if not detail['barrier_count_matches'] or not detail['resource_fields_unchanged']:
            reasons.append('incomplete or changed descriptor evidence')
        if (path['original_status'] != 0 or not path['list_table_matches'] or
                path['enhanced_barrier_calls'] or path['other_interface_calls']):
            reasons.append('unsupported FFX command path')
        if (output.get('dimension') != 3 or output.get('format') != 10 or
                output.get('mips') != 1 or output.get('array_size') != 1 or
                not 1 <= output.get('width', 0) <= 8192 or not 1 <= output.get('height', 0) <= 8192):
            reasons.append('unsupported output shape')
        if len([e for e in local if e.get('kind') == 'dispatch_begin']) != 1:
            reasons.append('multiple FFX calls in recording generation')
        transition = next((e for e in local if e.get('kind') == 'transition' and
                           e.get('resource') == pointer and e['sequence'] > call['end_sequence']), None)
        if (not transition or transition.get('flags') != 0 or transition.get('before') not in (8, 64) or
                transition.get('after') != 192 or transition.get('subresource') not in (0, 4294967295)):
            reasons.append('missing supported complete post-call transition')
        submits = [e for e in trace if e.get('kind') == 'submit_begin' and e.get('window', 0) == identity[0]
                   for m in e.get('lists', []) if (m.get('list'), m.get('generation')) == identity[1:3]]
        batch = None
        if len(submits) != 1:
            reasons.append('missing or repeated submission')
        else:
            matching = [b for b in states['submission_batch_evidence']['complete_batches']
                        if (b['window'], b['batch_id']) == (identity[0], submits[0]['batch_id'])]
            if len(matching) != 1:
                reasons.append('incomplete submission batch')
            else:
                batch = matching[0]
            closes = [e for e in local if e.get('kind') == 'close' and e.get('succeeded') is True and
                      transition and transition['sequence'] < e['sequence'] < submits[0]['sequence']]
            if len(closes) != 1:
                reasons.append('missing unique successful close after transition')
        if any(e.get('kind') in ('alias', 'enhanced_state_unknown') and
               e.get('window', 0) == identity[0] and e.get('list') == identity[1] for e in trace):
            reasons.append('alias or enhanced-state uncertainty on list')
        rows.append(dict(window=identity[0], sample=identity[3], list=identity[1], generation=identity[2],
                         output_resource=pointer, output_identity=output.get('identity'),
                         transition=transition, submission_batch_id=batch['batch_id'] if batch else None,
                         context_known=candidate.get('context_known') is True,
                         reasons=reasons, structurally_matched=not reasons))
    return dict(status='OUTPUT_BOUNDARY_CANDIDATES_ONLY', detailed_calls=len(rows),
                structurally_matched=sum(r.get('structurally_matched', False) for r in rows), calls=rows,
                trace_truncated=states['trace_truncated'], resource_detail_truncated=resources['detail_truncated'],
                limitations=['Transition events are recorded before original ResourceBarrier forwarding',
                             'No barrier-batch identifier: a later barrier in the same callback may supersede its state',
                             'No complete cross-queue ownership or live GPU validation',
                             'Output only; no input-frame/context/temporal dataset proof'],
                live_copy_ready=False, capture_authorized=False, game_frame_capture_verified=False, dlss_nr_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite report')
    report = analyze(read_events(args.log))
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'calls'}, indent=2))
