"""Explain same-batch legacy transition evidence, without certifying GPU state."""
import argparse
from collections import Counter
import json
from pathlib import Path
from analyze_ffx_state_trace import read_events, list_key, summarize


def analyze(events):
    # Validate window/sequence limits before using ordering metadata.
    summary = summarize(events)
    trace = [e for e in events if e.get('event') == 'state_trace']
    batches = summary['submission_batch_evidence']['complete_batches']
    windows = summary['dispatch_boundary_evidence']['windows']
    candidates = [e for e in events if e.get('event') == 'candidate']
    rows = []
    for call in windows:
        key = list_key(call)
        matches = [(b, i) for b in batches if b['window'] == call['window']
                   for i, member in enumerate(b['lists'])
                   if (member['list'], member['generation']) == (call['list'], call['generation'])]
        if len(matches) != 1:
            continue  # Repeated submission/ambiguous association is not resolved here.
        batch, index = matches[0]
        starts = [e for e in trace if e.get('kind') == 'submit_begin'
                  and e.get('window', 0) == batch['window'] and e.get('batch_id') == batch['batch_id']]
        if len(starts) != 1 or starts[0]['sequence'] <= call['end_sequence']:
            continue
        candidate = [e for e in candidates if list_key(e) == key and e.get('sample') == call['sample']]
        if len(candidate) != 1:
            continue
        roles = []
        for resource in candidate[0]['resources']:
            pointer = resource.get('resource')
            if not pointer:
                continue
            state, origin, gaps = None, None, []
            for member in batch['lists'][:index+1]:
                # CPU recording can interleave. Sort within each submitted list,
                # then use the explicit batch index, never global log adjacency.
                member_key = (batch['window'], member['list'], member['generation'])
                cutoff = call['begin_sequence'] if member['index'] == index else starts[0]['sequence']
                transitions = [e for e in trace if list_key(e) == member_key
                               and e.get('kind') == 'transition' and e.get('resource') == pointer
                               and e['sequence'] < cutoff and e.get('subresource') in (0, 4294967295)]
                for e in sorted(transitions, key=lambda e: e['sequence']):
                    if state is not None and state != e.get('before'):
                        gaps.append(dict(prior_after=state, next_before=e.get('before'), sequence=e['sequence']))
                    state = e.get('after') if e.get('flags') == 0 else None
                    origin = dict(list=member['list'], generation=member['generation'],
                                  batch_index=member['index'], sequence=e['sequence'])
            # Alias events lack a generation in this recorder. Conservatively
            # withhold the result if any prefix-list alias was observed in this window.
            prefix_lists = {m['list'] for m in batch['lists'][:index+1]}
            alias = any(e.get('kind') == 'alias' and e.get('window', 0) == batch['window']
                        and e.get('list') in prefix_lists and e['sequence'] < starts[0]['sequence'] for e in trace)
            enhanced = any(e.get('kind') == 'enhanced_state_unknown' and e.get('window', 0) == batch['window']
                           and e.get('list') in prefix_lists and e['sequence'] < starts[0]['sequence'] for e in trace)
            if alias or enhanced:
                state = None
            roles.append(dict(role=resource['role'], resource=pointer, declared_state=resource['expected_state'],
                              last_observed_prefix_state=state, origin=origin,
                              transition_chain_gaps=gaps, alias_uncertainty=alias, enhanced_uncertainty=enhanced))
        rows.append(dict(window=call['window'], sample=call['sample'], batch_id=batch['batch_id'],
                         fsr_batch_index=index, batch_count=batch['batch_count'],
                         fsr_reset_observed=candidate[0].get('reset_observed') is True,
                         untracked_prefix_members=sum(m.get('tracked') is not True for m in batch['lists'][:index+1]),
                         roles=roles))
    aggregate = {}
    for role in sorted({r['role'] for row in rows for r in row['roles']}):
        selected = [r for row in rows for r in row['roles'] if r['role'] == role]
        aggregate[role] = dict(observed_states=dict(Counter(str(r['last_observed_prefix_state']) for r in selected)),
                               rows_with_chain_gaps=sum(bool(r['transition_chain_gaps']) for r in selected))
    return dict(status='PREFIX_EVIDENCE_ONLY', associated_calls=len(rows), roles=aggregate, calls=rows,
                trace_truncated=summary['trace_truncated'],
                limitations=['Legacy barriers only', 'No inter-batch or cross-queue state propagation',
                             'Last observed state is not a complete transition chain or capture permission',
                             'Reset/lifecycle coverage of all predecessor lists is not established',
                             'Unobserved transitions and alternate interfaces are not excluded'],
                capture_authorized=False, gpu_state_verified=False, game_frame_capture_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(read_events(args.log))
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'calls'}, indent=2))
