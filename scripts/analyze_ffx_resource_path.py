"""Audit within-call resource identity evidence, never infer state or GPU aliasing."""
import argparse
from collections import Counter
import json
from pathlib import Path
from analyze_ffx_state_trace import read_events
from analyze_ffx_command_path import analyze as analyze_commands, controls as command_controls

ROLES = {'color', 'depth', 'motion', 'exposure', 'reactive', 'transparency', 'output'}


def key(e):
    return e.get('window', 0), e.get('ffx_list', e.get('list')), e.get('generation'), e.get('sample')


def present(p):
    return isinstance(p, str) and p.startswith('0x') and int(p, 16) != 0


def analyze(events):
    commands = analyze_commands(events)
    detail = [e for e in events if e.get('event') == 'resource_path']
    windows = [e.get('window', 0) for e in detail]
    if any(type(w) is not int or not 0 <= w <= 8 for w in windows) or windows != sorted(windows):
        raise ValueError('resource window order')
    for window in set(windows):
        rows = [e for e in detail if e.get('window', 0) == window]
        if len(rows) > 512 or [e.get('sequence') for e in rows] != list(range(1, len(rows) + 1)):
            raise ValueError('resource sequence/budget')
    for e in detail:
        if type(e.get('sample')) is not int or not 1 <= e['sample'] <= 4:
            raise ValueError('resource sample budget')
    grouped = {}
    for e in detail:
        grouped.setdefault(key(e), []).append(e)
    calls, incomplete = [], []
    for identity, rows in grouped.items():
        before = [e for e in rows if e.get('kind') == 'resources_before']
        after = [e for e in rows if e.get('kind') == 'resources_after']
        candidates = [e for e in events if e.get('event') == 'candidate' and key(e) == identity]
        paths = [e for e in events if e.get('event') == 'command_path' and key(e) == identity]
        if len(before) > 1 or len(after) > 1:
            raise ValueError('duplicate snapshot')
        if len(before) != 1 or len(after) != 1 or len(candidates) != 1 or len(paths) != 1:
            incomplete.append(identity)
            continue
        before, after = before[0], after[0]
        roles = before.get('roles', [])
        if len(roles) != 7 or {r.get('role') for r in roles} != ROLES:
            raise ValueError('resource role set')
        if len(after.get('roles', [])) != 7 or {r.get('role') for r in after['roles']} != ROLES:
            raise ValueError('post resource role set')
        known = {r['role']: r['info'] for r in roles}
        for role, info in known.items():
            if present(info.get('resource')) and not present(info.get('identity')):
                raise ValueError('missing COM identity')
        for c in candidates[0].get('resources', []):
            if c.get('role') not in known or c.get('resource') != known[c['role']].get('resource'):
                raise ValueError('candidate pointer mismatch')

        def validate_info(info):
            if not present(info.get('resource')):
                if info.get('raw_matches') or info.get('identity_matches') or present(info.get('identity')):
                    raise ValueError('null resource cannot match')
                return
            if not present(info.get('identity')):
                raise ValueError('missing queried identity')
            raw = sorted(r for r, v in known.items() if v.get('resource') == info['resource'])
            canonical = sorted(r for r, v in known.items() if v.get('identity') == info['identity'])
            if sorted(info.get('raw_matches', [])) != raw or sorted(info.get('identity_matches', [])) != canonical:
                raise ValueError('resource match classification')

        barriers = [e for e in rows if e.get('kind') == 'barrier']
        if any(e.get('kind') not in ('resources_before', 'barrier', 'resources_after') for e in rows):
            raise ValueError('unknown resource event')
        if not before['sequence'] < after['sequence'] or any(not before['sequence'] < b['sequence'] < after['sequence'] for b in barriers):
            raise ValueError('resource boundary order')
        for snapshot in (before, after):
            for r in snapshot['roles']:
                validate_info(r['info'])
        before_fields = {r['role']: (r['info']['resource'], r['declared_ffx_state']) for r in before['roles']}
        after_fields = {r['role']: (r['info']['resource'], r['declared_ffx_state']) for r in after['roles']}
        unchanged = before_fields == after_fields and before.get('command_list') == after.get('command_list') == identity[1]
        if type(after.get('resource_fields_unchanged')) is not bool or after['resource_fields_unchanged'] != unchanged:
            raise ValueError('descriptor mutation classification')
        raw_hits, canonical_only, null_refs = 0, 0, 0
        for b in barriers:
            if b.get('type') not in (0, 1, 2):
                raise ValueError('invalid barrier type')
            infos = [b['before_info'], b['after_info']] if b['type'] == 1 else [b['info']]
            for info in infos:
                validate_info(info)
            raw_hits += any(i.get('raw_matches') for i in infos)
            canonical_only += any(set(i.get('identity_matches', [])) - set(i.get('raw_matches', [])) for i in infos)
            null_refs += sum(not present(i.get('resource')) for i in infos)
        calls.append(dict(window=identity[0], list=identity[1], generation=identity[2], sample=identity[3],
                          observed_barriers=len(barriers), expected_barrier_callbacks=paths[0]['legacy_barriers'],
                          barrier_count_matches=len(barriers) == paths[0]['legacy_barriers'],
                          types=dict(Counter({0: 'transition', 1: 'aliasing', 2: 'uav'}[b['type']] for b in barriers)),
                          raw_matched_barriers=raw_hits, identity_only_matched_barriers=canonical_only,
                          null_resource_references=null_refs, resource_fields_unchanged=after.get('resource_fields_unchanged') is True))
    return dict(status='RESOURCE_IDENTITY_EVIDENCE', complete_calls=len(calls), calls=calls,
                incomplete_calls=incomplete, detail_events=len(detail),
                detail_truncated=any(e.get('event') == 'resource_path_limit' for e in events),
                command_calls=commands['calls'], scope='Within-call raw/COM identity only; not GPU storage aliasing or complete state',
                capture_authorized=False, resource_state_verified=False, game_frame_capture_verified=False)


def controls(root):
    command_controls(root)
    reports = {}
    for provider in ('fsr3', 'fsr4'):
        reports[provider] = {}
        for suffix in ('_observe', '_window'):
            report = analyze(read_events(root / (provider + suffix) / 'session.jsonl'))
            if (report['complete_calls'] != (2 if suffix == '_window' else 4) or report['detail_truncated'] or report['incomplete_calls'] or
                    any(not c['barrier_count_matches'] or not c['resource_fields_unchanged'] for c in report['calls'])):
                raise ValueError('incomplete/changed resource control')
            reports[provider][suffix[1:]] = report
    return dict(status='RESOURCE_IDENTITY_CONTROL_PASS', providers=reports, game_frame_capture_verified=False)


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
