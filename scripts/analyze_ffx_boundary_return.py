"""Audit post-forward whole-barrier-callback evidence, not GPU copy permission."""
import argparse
import json
from pathlib import Path
from analyze_ffx_resource_path import analyze as resources, key
from analyze_ffx_state_trace import read_events, validate_observe_pair
from analyze_ffx_command_path import controls as command_controls
from analyze_ffx_dispatch_probe import analyze_provider
from analyze_ffx_output_patch import analyze as analyze_patch
from analyze_ffx_output_filter import analyze as analyze_filter
from analyze_ffx_output_network import analyze as analyze_network


def analyze(events):
    resource_report = resources(events)
    returns = [e for e in events if e.get('event') == 'output_boundary_return']
    seen = set()
    rows = []
    for event in returns:
        identity = key(event)
        if identity in seen:
            raise ValueError('duplicate boundary callback')
        seen.add(identity)
        paths = [e for e in events if e.get('event') == 'command_path' and key(e) == identity]
        snapshots = [e for e in events if e.get('event') == 'resource_path' and key(e) == identity and e.get('kind') == 'resources_after']
        details = [e for e in resource_report['calls'] if key(e) == identity]
        if len(paths) != 1 or len(snapshots) != 1 or len(details) != 1:
            raise ValueError('unmatched boundary callback')
        snapshot = snapshots[0]
        output = next(r['info'] for r in snapshot['roles'] if r['role'] == 'output')
        if (event.get('resource') != output.get('resource') or paths[0]['original_status'] != 0 or
                events.index(event) <= events.index(paths[0])):
            raise ValueError('boundary identity/order/status')
        for field in ('barrier_count', 'target_transitions', 'alias_barriers', 'unknown_barriers'):
            if type(event.get(field)) is not int or not 0 <= event[field] <= 4096:
                raise ValueError('invalid batch count')
        if not 1 <= event['barrier_count'] or sum(event[f] for f in ('target_transitions', 'alias_barriers', 'unknown_barriers')) > event['barrier_count']:
            raise ValueError('inconsistent batch count')
        supported = (event['target_transitions'] == 1 and event['alias_barriers'] == event['unknown_barriers'] == 0 and
                     event.get('last_target_before') in (8, 64) and event.get('last_target_after') == 192 and
                     event.get('last_target_subresource') in (0, 4294967295) and event.get('last_target_flags') == 0)
        if type(event.get('batch_supported')) is not bool or event['batch_supported'] != supported:
            raise ValueError('false batch classification')
        if (event.get('original_callback_returned') is not True or event.get('original_forward_calls') != 1 or
                event.get('capture_authorized') is not False or type(event.get('lifecycle_match')) is not bool):
            raise ValueError('forwarding/authorization contract')
        if not details[0]['resource_fields_unchanged']:
            raise ValueError('changed resource descriptor')
        rows.append(event)
    return dict(status='OUTPUT_BOUNDARY_RETURN_EVIDENCE', callbacks=len(rows),
                supported_callbacks=sum(r['batch_supported'] and r['lifecycle_match'] for r in rows),
                rejected_callbacks=sum(not r['batch_supported'] or not r['lifecycle_match'] for r in rows),
                calls=rows, scope='Boundary metadata alone never authorizes live capture; isolated copy tests require separate fence/artifact checks',
                capture_authorized=False, gpu_completion_verified=False, game_frame_capture_verified=False)


def controls(root):
    command_controls(root)
    reports = {}
    for provider in ('fsr3', 'fsr4'):
        reports[provider] = {}
        for suffix in ('observe', 'window', 'boundary_negative'):
            directory = root / (provider + '_' + suffix)
            manifest = json.loads((directory / 'manifest.json').read_text())
            negative = suffix == 'boundary_negative'
            if manifest.get('boundary_return_fixture') is not True or manifest.get('boundary_adversarial_fixture') is not negative:
                raise ValueError('missing native boundary fixture')
            if negative:
                validate_observe_pair(root / provider, directory)
                messages = (directory / 'd3d12_messages.log').read_text().splitlines()
                if (manifest.get('debug_warnings') != 2 or len(messages) != 2 or
                    not messages[0].startswith('2: ID3D12CommandList::ResourceBarrier: Called on the same subresource(s)') or
                    not messages[1].startswith('2: ID3D12CommandList::ResourceBarrier: Begin and End split barrier called on the same subresource(s)') or
                    any('inefficient and likely unintentional' not in m for m in messages)):
                    raise ValueError('unexpected negative-fixture debug messages')
            elif manifest.get('debug_warnings') != 0:
                raise ValueError('normal boundary path must be warning-free')
            report = analyze(read_events(directory / 'session.jsonl'))
            expected = (4, 2, 2) if negative else ((2, 2, 0) if suffix == 'window' else (4, 4, 0))
            if tuple(report[k] for k in ('callbacks', 'supported_callbacks', 'rejected_callbacks')) != expected:
                raise ValueError('boundary positive/negative control counts')
            if negative:
                rejected = [e for e in report['calls'] if not e['batch_supported']]
                if [e['sample'] for e in rejected] != [2, 3] or any(e['target_transitions'] != 2 for e in rejected):
                    raise ValueError('multi/split batch rejection missing')
            reports[provider][suffix] = report
        baseline = analyze_provider(root / provider)
        for report_name, suffix, fixture_field, folder, write_back in (
                ('hook_copy', '_boundary_hook', 'boundary_hook_capture_fixture', 'output_boundary', False),
                ('hook_roundtrip', '_roundtrip_hook', 'boundary_roundtrip_hook_fixture', 'output_roundtrip', True)):
            directory = root / (provider + suffix)
            captured = analyze_provider(directory)
            manifest = json.loads((directory / 'manifest.json').read_text())
            if (manifest.get(fixture_field) is not True or manifest.get('debug_warnings') != 0 or
                    manifest.get('session_poll_status') != 1 or manifest.get('capture_early_poll_checks') != 1 or
                    baseline['output_sha256'] != captured['output_sha256']):
                raise ValueError('hook on/off/fence fixture')
            events = read_events(directory / 'session.jsonl')
            report = analyze(events)
            if report['callbacks'] != 1 or report['supported_callbacks'] != 1 or any(e.get('event') == 'failure' for e in events):
                raise ValueError('hook boundary evidence')
            expected_events = ('observation_idle', 'boundary_output_arm', 'output_boundary_return', 'boundary_output_recorded', 'boundary_output_submitted', 'boundary_output_complete')
            selected = [e for e in events if e.get('event') in expected_events]
            if (tuple(e['event'] for e in selected) != expected_events or selected[-1].get('fence_completed') is not True or
                    selected[-1].get('game_frame_capture_verified') is not False or selected[-1].get('nr_verified') is not False or
                    any(e.get('write_back_performed') is not write_back for e in (selected[1], selected[3], selected[4], selected[5]))):
                raise ValueError('hook lifecycle order/mode')
            submitted = selected[-2]
            if (submitted.get('submitted_identity_matches') != 1 or
                    not isinstance(submitted.get('identity_aliases'), int) or submitted['identity_aliases'] < 1 or
                    not isinstance(submitted.get('closed_identity_aliases'), int) or submitted['closed_identity_aliases'] < 1 or
                    type(submitted.get('raw_pointer_match')) is not bool):
                raise ValueError('hook canonical submission proof')
            raw = (directory / folder / 'boundary_output.raw').read_bytes()
            metadata = json.loads((directory / folder / 'metadata.json').read_text())
            if raw != (root / provider / 'output_0.raw').read_bytes() or len(raw) != selected[-1].get('raw_bytes'):
                raise ValueError('hook output bytes')
            if (metadata.get('raw_bytes') != len(raw) or metadata.get('fence_completed') is not True or
                    metadata.get('write_back_performed') is not write_back or metadata.get('replacement_pixels_supplied') is not False or
                    metadata.get('game_frame_capture_verified') is not False or metadata.get('nr_verified') is not False):
                raise ValueError('hook output metadata')
            reports[provider][report_name] = dict(status='ISOLATED_BOUNDARY_HOOK_ROUNDTRIP_PASS' if write_back else 'ISOLATED_BOUNDARY_HOOK_COPY_PASS',
                                                  raw_bytes=len(raw), on_off_output_equal=True, write_back_performed=write_back,
                                                  early_poll_blocked=True, game_frame_capture_verified=False)
        patch_directory = root / (provider + '_patch_hook')
        if patch_directory.exists():
            patch = analyze_patch(patch_directory, live=False)
            manifest = json.loads((patch_directory / 'manifest.json').read_text())
            before = (patch_directory / 'output_patch' / 'boundary_before.raw').read_bytes()
            after = (patch_directory / 'output_patch' / 'boundary_output.raw').read_bytes()
            if (manifest.get('boundary_patch_hook_fixture') is not True or manifest.get('debug_warnings') != 0 or
                    manifest.get('session_poll_status') != 1 or manifest.get('capture_early_poll_checks') != 1 or
                    before != (root / provider / 'output_0.raw').read_bytes() or
                    after != (patch_directory / 'output_0.raw').read_bytes()):
                raise ValueError('hook patch fixture')
            reports[provider]['hook_patch'] = dict(patch, status='ISOLATED_BOUNDARY_HOOK_PATCH_PASS',
                                                   early_poll_blocked=True, baseline_before_equal=True,
                                                   downstream_after_equal=True)
        filter_directory = root / (provider + '_filter_hook')
        if filter_directory.exists():
            filtered = analyze_filter(filter_directory, live=False)
            manifest = json.loads((filter_directory / 'manifest.json').read_text())
            before = (filter_directory / 'output_filter' / 'boundary_before.raw').read_bytes()
            after = (filter_directory / 'output_filter' / 'boundary_output.raw').read_bytes()
            if (manifest.get('boundary_filter_hook_fixture') is not True or manifest.get('debug_warnings') != 0 or
                    manifest.get('debug_errors') != 0 or manifest.get('session_poll_status') != 1 or
                    manifest.get('capture_early_poll_checks') != 0 or manifest.get('dispatches') != 1 or
                    before != (root / provider / 'output_0.raw').read_bytes() or
                    after != (filter_directory / 'output_0.raw').read_bytes()):
                raise ValueError('hook filter fixture')
            reports[provider]['hook_filter'] = dict(
                filtered, status='ISOLATED_BOUNDARY_HOOK_FILTER_PASS', baseline_before_equal=True,
                downstream_after_equal=True)
        network_directory = root / (provider + '_network_hook')
        if network_directory.exists():
            networked = analyze_network(network_directory, live=False)
            manifest = json.loads((network_directory / 'manifest.json').read_text())
            before = (network_directory / 'output_network' / 'boundary_before.raw').read_bytes()
            after = (network_directory / 'output_network' / 'boundary_output.raw').read_bytes()
            calibration_exact = not filter_directory.exists() or after == (
                filter_directory / 'output_filter' / 'boundary_output.raw').read_bytes()
            if (manifest.get('boundary_network_hook_fixture') is not True or manifest.get('debug_warnings') != 0 or
                    manifest.get('debug_errors') != 0 or manifest.get('session_poll_status') != 1 or
                    manifest.get('capture_early_poll_checks') != 0 or manifest.get('dispatches') != 1 or
                    before != (root / provider / 'output_0.raw').read_bytes() or
                    after != (network_directory / 'output_0.raw').read_bytes() or not calibration_exact):
                raise ValueError('hook network fixture')
            reports[provider]['hook_network'] = dict(
                networked, status='ISOLATED_BOUNDARY_HOOK_NETWORK_PASS', baseline_before_equal=True,
                downstream_after_equal=True, calibration_filter_exact=calibration_exact)
    return dict(status='OUTPUT_BOUNDARY_RETURN_CONTROL_PASS', providers=reports,
                capture_authorized=False, game_frame_capture_verified=False)


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
