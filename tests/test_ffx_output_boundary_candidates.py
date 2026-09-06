import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_ffx_output_boundary_candidates import analyze
from analyze_ffx_command_path import COUNTERS


def fixture():
    common = dict(window=1, list='l', generation=2)
    info = dict(resource='0x10', identity='0x20', dimension=3, format=10,
                width=2342, height=1317, mips=1, array_size=1,
                raw_matches=['output'], identity_matches=['output'])
    roles = [dict(role=r, declared_ffx_state=8, info=info if r == 'output' else dict(resource='0x0'))
             for r in ('color', 'depth', 'motion', 'exposure', 'reactive', 'transparency', 'output')]
    before = dict(event='resource_path', kind='resources_before', sequence=1, sample=1,
                  ffx_list='l', command_list='l', roles=roles, resource_fields_unchanged=True, **common)
    after = copy.deepcopy(before);after.update(kind='resources_after', sequence=2)
    path = dict(event='command_path', sample=1, original_status=0, list_table_matches=True, capture_authorized=False,
                **common, **{k: 0 for k in COUNTERS})
    def t(kind, sequence, **fields):
        return dict(event='state_trace', kind=kind, sequence=sequence, **common, **fields)
    return [before, t('dispatch_begin', 1, sample=1),
            dict(event='candidate', sample=1, context_known=False, reset_observed=True,
                 resources=[dict(role='output', resource='0x10', plane0_recorded_state=64, expected_state=8)], **common),
            after, path, t('dispatch_end', 2, original_status=0),
            t('transition', 3, resource='0x10', before=64, after=192, flags=0, subresource=4294967295),
            t('close', 4, succeeded=True),
            dict(event='command_path_hooks', installed=True, enhanced_supported=True),
            t('submit_begin', 5, batch_id=1, queue='q', batch_count=1,
              lists=[dict(index=0, list='l', generation=2, tracked=True)]),
            t('submit_return', 6, batch_id=1, queue='q', batch_count=1, batch_index=0),
            t('submit_end', 7, batch_id=1, queue='q')]


def test_explicit_boundary_does_not_need_invented_ffx_state_or_context():
    report = analyze(fixture())
    assert report['structurally_matched'] == 1
    assert not report['calls'][0]['context_known']
    assert report['calls'][0]['transition']['before'] == 64
    assert all(report[k] is False for k in ('live_copy_ready', 'capture_authorized', 'game_frame_capture_verified', 'dlss_nr_verified'))


@pytest.mark.parametrize('field,value', [('flags', 1), ('flags', 2), ('subresource', 1),
                                        ('before', 16), ('after', 8), ('resource', '0x11')])
def test_unsupported_transition_rejected(field, value):
    events = fixture();events[6][field] = value
    assert analyze(events)['structurally_matched'] == 0


def test_missing_reset_rejected():
    events = fixture();events[2]['reset_observed'] = False
    assert analyze(events)['structurally_matched'] == 0


def test_failed_close_rejected():
    events = fixture();events[7]['succeeded'] = False
    assert analyze(events)['structurally_matched'] == 0


def test_missing_submission_end_rejected():
    assert analyze(fixture()[:-1])['structurally_matched'] == 0


def test_other_generation_transition_rejected():
    events = fixture();events[6]['generation'] = 9
    assert analyze(events)['structurally_matched'] == 0


def test_truncated_trace_never_promoted_to_live_ready():
    report = analyze(fixture() + [dict(event='state_trace_limit', window=1)])
    assert report['trace_truncated'] and not report['live_copy_ready']


def test_repeated_submission_rejected():
    events = fixture()
    for event in copy.deepcopy(events[-3:]):
        event['sequence'] += 3;event['batch_id'] = 2;events.append(event)
    assert analyze(events)['structurally_matched'] == 0
