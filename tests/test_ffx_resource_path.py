import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_ffx_resource_path import analyze, ROLES
from analyze_ffx_command_path import COUNTERS


def info(pointer='0x10', identity='0x20'):
    return dict(resource=pointer, identity=identity, width=640, height=360, format=10,
                raw_matches=['color'] if pointer == '0x10' else [],
                identity_matches=['color'] if identity == '0x20' else [])


def fixture():
    roles = [dict(role=r, declared_ffx_state=4 if r == 'color' else 0,
                  info=info() if r == 'color' else dict(resource='0x0')) for r in sorted(ROLES)]
    common = dict(window=1, generation=2, sample=1)
    before = dict(event='resource_path', sequence=1, kind='resources_before', ffx_list='l',
                  command_list='l', roles=roles, resource_fields_unchanged=True, **common)
    barrier = dict(event='resource_path', sequence=2, kind='barrier', ffx_list='l', list='l',
                   type=0, flags=0, before=8, after=64, subresource=0, info=info(), **common)
    after = copy.deepcopy(before)
    after.update(sequence=3, kind='resources_after')
    path = dict(event='command_path', list='l', original_status=0, list_table_matches=True,
                capture_authorized=False, **common, **{k: 0 for k in COUNTERS})
    path.update(legacy_barrier_calls=1, legacy_barriers=1)
    return [dict(event='command_path_hooks', installed=True, enhanced_supported=True),
            before, dict(event='candidate', list='l', resources=[dict(role='color', resource='0x10')], **common),
            barrier, after, path]


def test_exact_pointer_evidence_not_state_permission():
    report = analyze(fixture())
    assert report['complete_calls'] == 1
    assert report['calls'][0]['raw_matched_barriers'] == 1
    assert not report['capture_authorized'] and not report['resource_state_verified']


def test_distinct_interface_same_identity():
    events = fixture()
    events[3]['info'] = info('0x30', '0x20')
    call = analyze(events)['calls'][0]
    assert call['raw_matched_barriers'] == 0
    assert call['identity_only_matched_barriers'] == 1


def test_same_dimensions_never_prove_identity():
    events = fixture()
    events[3]['info'] = info('0x30', '0x40')
    call = analyze(events)['calls'][0]
    assert call['raw_matched_barriers'] == call['identity_only_matched_barriers'] == 0


def test_fabricated_identity_match_rejected():
    events = fixture()
    events[3]['info'] = info('0x30', '0x40')
    events[3]['info']['identity_matches'] = ['color']
    with pytest.raises(ValueError, match='classification'):
        analyze(events)


def test_global_uav_null_resource_is_not_texture():
    events = fixture()
    events[3].update(type=2, info=dict(resource='0x0'))
    call = analyze(events)['calls'][0]
    assert call['types'] == {'uav': 1}
    assert call['null_resource_references'] == 1


def test_alias_both_resources_checked():
    events = fixture()
    del events[3]['info']
    events[3].update(type=1, before_info=dict(resource='0x0'), after_info=info('0x30', '0x20'))
    assert analyze(events)['calls'][0]['identity_only_matched_barriers'] == 1


def test_missing_after_is_incomplete():
    events = fixture()
    del events[4]
    report = analyze(events)
    assert report['complete_calls'] == 0 and len(report['incomplete_calls']) == 1


def test_missing_barrier_does_not_silently_pass():
    events = fixture()
    events[5]['legacy_barriers'] = 2
    assert not analyze(events)['calls'][0]['barrier_count_matches']


def test_resource_sequence_gap_rejected():
    events = fixture()
    events[3]['sequence'] = 10
    with pytest.raises(ValueError, match='sequence'):
        analyze(events)


def test_false_unchanged_claim_rejected():
    events = fixture()
    events[4]['command_list'] = 'other'
    with pytest.raises(ValueError, match='mutation'):
        analyze(events)


def test_descriptor_change_retained():
    events = fixture()
    events[4].update(command_list='other', resource_fields_unchanged=False)
    assert not analyze(events)['calls'][0]['resource_fields_unchanged']


def test_different_window_not_paired():
    events = fixture()
    events[4].update(window=2, sequence=1)
    report = analyze(events)
    assert report['complete_calls'] == 0


def test_truncation_remains_explicit():
    assert analyze(fixture() + [dict(event='resource_path_limit', window=1)])['detail_truncated']
