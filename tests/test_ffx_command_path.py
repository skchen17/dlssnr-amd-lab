import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_ffx_command_path import analyze, COUNTERS


def sample():
    return [dict(event='command_path_hooks', installed=True, enhanced_supported=True),
            dict(event='candidate', window=1, list='l', generation=2, sample=3, resources=[]),
            dict(event='command_path', window=1, list='l', generation=2, sample=3,
                 original_status=0, list_table_matches=True, capture_authorized=False,
                 **{k: 0 for k in COUNTERS})]


def test_zero_is_not_no_gpu_work_or_capture_permission():
    report = analyze(sample())
    assert report['zero_observed_command_calls'] == 1
    assert not report['capture_authorized']
    assert not report['gpu_execution_verified']


def test_counters_are_not_unique_dispatch_claims():
    events = sample()
    events[-1].update(gpu_dispatch_calls=14, other_interface_calls=7)
    assert analyze(events)['totals']['gpu_dispatch_calls'] == 14
    assert 'double-count' in analyze(events)['scope']


@pytest.mark.parametrize('key,value', [('window', 2), ('generation', 3), ('sample', 4), ('list', 'other')])
def test_cross_identity_rejected(key, value):
    events = sample()
    events[-1][key] = value
    with pytest.raises(ValueError, match='identity'):
        analyze(events)


@pytest.mark.parametrize('value', [-1, True, 1.0, 1000001])
def test_bad_counter_rejected(value):
    events = sample()
    events[-1]['gpu_dispatch_calls'] = value
    with pytest.raises(ValueError, match='counter'):
        analyze(events)


def test_duplicate_rejected():
    events = sample()
    with pytest.raises(ValueError, match='duplicate'):
        analyze(events + [events[-1]])


def test_missing_path_remains_visible():
    assert analyze(sample()[:-1])['unpaired_candidates'] == 1


def test_failed_hook_rejected():
    events = sample()
    events[0]['installed'] = False
    with pytest.raises(ValueError, match='hooks'):
        analyze(events)


def test_extra_interface_counter_cannot_exceed_all_calls():
    events = sample()
    events[-1]['other_interface_calls'] = 1
    with pytest.raises(ValueError, match='inconsistent'):
        analyze(events)


def test_return_and_table_failures_are_visible():
    events = sample()
    events[-1].update(original_status=1, list_table_matches=False)
    report = analyze(events)
    assert report['table_mismatches'] == report['nonzero_return_statuses'] == 1
