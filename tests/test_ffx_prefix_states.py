import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_ffx_prefix_states import analyze


def trace(kind, **fields):
    return dict(event='state_trace', kind=kind, window=1, **fields)


def resequence(events):
    counts = {}
    for event in events:
        if event['event'] == 'state_trace':
            window = event.get('window', 0)
            counts[window] = counts.get(window, 0) + 1
            event['sequence'] = counts[window]
    return events


def fixture():
    # The producer records on the CPU after FFX returns, but is submitted first.
    return resequence([
        trace('transition', list='fsr', generation=2, resource='r',
              before=64, after=192, flags=0, subresource=0),
        trace('dispatch_begin', list='fsr', generation=2, sample=1),
        dict(event='candidate', window=1, list='fsr', generation=2, sample=1,
             reset_observed=True, resources=[dict(role='depth', resource='r',
                 plane0_recorded_state=192, expected_state=192)]),
        trace('dispatch_end', list='fsr', generation=2, original_status=0),
        trace('transition', list='producer', generation=3, resource='r',
              before=16, after=64, flags=0, subresource=4294967295),
        trace('submit_begin', batch_id=7, queue='q', batch_count=2, lists=[
            dict(index=0, list='producer', generation=3, tracked=True),
            dict(index=1, list='fsr', generation=2, tracked=True)]),
        trace('submit_return', batch_id=7, queue='q', batch_count=2,
              batch_index=0, list='producer', generation=3),
        trace('submit_return', batch_id=7, queue='q', batch_count=2,
              batch_index=1, list='fsr', generation=2),
        trace('submit_end', batch_id=7, queue='q')])


def role(events):
    return analyze(resequence(events))['calls'][0]['roles'][0]


def test_batch_order_not_cpu_recording_order():
    result = analyze(fixture())
    assert result['associated_calls'] == 1
    value = result['calls'][0]['roles'][0]
    assert value['last_observed_prefix_state'] == 192
    assert value['origin']['list'] == 'fsr'
    assert value['transition_chain_gaps'] == []
    assert result['calls'][0]['fsr_reset_observed']
    for key in ('capture_authorized', 'gpu_state_verified', 'game_frame_capture_verified'):
        assert result[key] is False


def test_producer_transition_later_on_cpu_is_still_in_prefix():
    events = fixture()[1:]
    assert role(events)['last_observed_prefix_state'] == 64


@pytest.mark.parametrize('field,value', [('generation', 99), ('resource', 'other'),
                                        ('subresource', 1)])
def test_irrelevant_producer_transition_excluded(field, value):
    events = fixture()[1:]
    events[3][field] = value
    assert role(events)['last_observed_prefix_state'] is None


def test_transition_after_dispatch_on_fsr_list_excluded():
    events = fixture()
    events.insert(4, trace('transition', list='fsr', generation=2, resource='r',
                           before=192, after=8, flags=0, subresource=0))
    assert role(events)['last_observed_prefix_state'] == 192


@pytest.mark.parametrize('flags', [1, 2])
def test_split_barrier_not_treated_as_complete(flags):
    events = fixture()[1:]
    events[3]['flags'] = flags
    assert role(events)['last_observed_prefix_state'] is None


def test_chain_gap_is_reported_not_hidden():
    events = fixture()
    events[0]['before'] = 8
    result = role(events)
    assert result['transition_chain_gaps'][0]['prior_after'] == 64
    assert result['transition_chain_gaps'][0]['next_before'] == 8


def test_alias_uncertainty_withholds_state():
    events = fixture()
    events.insert(4, trace('alias', list='producer'))
    result = role(events)
    assert result['alias_uncertainty']
    assert result['last_observed_prefix_state'] is None


def test_enhanced_uncertainty_withholds_state():
    events = fixture()
    events.insert(4, trace('enhanced_state_unknown', list='producer', generation=3, group_count=1))
    result = role(events)
    assert result['enhanced_uncertainty']
    assert result['last_observed_prefix_state'] is None


def test_different_window_never_supplies_state():
    events = fixture()[1:]
    producer = events.pop(3)
    producer['window'] = 0
    events.insert(0, producer)
    assert role(events)['last_observed_prefix_state'] is None


def test_repeated_submission_is_not_guessed():
    events = fixture()
    second = copy.deepcopy(events[-4:])
    for event in second:
        event['batch_id'] = 8
    assert analyze(resequence(events + second))['associated_calls'] == 0


def test_incomplete_batch_not_associated():
    assert analyze(fixture()[:-1])['associated_calls'] == 0


def test_truncation_is_retained_even_with_complete_earlier_batch():
    result = analyze(fixture() + [dict(event='state_trace_limit', window=1)])
    assert result['trace_truncated']
    assert not result['gpu_state_verified']


def test_missing_reset_is_visible():
    events = fixture()
    del events[2]['reset_observed']
    assert not analyze(events)['calls'][0]['fsr_reset_observed']


def test_invalid_sequence_rejected():
    events = fixture()
    events[0]['sequence'] = 5
    with pytest.raises(ValueError, match='sequence'):
        analyze(events)
