import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from analyze_ffx_state_trace import summarize, correlate_dispatch_windows, correlate_batches


def sample():
    return [dict(event='context_create', context='c', epoch=4),
            dict(event='candidate', context='c', epoch=4, context_known=True,
                 recording_gate_ready=False, resources=[
                     dict(role='color', plane0_recorded_state=64, expected_state=192),
                     dict(role='depth', plane0_recorded_state=-1, expected_state=192)])]


def test_real_state_gap_is_not_a_capture_pass():
    r = summarize(sample())
    assert r['latest_context_verified_in_log']
    assert [i['reason'] for i in r['latest_resource_state_issues']] == ['mismatch', 'unknown']
    assert not r['game_frame_capture_verified']
    assert not r['cross_list_gpu_state_verified']


def test_destroy_invalidates_context():
    assert not summarize(sample()+[dict(event='context_destroy_begin', context='c')])['latest_context_verified_in_log']


def test_pointer_reuse_requires_epoch():
    assert not summarize(sample()+[dict(event='context_create', context='c', epoch=5)])['latest_context_verified_in_log']


def test_empty_is_not_verified():
    assert not summarize([])['latest_context_verified_in_log']


@pytest.mark.parametrize('seq', [[2], [1, 1], [1, 3], list(range(1, 1026))])
def test_broken_trace_sequence(seq):
    with pytest.raises(ValueError):
        summarize([dict(event='state_trace', kind='transition', sequence=i) for i in seq])


def test_limit_is_explicit():
    assert summarize([dict(event='state_trace_limit')])['trace_truncated']


def test_successful_flags_do_not_certify_pixels():
    events = sample()
    events[1]['recording_gate_ready'] = True
    r = summarize(events)
    assert r['ready_candidates'] == 1
    assert not r['game_frame_capture_verified']


def window_events():
    return [dict(event='state_trace', sequence=1, kind='transition', list='producer', generation=1, resource='depth', before=16, after=64),
            dict(event='state_trace', sequence=2, kind='dispatch_begin', list='fsr', generation=2, sample=1),
            dict(event='candidate', list='fsr', generation=2, sample=1, resources=[
                dict(role='depth', resource='depth', plane0_recorded_state=-1),
                dict(role='output', resource='out', plane0_recorded_state=64)]),
            dict(event='state_trace', sequence=3, kind='dispatch_end', list='fsr', generation=2, original_status=0),
            dict(event='state_trace', sequence=4, kind='transition', list='fsr', generation=2, resource='out', before=64, after=192)]


def test_external_transition_does_not_become_local_state():
    r = correlate_dispatch_windows(window_events())
    assert r['complete_observed_call_windows'] == 1
    depth, output = r['windows'][0]['roles']
    assert depth['local_state_at_boundary'] == -1
    assert depth['other_lists_with_transitions'] == ['producer']
    assert output['first_local_transition_after']['before'] == 64
    assert not r['cross_list_gpu_state_verified']


@pytest.mark.parametrize('index', [1, 2, 3])
def test_incomplete_window_ignored(index):
    events = window_events()
    del events[index]
    assert correlate_dispatch_windows(events)['complete_observed_call_windows'] == 0


def test_wrong_generation_not_associated():
    events = window_events()
    events[3]['generation'] = 1
    assert correlate_dispatch_windows(events)['complete_observed_call_windows'] == 0


def test_other_generation_transition_not_attributed_after_call():
    events = window_events()
    events[4]['generation'] = 3
    r = correlate_dispatch_windows(events)
    assert r['windows'][0]['roles'][1]['first_local_transition_after'] is None


def test_window_restart_allows_sequence_reset():
    events = [dict(event='state_trace', window=w, sequence=n, kind='watch')
              for w in (1, 2) for n in (1, 2)]
    assert summarize(events)['trace_events'] == 4


@pytest.mark.parametrize('windows', [[2, 1], [9], [-1]])
def test_invalid_window_identity(windows):
    with pytest.raises(ValueError, match='window identity'):
        summarize([dict(event='state_trace', window=w, sequence=1, kind='watch') for w in windows])


def test_no_cross_window_call_pairing():
    events = window_events()
    events[1]['window'] = 1
    events[2]['window'] = 1
    events[3]['window'] = 2
    assert correlate_dispatch_windows(events)['complete_observed_call_windows'] == 0


def batch_events():
    return [dict(event='state_trace', sequence=1, kind='submit_begin', window=1, batch_id=7,
                 queue='q', batch_count=1, lists=[dict(index=0, list='a', generation=3, tracked=True)]),
            dict(event='state_trace', sequence=2, kind='submit_return', window=1, batch_id=7,
                 queue='q', batch_count=1, batch_index=0, list='a', generation=3),
            dict(event='state_trace', sequence=3, kind='submit_end', window=1, batch_id=7, queue='q')]


def test_explicit_batch_matches_without_gpu_claim():
    r = correlate_batches(batch_events())
    assert len(r['complete_batches']) == 1
    assert not r['gpu_execution_order_verified']


@pytest.mark.parametrize('key,value', [('list', 'wrong'), ('generation', 4), ('queue', 'other'), ('batch_index', 1), ('batch_count', 2)])
def test_wrong_batch_member_rejected(key, value):
    events = batch_events()
    events[1][key] = value
    assert len(correlate_batches(events)['invalid_batches']) == 1


def test_duplicate_batch_return_rejected():
    events = batch_events()
    events.insert(2, dict(events[1]))
    assert len(correlate_batches(events)['invalid_batches']) == 1


def test_no_cross_window_batch_merge():
    events = batch_events()
    events[-1]['window'] = 2
    r = correlate_batches(events)
    assert len(r['incomplete_batches']) == 2
    assert not r['complete_batches']


def test_truncated_batch_is_incomplete():
    r = correlate_batches(batch_events()[:-1])
    assert len(r['incomplete_batches']) == 1
    assert not r['complete_batches']
