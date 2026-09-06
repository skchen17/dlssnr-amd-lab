import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_ffx_boundary_return import analyze
from test_ffx_output_boundary_candidates import fixture as candidate_fixture


def fixture():
    events = candidate_fixture()
    events.append(dict(event='output_boundary_return', window=1, list='l', generation=2, sample=1,
                       resource='0x10', barrier_count=1, target_transitions=1, alias_barriers=0, unknown_barriers=0,
                       last_target_before=64, last_target_after=192, last_target_subresource=4294967295,
                       last_target_flags=0, batch_supported=True, lifecycle_match=True,
                       original_callback_returned=True, original_forward_calls=1, capture_authorized=False))
    return events


def test_return_evidence_not_gpu_or_capture_permission():
    report = analyze(fixture())
    assert report['supported_callbacks'] == 1
    assert report['capture_authorized'] is report['gpu_completion_verified'] is report['game_frame_capture_verified'] is False


@pytest.mark.parametrize('field,value', [('target_transitions', 2), ('alias_barriers', 1),
                                       ('unknown_barriers', 1), ('last_target_after', 8),
                                       ('last_target_flags', 1), ('last_target_subresource', 1)])
def test_false_supported_classification_rejected(field, value):
    events = fixture();events[-1].update(barrier_count=4);events[-1][field] = value
    with pytest.raises(ValueError, match='classification'):
        analyze(events)


@pytest.mark.parametrize('field,value', [('original_callback_returned', False), ('original_forward_calls', 2),
                                       ('capture_authorized', True), ('lifecycle_match', None)])
def test_invalid_forwarding_contract_rejected(field, value):
    events = fixture();events[-1][field] = value
    with pytest.raises(ValueError, match='contract'):
        analyze(events)


def test_multiple_output_transitions_retained_as_rejection():
    events = fixture();events[-1].update(barrier_count=2, target_transitions=2, batch_supported=False)
    report = analyze(events)
    assert report['supported_callbacks'] == 0 and report['rejected_callbacks'] == 1


def test_lifecycle_mismatch_cannot_support_boundary():
    events = fixture();events[-1]['lifecycle_match'] = False
    assert analyze(events)['supported_callbacks'] == 0


def test_reentered_duplicate_rejected():
    events = fixture();events.append(copy.deepcopy(events[-1]))
    with pytest.raises(ValueError, match='duplicate'):
        analyze(events)


def test_wrong_output_identity_rejected():
    events = fixture();events[-1]['resource'] = '0x99'
    with pytest.raises(ValueError, match='identity'):
        analyze(events)


def test_return_before_ffx_summary_rejected():
    events = fixture();event = events.pop();events.insert(0, event)
    with pytest.raises(ValueError, match='order'):
        analyze(events)


def test_impossible_barrier_counts_rejected():
    events = fixture();events[-1]['target_transitions'] = 2
    with pytest.raises(ValueError, match='count'):
        analyze(events)
