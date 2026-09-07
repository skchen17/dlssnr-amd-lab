import pytest
from scripts.report_native_operation_timings import summarize, dot_names, short_name, time_stats


def sample(ms=2.0, name='fused'):
    return {'nodes': [{'index': 0, 'scope': 'C64/block5', 'kind': 'kernel', 'name': name, 'event_ms': ms}]}


def test_median_and_scope():
    result = summarize([sample(1), sample(3)])
    assert result['nodes'][0]['median_ms'] == 2
    assert result['aggregate'][0]['stage'] == 'C64'
    assert result['aggregate'][0]['nodes_per_frame'] == 1


def test_reject_topology_change():
    with pytest.raises(ValueError, match='topology'):
        summarize([sample(), sample(name='other')])


def test_invalid_event():
    with pytest.raises(ValueError, match='duration'):
        summarize([sample(float('nan'))])


def test_dot_dependency_order():
    text = '\n'.join(['"graph_0_node_9"[label="9\nsecond\n"];',
                      '"graph_0_node_2"[label="2\nfirst\n"];',
                      '"graph_0_node_2" -> "graph_0_node_9";'])
    assert dot_names(text) == ['first', 'second']


def test_demangle_retains_channel_specialization():
    assert short_name('_Z28stage_qkv_full_fused_lut_fp8ILi32ELb0EEv') == 'stage_qkv_full_fused_lut_fp8<32>'


def test_nearest_rank_p95():
    assert time_stats(list(range(1, 49)))['p95_nearest_rank_ms'] == 46
