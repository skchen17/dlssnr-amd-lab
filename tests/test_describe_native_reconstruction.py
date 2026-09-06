import pytest
from scripts.describe_native_reconstruction import record_role, evidence_records, ORIGINAL_SHA


def test_unknown_record_does_not_inherit_a_shape_based_implementation():
    assert record_role('block999.layer0.layer')[1] == 'UNRESOLVED'
    assert record_role('untrusted')[1] == 'UNRESOLVED'


def test_gpu_and_cpu_evidence_are_distinct():
    assert 'ROCM' in record_role('block6.layer0.layer')[1]
    assert 'CPU_REFERENCE_GPU_PENDING' in record_role('block40.layer2.layer')[1]
    assert 'CPU_REFERENCE_GPU_PENDING' in record_role('block31.layer2.layer')[1]


def test_missing_graph_edges_stay_visible():
    assert 'CAPTURED_CPU_BOUNDARY_ONLY' in record_role('block48.layer0.layer')[1]
    assert 'CAPTURED_CPU_BOUNDARY_ONLY' in record_role('block14.layer0.layer')[1]
    assert 'OUTPUT_LAYOUT_UNRESOLVED' in record_role('block4.layer0.layer')[1]
    assert 'CAPTURED_CPU_BOUNDARY_ONLY' in record_role('block39.layer0.layer')[1]
    assert 'SINGLE_COLOR_BRANCH_ONLY' in record_role('block0.layer0.layer')[1]


def test_abi_placeholder_is_not_claimed_as_a_used_weight():
    assert 'PLACEHOLDER' in record_role('block31.layer3.layer')[1]
    assert 'COLOR_COMPOSITION' in record_role('block70.layer0.blend_scale')[1]


def test_gpu_evidence_requires_completed_process_not_just_numerics():
    with pytest.raises(ValueError):
        evidence_records({'pass': True, 'normal_exit': False})


def test_split_probe_does_not_promote_other_blocks_or_claim_quality():
    record = {'pass': True, 'normal_exit': True, 'host_timeout': False, 'returncode': 0,
              'probe': {'name': 'split512', 'checks_pass': True, 'allocated_after_release_bytes': 0,
                        'backend': 'pytorch_rocm', 'cpu_neural_fallback': False,
                        'original_model_sha256': ORIGINAL_SHA,
                        'record_hashes': {k: 'hash' for k in ('ffwd', 'ffwd_projection', 'attention', 'attention_projection')}}}
    evidence = evidence_records(record)
    assert len(evidence) == 4
    assert all(name.startswith('block40.') for name in evidence)
    assert all('SYNTHETIC_PROBE_ONLY' in item['status'] for item in evidence.values())


def decoder_report():
    return {'pass': True, 'normal_exit': True, 'host_timeout': False, 'returncode': 0,
            'probe': {'name': 'decoder_pyramid', 'checks_pass': True, 'allocated_after_release_bytes': 0,
                      'backend': 'pytorch_rocm', 'cpu_neural_fallback': False,
                      'original_model_sha256': ORIGINAL_SHA,
                      'encoder_skip_source': 'ORIGINAL_CAPTURED_EXTERNAL_INPUTS',
                      'external_input_hashes': {'entry': 'hash'},
                      'decoder_interior_rtx_substitution': False,
                      'stages': [{'block': b, 'original_record_sha256': str(b),
                                  'comparison_diagnostic': {'nonfinite': 0}} for b in range(48, 70)]}}


def test_decoder_evidence_does_not_promote_encoder_or_full_graph():
    evidence = evidence_records(decoder_report())
    assert set(evidence) == {f'block{b}.layer0.layer' for b in range(48, 70)}
    assert all('CAPTURED_DECODER_SUBGRAPH_ONLY' in e['status'] for e in evidence.values())


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'teacher', 'nonfinite'])
def test_decoder_evidence_fails_closed(failure):
    report = decoder_report()
    probe = report['probe']
    if failure == 'missing':
        probe['stages'].pop()
    elif failure == 'duplicate':
        probe['stages'][-1]['block'] = 48
    elif failure == 'teacher':
        probe['decoder_interior_rtx_substitution'] = True
    else:
        probe['stages'][0]['comparison_diagnostic']['nonfinite'] = 1
    with pytest.raises(ValueError):
        evidence_records(report)


def test_single_color_evidence_rejects_any_captured_skip_substitution():
    report=decoder_report()
    probe=report['probe']
    probe.update(name='whole_frame640',stages=[{'block':b} for b in range(71)],
                 all_encoder_skips_generated=False,rtx_intermediate_inputs=False,
                 single_color_tensor_chain_connected=True,composition='LEGACY_SDR_DIAGNOSTIC_ONLY')
    with pytest.raises(ValueError,match='teacher-fed'):
        evidence_records(report)
