from copy import deepcopy
from scripts.assess_native_realtime import assess


def synthetic_schema_fixture():
    # Tests assessment logic ONLY. This fixture is not saved as game evidence.
    report = {'kind': 'native_full_frame_game_run', 'device': 'AMD Radeon RX 9070 XT',
              'native_graph_complete': True, 'pre_hud_output_visible': True,
              'resource_generation_and_fence_validation': True, 'dynamic_size_reset_validation': True,
              'game_settings_restored': True,
              **{k: False for k in ('cpu_neural_fallback', 'cpu_image_transfer_on_frame_path',
                                    'nvidia_runtime_dependency', 'independent_image_tiles',
                                    'silent_input_downscale', 'device_fault_observed')}}
    session = {'resolution': [3840, 2160], 'visual_review': 'PASS_NO_BLACK_FRAMES_SEAMS_FLICKER_OR_HUD_DAMAGE',
               'visual_evidence_paths': ['synthetic-test-only'],
               'frames': [{'frame_id': i, 'source_frame_id': i, 'network_applied': True, 'fallback_used': False,
                           'present_ms': i*16, 'inference_complete_host_ms': 5.} for i in range(1000)]}
    report['sessions'] = {'sdr': deepcopy(session), 'hdr': deepcopy(session)}
    return report


def test_realtime_gate_does_not_require_nvidia_quality():
    result = assess(synthetic_schema_fixture())
    assert result['pass']
    assert not result['nvidia_matching_quality_required'] and not result['dlss5_quality_equivalence_claimed']


def test_fast_subgraph_is_not_a_realtime_game_pass():
    assert not assess({'status': 'BOUNDED_ROCM_LIFECYCLE_PASS', 'pass': True})['pass']


def test_original_game_fps_without_neural_output_fails():
    report = synthetic_schema_fixture()
    report['sessions']['sdr']['frames'][12]['fallback_used'] = True
    assert not assess(report)['pass']
    report['sessions']['sdr']['frames'][12]['fallback_used'] = False
    report['sessions']['sdr']['frames'][12]['source_frame_id'] = 11
    assert not assess(report)['pass']


def test_actual_present_latency_not_just_inference_speed():
    report = synthetic_schema_fixture()
    for frame in report['sessions']['hdr']['frames']:
        frame['present_ms'] = frame['frame_id']*25
    assert not assess(report)['pass']


def test_missing_hdr_invalid_times_or_resize_shortcuts_fail():
    report = synthetic_schema_fixture()
    del report['sessions']['hdr']
    assert not assess(report)['pass']
    report = synthetic_schema_fixture()
    report['sessions']['sdr']['frames'][5]['present_ms'] = float('nan')
    assert not assess(report)['pass']
    report = synthetic_schema_fixture()
    report['silent_input_downscale'] = True
    assert not assess(report)['pass']
