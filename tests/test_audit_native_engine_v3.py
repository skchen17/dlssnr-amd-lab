from copy import deepcopy

from scripts.audit_native_engine_v3 import audit


def accepted_report():
    return {
        'resolution': [1920, 1080],
        'performance': {'pure_network_gpu_median_ms': 32.9,
                        'pure_network_gpu_p95_ms': 33.5, 'warmup_frames': 5,
                        'abba_groups': 2, 'samples_per_process': 12},
        'graph': {'kernel_nodes': 512, 'pytorch_or_library_hot_path_nodes': 0},
        'resources': {'owns_workspace': True, 'owns_weights': True,
                      'owns_graph_source': True, 'owns_graph_executable': True,
                      'deployment_ready': True,
                      'graph_references_external_allocations': False,
                      'frame_bindings_abi_version': 3,
                      'complete_native_topology': True,
                      'allocator_peak_bytes': 5_000_000_000,
                      'device_peak_bytes': 6_000_000_000},
        'model': {'package_version': 2, 'target_arch': 'gfx1201',
                  'original_model_sha256': 'A' * 64,
                  'selected_weight_segments': 153,
                  'resident_fp8_cross_stage': True},
        'temporal': {'contract_verified': True, 'original_branch': True,
                     'dataset_sequences': 24, 'dataset_frames': 768,
                     'split': {'train': 16, 'validation': 4, 'test': 4},
                     'reset_inherits_old_history': False},
        'quality': {'finite': True, 'sequence_psnr': 30.0,
                    'sequence_ssim': .95, 'temporal_regression_percent': 10.0},
        'game': {'additional_gpu_median_ms': 4.0, 'sdr_100_frames': True,
                 'hdr_100_frames': True, 'hud_unchanged': True,
                 'failure_fallback_verified': True,
                 'history_fence_lifetime_verified': True},
        'safety': {'rgp_counter_capture_used': False,
                   'new_gpu_reset_or_black_screen': False},
    }


def test_complete_boundary_report_passes():
    result = audit(accepted_report())
    assert result['accepted'] and not result['failures']


def test_current_transitional_graph_cannot_pass_by_reporting_fast_time():
    report = accepted_report()
    report['performance']['pure_network_gpu_median_ms'] = 20
    report['graph'].update(kernel_nodes=4653, pytorch_or_library_hot_path_nodes=1963)
    report['resources'].update(deployment_ready=False,
                               graph_references_external_allocations=True,
                               frame_bindings_abi_version=2)
    result = audit(report)
    assert not result['accepted']
    assert 'graph kernel_nodes<=512' in result['failures']
    assert 'pytorch_or_library_hot_path_nodes=0' in result['failures']


def test_quality_and_temporal_requirements_are_not_waived_by_native_ownership():
    report = deepcopy(accepted_report())
    report['quality']['sequence_psnr'] = 29.99
    report['temporal']['contract_verified'] = False
    result = audit(report)
    assert not result['accepted']
    assert any('PSNR' in item for item in result['failures'])
    assert any('temporal contract' in item for item in result['failures'])
