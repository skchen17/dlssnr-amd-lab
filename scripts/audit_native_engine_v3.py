"""Single fail-closed acceptance gate for the 1080p native temporal NR engine."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def audit(report: dict) -> dict:
    failures = []
    resolution = report.get('resolution')
    if resolution != [1920, 1080]: failures.append('resolution=1920x1080')
    perf = report.get('performance', {})
    median = perf.get('pure_network_gpu_median_ms')
    p95 = perf.get('pure_network_gpu_p95_ms')
    if not _finite(median) or median > 33: failures.append('pure_network_gpu_median_ms<=33')
    if not _finite(p95) or p95 <= 0: failures.append('finite positive pure_network_gpu_p95_ms')
    if perf.get('warmup_frames', 0) < 5 or perf.get('abba_groups', 0) < 2 or \
            perf.get('samples_per_process', 0) < 12:
        failures.append('5-warmup two-group 12-sample ABBA protocol')
    graph = report.get('graph', {})
    if graph.get('kernel_nodes', 1 << 30) > 512: failures.append('graph kernel_nodes<=512')
    if graph.get('pytorch_or_library_hot_path_nodes') != 0:
        failures.append('pytorch_or_library_hot_path_nodes=0')
    resources = report.get('resources', {})
    for key in ('owns_workspace', 'owns_weights', 'owns_graph_source',
                'owns_graph_executable', 'deployment_ready'):
        if resources.get(key) is not True: failures.append(f'resources.{key}=true')
    if resources.get('graph_references_external_allocations') is not False:
        failures.append('graph_references_external_allocations=false')
    if resources.get('frame_bindings_abi_version') != 3:
        failures.append('frame_bindings_abi_version=3')
    if resources.get('complete_native_topology') is not True:
        failures.append('complete_native_topology=true')
    if resources.get('allocator_peak_bytes', 1 << 63) > 5_000_000_000:
        failures.append('allocator_peak_bytes<=5GB')
    if resources.get('device_peak_bytes', 1 << 63) > 6_000_000_000:
        failures.append('device_peak_bytes<=6GB')
    model = report.get('model', {})
    if model.get('package_version') != 2 or model.get('target_arch') != 'gfx1201' or \
            not model.get('original_model_sha256') or model.get('selected_weight_segments', 0) <= 0:
        failures.append('verified model package v2/gfx1201/segmented weights')
    if model.get('resident_fp8_cross_stage') is not True:
        failures.append('resident_fp8_cross_stage=true')
    temporal = report.get('temporal', {})
    if temporal.get('contract_verified') is not True or temporal.get('original_branch') is not True:
        failures.append('original temporal contract/branch verified')
    if temporal.get('dataset_sequences', 0) < 24 or temporal.get('dataset_frames', 0) < 768 or \
            temporal.get('split') != {'train': 16, 'validation': 4, 'test': 4}:
        failures.append('24x32 temporal dataset with 16/4/4 split')
    if temporal.get('reset_inherits_old_history') is not False:
        failures.append('reset_inherits_old_history=false')
    quality = report.get('quality', {})
    if quality.get('finite') is not True or not _finite(quality.get('sequence_psnr')) or \
            quality.get('sequence_psnr') < 30 or not _finite(quality.get('sequence_ssim')) or \
            quality.get('sequence_ssim') < .95:
        failures.append('approximate quality PSNR>=30 SSIM>=0.95 finite')
    if not _finite(quality.get('temporal_regression_percent')) or \
            quality.get('temporal_regression_percent') > 10:
        failures.append('temporal regression<=10%')
    game = report.get('game', {})
    if not _finite(game.get('additional_gpu_median_ms')) or game.get('additional_gpu_median_ms') > 4:
        failures.append('game interop/temporal/compose<=4ms')
    for key in ('sdr_100_frames', 'hdr_100_frames', 'hud_unchanged',
                'failure_fallback_verified', 'history_fence_lifetime_verified'):
        if game.get(key) is not True: failures.append(f'game.{key}=true')
    safety = report.get('safety', {})
    if safety.get('rgp_counter_capture_used') is not False or \
            safety.get('new_gpu_reset_or_black_screen') is not False:
        failures.append('safe run: no RGP counters or new reset/black-screen')
    return {'schema': 1, 'milestone': '1080p_native_temporal_nr_33ms',
            'accepted': not failures, 'failures': failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = audit(json.loads(args.input.read_text(encoding='utf-8-sig')))
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['accepted'] else 2)


if __name__ == '__main__':
    main()
