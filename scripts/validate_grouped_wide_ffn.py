"""Safe correctness/performance gate for low-LDS C64/C128/C256 grouped FFN.

The candidate fuses only each independent C -> 128 -> 32 head.  The final
cross-head C -> C mix stays in the ROCm library GEMM path.  This validator does
not use RGP counters and never promotes a candidate to deployment defaults.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from validate_rocm_lifecycle import save, supervise


# This is a separate, explicit approximate track.  The activation and mix
# weights already sit on recovered E4M3 boundaries; the remaining difference
# is the FP8-WMMA reduction order versus rocBLAS.  Strict-track promotion still
# requires bitwise equality and this tolerance is never applied to FP16 edges.
FP8A_TOLERANCE = {'max_absolute_error': 0.001, 'nrmse': 0.0005}


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest().upper()


def compare(actual, expected):
    import torch
    delta = actual.float() - expected.float()
    unequal = actual.view(torch.int16) != expected.view(torch.int16)
    positions = unequal.flatten().nonzero()
    rmse = float(torch.sqrt(torch.mean(delta.square())).item())
    denominator = float(torch.sqrt(torch.mean(expected.float().square())).item())
    first = int(positions[0]) if len(positions) else None
    return {
        'bitwise_exact': first is None,
        'different_components': int(torch.count_nonzero(unequal).item()),
        'different_fraction': float(unequal.double().mean().item()),
        'first_different_flat': first,
        'max_absolute_error': float(delta.abs().max().item()),
        'rmse': rmse,
        'nrmse': rmse / denominator if denominator else (0.0 if rmse == 0 else None),
    }


def exercise(args, phase):
    import torch
    from native_execution_policy import execution_policy
    from native_matrix_fusion import matrix_fusion
    from native_model_package import load_package
    from native_packed_swin import RecoveredPackedSwin

    torch.set_num_threads(2)
    torch.manual_seed(3200 + args.channels)
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm required')
    props = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:
        raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(2_500_000_000 / props.total_memory)
    records, _, _ = load_package(
        'local_models/native_single_color_v1',
        'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    record = {64: 6, 128: 10, 256: 16}[args.channels]
    model = RecoveredPackedSwin(records[(record, 0)], record_kind=f'swin{args.channels}').eval().cuda().requires_grad_(False)
    count = args.windows * 64 * args.channels
    raw_guard = torch.full((count + 256,), 7.25, dtype=torch.float16, device='cuda')
    raw = raw_guard[128:-128].reshape(args.windows, 64 * args.channels)
    raw.copy_(torch.randn_like(raw) * .1)
    original = raw.clone()
    output_guard = torch.full((count + 256,), -5.75, dtype=torch.float16, device='cuda')
    output = output_guard[128:-128].reshape(args.windows, 64, args.channels)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def wait():
        event = torch.cuda.Event()
        event.record(stream)
        deadline = time.monotonic() + 10
        while not event.query():
            if time.monotonic() >= deadline:
                save(args.output / 'gpu_timeout.json', {'resources_retained': True})
                while True:
                    time.sleep(1)
            time.sleep(.001)

    def timed(call):
        samples = []
        for _ in range(args.iterations):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(stream)
            call()
            end.record(stream)
            wait()
            samples.append(float(begin.elapsed_time(end)))
        return samples

    with torch.no_grad(), torch.cuda.stream(stream), execution_policy('native_fp16'):
        expected = model.block.ffn(raw[:, model.a_index], raw[:, model.residual_index])
        wait()
        modules = (f'c{args.channels}_ffn_grouped_fp8a_lib' if args.resident_fp8_library_mix else
                   f'c{args.channels}_ffn_grouped_fp8a' if args.resident_fp8_activations else
                   f'c{args.channels}_ffn_grouped_fp8w' if args.resident_fp8_weights else
                   f'c{args.channels}_ffn_grouped',)
        torch.cuda.reset_peak_memory_stats()
        with matrix_fusion(args.dll, args.profile, modules=modules, waves=1) as operator:
            operator.wide_group_ffn(model, raw, output)
            wait()
            base = compare(output, expected)
            fixed = output.clone()
            operator.wide_group_ffn(model, raw, output)
            wait()
            repeat = compare(output, fixed)
            raw.add_(.03125)
            operator.wide_group_ffn(model, raw, output)
            wait()
            changed = compare(output, model.block.ffn(raw[:, model.a_index], raw[:, model.residual_index]))
            raw.copy_(original)
            wait()
            reference_times = timed(lambda: model.block.ffn(raw[:, model.a_index], raw[:, model.residual_index]))
            candidate_times = timed(lambda: operator.wide_group_ffn(model, raw, output))
            final_hash = digest(output)
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            launches_per_call = operator.launches / (args.iterations + 3)
        guards = bool((raw_guard[:128] == 7.25).all() and (raw_guard[-128:] == 7.25).all()
                      and (output_guard[:128] == -5.75).all() and (output_guard[-128:] == -5.75).all())
        finite = bool(torch.isfinite(output).all())
        wait()
    tolerance=dict(FP8A_TOLERANCE);approximate=args.resident_fp8_activations or args.resident_fp8_library_mix
    def accepted(error):
        return error['bitwise_exact'] or (approximate and
            error['max_absolute_error']<=tolerance['max_absolute_error'] and
            error['nrmse'] is not None and error['nrmse']<=tolerance['nrmse'])
    checks = accepted(base) and repeat['bitwise_exact'] and accepted(changed) and guards and finite
    return {
        'checks_pass': checks,
        'channels': args.channels,
        'windows': args.windows,
        'profile': args.profile,
        'resident_fp8_weights': args.resident_fp8_weights,
        'resident_fp8_activations': args.resident_fp8_activations,
        'resident_fp8_library_mix': args.resident_fp8_library_mix,
        'candidate_scope': ('E4M3 grouped activation is resident into ROCm library FP8 GEMM; FP16 residual preserved'
                            if args.resident_fp8_library_mix else
                            'E4M3 grouped activation is resident across the two authored kernels; FP8 WMMA CxC mix; FP16 residual preserved'
                            if args.resident_fp8_activations else
                            'one workgroup per (window,head,16-token tile); 4 KiB local hidden; library CxC mix remains separate'),
        'grid_workgroups': args.windows * (args.channels // 32) * 4,
        'threads_per_workgroup': 32,
        'static_lds_bytes_per_workgroup': 128 * 16 * 2,
        'logical_candidate_dispatches': 2 if args.resident_fp8_activations else 3,
        'logical_candidate_stages': (['group_expand_activation_contract_to_resident_e4m3_and_seed','rocm_library_fp8_mix','fp16_residual_add']
            if args.resident_fp8_library_mix else ['group_expand_activation_contract_to_resident_e4m3_and_seed','fp8_wmma_mix_and_residual_add']
            if args.resident_fp8_activations else ['group_expand_activation_contract_quantize_and_seed', 'library_mix_gemm', 'residual_add']),
        'custom_kernel_launches_per_call': launches_per_call,
        'temporary_workspace_bytes': args.windows*64*args.channels*(3 if approximate else 4),
        'reference_tolerance': tolerance if approximate else None,
        'correctness_track': 'explicit_approximate_original_e4m3_boundary_v1' if approximate else 'strict_bitwise',
        'tolerance_rationale': ('Only the recovered E4M3 grouped activation/mix boundary changes reduction order; '
                                'FP16 residual remains outside FP8 WMMA.' if approximate else None),
        'tolerance_used': approximate and not base['bitwise_exact'],
        'reference_gpu_event_ms': reference_times,
        'candidate_gpu_event_ms': candidate_times,
        'reference_median_gpu_event_ms': statistics.median(reference_times),
        'candidate_median_gpu_event_ms': statistics.median(candidate_times),
        'speedup': statistics.median(reference_times) / statistics.median(candidate_times),
        'errors': base,
        'repeat_errors': repeat,
        'changed_input_errors': changed,
        'guard_pass': guards,
        'finite': finite,
        'sha256': final_hash,
        'peak_allocated_bytes': peak_allocated,
        'peak_reserved_bytes': peak_reserved,
        'device': props.name,
        'architecture': props.gcnArchName,
        'weights_modified': False,
        'default_promoted': False,
    }


def child(args):
    def phase(name):
        with (args.output / 'phases.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase': name, 'pid': os.getpid(), 'monotonic': time.monotonic()}) + '\n')
    atexit.register(phase, 'python_atexit')
    phase('child_enter')
    result = exercise(args, phase)
    phase('gpu_work_complete')
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    result['allocated_after_release_bytes'] = torch.cuda.memory_allocated()
    result['reserved_after_release_bytes'] = torch.cuda.memory_reserved()
    result['checks_pass'] &= result['allocated_after_release_bytes'] == result['reserved_after_release_bytes'] == 0
    phase('resources_released')
    save(args.output / 'child.json', result)
    return result['checks_pass']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--channels', type=int, choices=(64, 128, 256), required=True)
    parser.add_argument('--windows', type=int, choices=(1, 16, 144, 256, 753, 768), required=True)
    parser.add_argument('--iterations', type=int, choices=(1, 12), default=1)
    parser.add_argument('--profile', choices=('wmma_fp16', 'wmma_fp8'), default='wmma_fp16')
    parser.add_argument('--resident-fp8-weights', action='store_true')
    parser.add_argument('--resident-fp8-activations', action='store_true')
    parser.add_argument('--resident-fp8-library-mix', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.resident_fp8_weights and args.profile!='wmma_fp8':parser.error('--resident-fp8-weights requires --profile wmma_fp8')
    if sum(bool(value) for value in (args.resident_fp8_weights,args.resident_fp8_activations,args.resident_fp8_library_mix))>1:parser.error('select one resident prototype')
    require_gpu_tests_enabled('low-LDS grouped wide FFN GPU gate')
    if args.child:
        return child(args)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--dll', str(args.dll.resolve()),
               '--channels', str(args.channels), '--windows', str(args.windows), '--iterations', str(args.iterations),
               '--profile', args.profile, '--output', str(args.output.resolve())]
    if args.resident_fp8_weights:command.append('--resident-fp8-weights')
    if args.resident_fp8_activations:command.append('--resident-fp8-activations')
    if args.resident_fp8_library_mix:command.append('--resident-fp8-library-mix')
    return supervise(command, args.output, timeout=120)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 2)
