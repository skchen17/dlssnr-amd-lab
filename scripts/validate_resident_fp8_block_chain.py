"""Supervised gate for FFN E4M3 producer -> FP8 QKV consumer block chains."""
from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from validate_rocm_lifecycle import save, supervise


def metrics(actual, expected):
    import torch
    delta = actual.float() - expected.float()
    unequal = actual.view(torch.int16) != expected.view(torch.int16)
    rmse = float(delta.square().mean().sqrt())
    denominator = float(expected.float().square().mean().sqrt())
    return {'bitwise_exact': not bool(unequal.any()),
            'different_components': int(unequal.sum()),
            'different_fraction': float(unequal.float().mean()),
            'max_absolute_error': float(delta.abs().max()), 'rmse': rmse,
            'nrmse': rmse / denominator if denominator else (0.0 if rmse == 0 else None)}


def exercise(args):
    import torch
    from native_execution_policy import execution_policy
    from native_matrix_fusion import matrix_fusion
    from native_model_package import load_package
    from native_packed_swin import RecoveredPackedSwin

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm required')
    props = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:
        raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(2_500_000_000 / props.total_memory)
    torch.manual_seed(9100 + args.channels)
    records, _, _ = load_package('local_models/native_single_color_v1',
        'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    record = {64: 6, 128: 10, 256: 16}[args.channels]
    model = RecoveredPackedSwin(records[(record, 0)], record_kind=f'swin{args.channels}').eval().cuda().requires_grad_(False)
    packed = (torch.randn(args.windows, 64 * args.channels, device='cuda', dtype=torch.float16) * .1).contiguous()
    original = packed.clone()
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())

    def wait():
        event = torch.cuda.Event(); event.record(stream); deadline = time.monotonic() + 10
        while not event.query():
            if time.monotonic() >= deadline:
                save(args.output / 'gpu_timeout.json', {'resources_retained': True})
                while True: time.sleep(1)
            time.sleep(.001)

    def timed(call):
        samples = []
        for _ in range(args.iterations):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(stream); value = call(); end.record(stream); wait()
            samples.append(float(begin.elapsed_time(end))); del value
        return samples

    family = f'c{args.channels}_ffn_attention_fp8a'
    with torch.no_grad(), torch.cuda.stream(stream), execution_policy('native_fp16'):
        expected = model(packed); wait()
        reference_times = timed(lambda: model(packed))
        with matrix_fusion(args.dll, 'wmma_fp16', modules=(family,), waves=1) as operator:
            actual = model(packed); wait(); error = metrics(actual, expected); fixed = actual.clone()
            repeated = model(packed); wait(); repeat = metrics(repeated, fixed)
            candidate_times = timed(lambda: model(packed))
            authored_launches = operator.launches / (args.iterations + 2)
        input_unchanged = bool(torch.equal(packed, original))
        finite = bool(torch.isfinite(actual).all())
    return {'checks_pass': repeat['bitwise_exact'] and finite and input_unchanged,
            'channels': args.channels, 'windows': args.windows, 'error_vs_reference': error,
            'repeat_error': repeat, 'finite': finite, 'input_unchanged': input_unchanged,
            'reference_gpu_event_ms': reference_times, 'candidate_gpu_event_ms': candidate_times,
            'reference_median_gpu_event_ms': statistics.median(reference_times),
            'candidate_median_gpu_event_ms': statistics.median(candidate_times),
            'speedup': statistics.median(reference_times) / statistics.median(candidate_times),
            'authored_launches_per_block': authored_launches,
            'scope': 'FFN dual FP16/E4M3 epilogue -> resident E4M3 hipBLASLt QKV -> native norm -> library attention/projection',
            'accuracy_track': 'approximate_deterministic', 'default_promoted': False,
            'device': props.name, 'architecture': props.gcnArchName}


def child(args):
    def phase(name):
        with (args.output / 'phases.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase': name, 'pid': os.getpid()}) + '\n')
    atexit.register(phase, 'python_atexit'); phase('child_enter')
    import torch, traceback
    try:
        result = exercise(args); phase('gpu_work_complete')
    except BaseException as error:
        with (args.output / 'traceback.log').open('x', encoding='utf-8') as trace:
            traceback.print_exc(file=trace)
        phase('gpu_work_failed')
        result = {'checks_pass': False, 'error_type': type(error).__name__, 'error': str(error),
                  'gpu_work_cancelled': False, 'automatic_retry': False}
    gc.collect(); torch.cuda.empty_cache()
    if hasattr(torch._C, '_cuda_clearCublasWorkspaces'): torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    result['allocated_after_release_bytes'] = torch.cuda.memory_allocated()
    result['reserved_after_release_bytes'] = torch.cuda.memory_reserved()
    result['checks_pass'] &= result['allocated_after_release_bytes'] == result['reserved_after_release_bytes'] == 0
    phase('resources_released'); save(args.output / 'child.json', result)
    return result['checks_pass']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--channels', type=int, choices=(64, 128, 256), required=True)
    parser.add_argument('--windows', type=int, choices=(1, 16, 144), default=1)
    parser.add_argument('--iterations', type=int, choices=(1, 12), default=1)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--child', action='store_true')
    args = parser.parse_args(); require_gpu_tests_enabled('resident FP8 block-chain gate')
    if args.child: return child(args)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--dll', str(args.dll.resolve()),
        '--channels', str(args.channels), '--windows', str(args.windows), '--iterations', str(args.iterations),
        '--output', str(args.output.resolve())]
    return supervise(command, args.output, timeout=120)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 2)
