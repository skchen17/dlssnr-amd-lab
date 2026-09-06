"""Supervised C64/C128 grouped-FFN WMMA accuracy and lifecycle gate.

This deliberately reports numerical differences instead of requiring an exact
hash.  A result is only a candidate record; it never changes deployment
defaults or silently falls back to the reference implementation.
"""
import argparse
import atexit
import gc
import hashlib
import sys
import time
from pathlib import Path

from validate_rocm_lifecycle import save, supervise


def metrics(actual, expected):
    import torch
    a, b = actual.detach().cpu(), expected.detach().cpu()
    difference = (a.float() - b.float()).double()
    unequal = a.view(torch.int16) != b.view(torch.int16)
    positions = unequal.flatten().nonzero()
    first = int(positions[0]) if len(positions) else None
    return {
        'max_abs': difference.abs().max().item(),
        'rmse': difference.square().mean().sqrt().item(),
        'different_fraction': unequal.double().mean().item(),
        'first_different_flat': first,
        'first_actual_bits': int(a.view(torch.int16).flatten()[first]) if first is not None else None,
        'first_expected_bits': int(b.view(torch.int16).flatten()[first]) if first is not None else None,
        'bitwise_exact': first is None,
    }


def exercise(args, phase):
    import torch
    from native_execution_policy import execution_policy
    from native_matrix_fusion import matrix_fusion
    from native_model_package import load_package
    from native_packed_swin import RecoveredPackedSwin

    torch.set_num_threads(2)
    torch.manual_seed(1201 + args.channels)
    if not torch.version.hip or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('gfx1201 ROCm device required')
    torch.cuda.set_per_process_memory_fraction(2_500_000_000 / torch.cuda.get_device_properties(0).total_memory)
    records, _, _ = load_package(
        'local_models/native_single_color_v1',
        'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    block = {64: 6, 128: 10, 256: 16}[args.channels]
    model = RecoveredPackedSwin(records[(block, 0)], record_kind=f'swin{args.channels}').eval().cuda().requires_grad_(False)
    raw_guard = torch.full((args.batch * 64 * args.channels + 256,), 7.25, dtype=torch.float16, device='cuda')
    raw = raw_guard[128:-128].reshape(args.batch, 64 * args.channels)
    raw.copy_(torch.randn_like(raw) * .1)
    original = raw.clone()
    output_guard = torch.full((args.batch * 64 * args.channels + 256,), -5.75, dtype=torch.float16, device='cuda')
    output = output_guard[128:-128].reshape(args.batch, 64, args.channels)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def wait():
        event = torch.cuda.Event()
        event.record()
        deadline = time.monotonic() + 10
        while not event.query():
            if time.monotonic() >= deadline:
                save(args.output / 'gpu_timeout.json', {'resources_retained': True})
                while True:
                    time.sleep(1)
            time.sleep(.001)

    rows = []
    with torch.no_grad(), torch.cuda.stream(stream), execution_policy('native_fp16'):
        expected = model.block.ffn(raw[:, model.a_index], raw[:, model.residual_index])
        wait()
        for profile in ('wmma_fp16', 'wmma_fp8'):
            for waves in ((1, 2) if args.channels == 256 else (1, 2, 4)):
                with matrix_fusion(args.dll, profile, modules=(f'c{args.channels}_ffn',), waves=waves) as operator:
                    operator.wide_ffn(model, raw, output)
                    wait()
                    base = metrics(output, expected)
                    if not bool(torch.isfinite(output).all()):
                        raise RuntimeError(f'nonfinite candidate profile={profile} waves={waves}')
                    fixed = output.clone()
                    operator.wide_ffn(model, raw, output)
                    wait()
                    repeat = metrics(output, fixed)
                    if not repeat['bitwise_exact']:
                        raise RuntimeError(f'nondeterministic candidate profile={profile} waves={waves}')
                    if not torch.equal(raw, original):
                        raise RuntimeError('input corruption')
                    guards = bool((raw_guard[:128] == 7.25).all() and (raw_guard[-128:] == 7.25).all()
                                  and (output_guard[:128] == -5.75).all() and (output_guard[-128:] == -5.75).all())
                    if not guards:
                        raise RuntimeError('guard corruption')
                    timings = {}
                    for mode in ('reference', 'candidate'):
                        wait()
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            if mode == 'reference':
                                value = model.block.ffn(raw[:, model.a_index], raw[:, model.residual_index])
                            else:
                                operator.wide_ffn(model, raw, output)
                        wait()
                        timings[mode] = (time.perf_counter() - start) * 1000 / args.iterations
                    raw.add_(.03125)
                    operator.wide_ffn(model, raw, output)
                    wait()
                    changed_expected = model.block.ffn(raw[:, model.a_index], raw[:, model.residual_index])
                    changed = metrics(output, changed_expected)
                    raw.copy_(original)
                    ffn = model.block.ffn
                    old = tuple(parameter.clone() for parameter in (ffn.expand, ffn.contract, ffn.mix))
                    ffn.expand.add_(.001)
                    ffn.contract.sub_(.001)
                    ffn.mix.add_(.001)
                    operator.wide_ffn(model, raw, output)
                    wait()
                    mutation_expected = ffn(raw[:, model.a_index], raw[:, model.residual_index])
                    mutation = metrics(output, mutation_expected)
                    ffn.expand.copy_(old[0])
                    ffn.contract.copy_(old[1])
                    ffn.mix.copy_(old[2])
                    wait()
                    rows.append({
                        'profile': profile,
                        'waves': waves,
                        'errors': base,
                        'changed_errors': changed,
                        'weight_mutation_errors': mutation,
                        'repeat_errors': repeat,
                        'guard_pass': guards,
                        'host_ms_per_call': timings,
                        'candidate_kernel_nodes': 1,
                        'reference_logical_gemms': 3,
                    })
                    phase(f'{profile}_{waves}_measured')
                    del fixed, changed_expected, mutation_expected, old
    del model, raw_guard, raw, original, output_guard, output, expected, stream, value
    return {
        'checks_pass': True,
        'channels': args.channels,
        'batch': args.batch,
        'iterations': args.iterations,
        'rows': rows,
        'scope': 'actual frozen grouped-FFN weights; isolated packed-window core; host submit/wait timing',
        'strict_candidate_available': all(row['errors']['bitwise_exact'] and row['changed_errors']['bitwise_exact']
                                          and row['weight_mutation_errors']['bitwise_exact'] for row in rows),
        'statically_rejected_configurations': ([{'waves': 4, 'reason': '128 KiB LDS exceeds gfx1201 64 KiB/workgroup'}]
                                                if args.channels == 256 else []),
        'default_promoted': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--channels', type=int, choices=(64, 128, 256), required=True)
    parser.add_argument('--batch', type=int, choices=(1, 753, 768), default=1)
    parser.add_argument('--iterations', type=int, choices=(1, 12), default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    arguments = parser.parse_args()
    if arguments.child:
        def record_phase(name):
            import json
            with (arguments.output / 'phases.jsonl').open('a') as stream:
                stream.write(json.dumps({'phase': name}) + '\n')
        atexit.register(record_phase, 'python_atexit')
        paths = list(Path(__file__).parent.glob('native_*.py')) + [Path(__file__), arguments.dll]
        sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        result = exercise(arguments, record_phase)
        record_phase('gpu_work_complete')
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        result['allocated_after_release_bytes'] = torch.cuda.memory_allocated()
        result['reserved_after_release_bytes'] = torch.cuda.memory_reserved()
        result['sources'] = sources
        result['sources_unchanged'] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in sources.items())
        result['checks_pass'] &= result['sources_unchanged'] and result['allocated_after_release_bytes'] == result['reserved_after_release_bytes'] == 0
        record_phase('resources_released')
        save(arguments.output / 'child.json', result)
    else:
        arguments.output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), '--child', '--dll', str(arguments.dll.resolve()),
                   '--channels', str(arguments.channels), '--batch', str(arguments.batch), '--iterations', str(arguments.iterations),
                   '--output', str(arguments.output.resolve())]
        raise SystemExit(0 if supervise(command, arguments.output, timeout=120) else 2)
