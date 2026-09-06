"""Supervised C512 local grouped-MLP WMMA candidate gate."""
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
    return {'max_abs': difference.abs().max().item(), 'rmse': difference.square().mean().sqrt().item(),
            'different_fraction': unequal.double().mean().item(),
            'first_different_flat': int(positions[0]) if len(positions) else None,
            'bitwise_exact': not bool(unequal.any())}


def exercise(args, phase):
    import torch
    from native_execution_policy import execution_policy
    from native_matrix_fusion import matrix_fusion
    from native_model_package import load_package
    from native_split_swin512 import SplitFFWD512, chunked_linear, cubic_silu
    from native_swin_torch import quantize_e4

    torch.set_num_threads(2)
    torch.manual_seed(1712)
    if not torch.version.hip or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('gfx1201 ROCm device required')
    torch.cuda.set_per_process_memory_fraction(2_500_000_000 / torch.cuda.get_device_properties(0).total_memory)
    records, _, _ = load_package('local_models/native_single_color_v1',
        'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    model = SplitFFWD512(records[(24, 0)]).eval().cuda().requires_grad_(False)
    projected = (torch.randn(args.batch, 64, 512, device='cuda', dtype=torch.float16) * .1).contiguous()
    original = projected.clone()
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

    def reference(value):
        grouped = value.reshape(-1, 64, 8, 64).transpose(1, 2)
        hidden = chunked_linear(grouped[..., model.perm64], model.expand)
        hidden = cubic_silu(hidden)
        contracted = chunked_linear(hidden[..., model.perm256], model.contract)
        return quantize_e4(contracted.transpose(1, 2).reshape(-1, 64, 512))

    rows = []
    with torch.no_grad(), torch.cuda.stream(stream), execution_policy('native_fp16'):
        expected = reference(projected)
        wait()
        for profile in ('wmma_fp16', 'wmma_fp8'):
            for waves in (1, 2, 4):
                with matrix_fusion(args.dll, profile, modules=('c512_ffn',), waves=waves) as operator:
                    output = operator.c512_group_ffn(model, projected)
                    wait()
                    base = metrics(output, expected)
                    fixed = output.clone()
                    output = operator.c512_group_ffn(model, projected)
                    wait()
                    repeat = metrics(output, fixed)
                    if not repeat['bitwise_exact'] or not bool(torch.isfinite(output).all()):
                        raise RuntimeError(f'C512 candidate unstable profile={profile} waves={waves}')
                    timings = {}
                    for mode in ('reference', 'candidate'):
                        wait()
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            output = reference(projected) if mode == 'reference' else operator.c512_group_ffn(model, projected)
                        wait()
                        timings[mode] = (time.perf_counter() - start) * 1000 / args.iterations
                    projected.add_(.03125)
                    output = operator.c512_group_ffn(model, projected)
                    wait()
                    changed = metrics(output, reference(projected))
                    projected.copy_(original)
                    old_expand, old_contract = model.expand.clone(), model.contract.clone()
                    model.expand.add_(.001)
                    model.contract.sub_(.001)
                    output = operator.c512_group_ffn(model, projected)
                    wait()
                    mutation = metrics(output, reference(projected))
                    model.expand.copy_(old_expand)
                    model.contract.copy_(old_contract)
                    wait()
                    rows.append({'profile': profile, 'waves': waves, 'errors': base, 'changed_errors': changed,
                                 'weight_mutation_errors': mutation, 'repeat_errors': repeat,
                                 'host_ms_per_call': timings, 'candidate_kernel_nodes': 1,
                                 'reference_logical_gemms': 2})
                    phase(f'{profile}_{waves}_measured')
                    del fixed, old_expand, old_contract
    del model, projected, original, expected, output, stream
    return {'checks_pass': True, 'batch': args.batch, 'iterations': args.iterations, 'rows': rows,
            'strict_candidate_available': all(row['errors']['bitwise_exact'] and row['changed_errors']['bitwise_exact']
                                              and row['weight_mutation_errors']['bitwise_exact'] for row in rows),
            'scope': 'C512 grouped 64->256->64 MLP only; surrounding 512x512 projections and attention excluded',
            'default_promoted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 144), default=1)
    parser.add_argument('--iterations', type=int, choices=(1, 12), default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    arguments = parser.parse_args()
    if arguments.child:
        def phase(name):
            import json
            with (arguments.output / 'phases.jsonl').open('a') as stream:
                stream.write(json.dumps({'phase': name}) + '\n')
        atexit.register(phase, 'python_atexit')
        paths = list(Path(__file__).parent.glob('native_*.py')) + [Path(__file__), arguments.dll]
        sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        result = exercise(arguments, phase)
        phase('gpu_work_complete')
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
        phase('resources_released')
        save(arguments.output / 'child.json', result)
    else:
        arguments.output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), '--child', '--dll', str(arguments.dll.resolve()),
                   '--batch', str(arguments.batch), '--iterations', str(arguments.iterations), '--output', str(arguments.output.resolve())]
        raise SystemExit(0 if supervise(command, arguments.output, timeout=120) else 2)
