"""One bounded original-weight ROCm probe per process, including normal teardown.

This is a GPU capability/lifecycle test, NOT a complete model or quality gate.
The host timeout never cancels GPU work and deliberately does not kill/retry the
child. On timeout, inspect the logged PID and stop all further GPU experiments.
"""
from __future__ import annotations
import argparse
import atexit
import faulthandler
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ORIGINAL_SHA = 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5'
PROBES = ('matmul', 'split512', 'vit1024', 'downsample128', 'vit_bottleneck', 'upsample256', 'upsample32', 'decoder_pyramid', 'whole_frame128', 'whole_frame640')


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def acceptance(returncode, timed_out, child, phases):
    required = {'gpu_work_complete', 'resources_released', 'python_atexit'}
    return (not timed_out and returncode == 0 and child.get('checks_pass') is True
            and child.get('allocated_after_release_bytes') == 0
            and required <= set(phases))


def summarize_event_times(values):
    valid = bool(values) and all(math.isfinite(v) and v >= 0 for v in values)
    return {'event_timing_valid': valid, 'gpu_event_sum_ms': sum(values) if valid else None,
            'event_timing_note': 'Uncalibrated diagnostic only' if valid else
            'Invalid negative/nonfinite event duration; use host timing, do not infer GPU throughput'}


def configure_fault_logging(trace, handler=faulthandler):
    """Keep fatal-error diagnostics; no concurrent periodic stack walking.

    The whole_frame640 twelve-run crashed at the 30-second traceback timer in
    python312!PyCode_Addr2Line. This is a monitoring mitigation, NOT proof that
    its underlying cause is fixed. Preserve the failed run and the GPU hold.
    Host timeout/phase/PID evidence remain enabled; no automatic kill or retry.
    """
    handler.enable(file=trace)


def source_fingerprints():
    """Bind a new probe to the actual monitor and model source, not old reports."""
    names=('validate_rocm_lifecycle.py','validate_rocm_whole_frame.py',
           'native_whole_frame.py','native_preblock.py','native_split_image512.py',
           'native_multiscale.py','native_transition_projections.py',
           'native_vit1024.py','native_decoder_pyramid.py','native_head_torch.py',
           'native_execution_policy.py','native_capture_fixture.py')
    return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest().upper()
            for name in names}


def host_timeout_for(probe, iterations):
    """Reviewed host budget only; does not change GPU event/TDR deadlines."""
    if iterations not in (1,12,100):
        raise ValueError('unreviewed iteration count')
    if iterations==100 and probe not in ('whole_frame128','whole_frame640'):
        raise ValueError('100 iterations only reviewed for bounded single-color whole-frame probes')
    return 360 if iterations==100 else 90


def supervise(command, output, timeout=90):
    """No automatic termination, no retries, parent does not import torch."""
    start = time.monotonic()
    with (output / 'stdout.log').open('x') as stdout, (output / 'stderr.log').open('x') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   env=dict(os.environ, PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1'))
        save(output / 'process.json', {'pid': process.pid, 'command': command,
                                     'timeout_seconds': timeout, 'automatic_kill': False})
        try:
            code = process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    child_path = output / 'child.json'
    child = json.loads(child_path.read_text()) if child_path.exists() else {}
    phase_path = output / 'phases.jsonl'
    phases = []
    if phase_path.exists():
        for line in phase_path.read_text().splitlines():
            try:
                phases.append(json.loads(line)['phase'])
            except (ValueError, KeyError):
                pass  # A timed-out writer may leave an incomplete final line.
    ok = acceptance(code, timed_out, child, phases)
    report = {'schema': 1, 'status': 'BOUNDED_ROCM_LIFECYCLE_PASS' if ok else 'STOP_GPU_REQUIRES_REVIEW',
              'supervision_timeout_seconds':timeout,
              'pass': ok, 'child_pid': process.pid, 'returncode': code,
              'normal_exit': not timed_out and code == 0, 'host_timeout': timed_out,
              'child_left_running_on_timeout': timed_out, 'automatic_retry': False,
              'wall_seconds_including_startup_and_teardown': time.monotonic()-start,
              'phases': phases, 'probe': child, 'native_graph_complete': False,
              'image_quality_verified': False, 'historical_hang_root_cause_resolved': False}
    save(output / 'manifest.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'pass', 'normal_exit', 'host_timeout',
                                                  'wall_seconds_including_startup_and_teardown')}, indent=2), flush=True)
    return ok


def original_records(block, sizes):
    root = Path('local_models/decoded_310_8')
    arena = (root / 'model_arena.raw').read_bytes()
    if hashlib.sha256(arena).hexdigest().upper() != ORIGINAL_SHA:
        raise ValueError('original model hash changed')
    table = {t['name']: t for t in json.loads((root / 'manifest.json').read_bytes())['tensors']}
    records = {}
    for layer, (name, size) in enumerate(sizes.items()):
        spec = table[f'block{block}.layer{layer}.layer']
        if spec['data_bytes'] != size:
            raise ValueError('original named record size differs')
        records[name] = arena[spec['arena_offset']:spec['arena_offset'] + size]
    return records


def exercise(probe, phase, iterations=1, synchronization='block', execution='eager', artifact_output=None):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(1201)
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm required; CPU/CUDA fallback forbidden')
    properties = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in properties.gcnArchName:
        raise RuntimeError('expected gfx1201')
    phase('gpu_initialized')
    torch.cuda.reset_peak_memory_stats()

    def wait():
        event = torch.cuda.Event()
        event.record()
        deadline = time.monotonic()+10
        while not event.query():
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU event timeout; stop, no retry or implied cancellation')
            time.sleep(.005)

    if probe == 'vit_bottleneck':
        from validate_rocm_vit_chain import exercise_chain
        return exercise_chain(phase, wait, properties, iterations=iterations, synchronization=synchronization, execution=execution)

    if probe == 'decoder_pyramid':
        from validate_rocm_decoder_pyramid import exercise_decoder
        return exercise_decoder(phase,wait,properties)

    if probe in ('whole_frame128','whole_frame640'):
        from validate_rocm_whole_frame import exercise_whole_frame
        return exercise_whole_frame(phase,wait,properties,probe,iterations,artifact_output)

    if probe == 'matmul':
        model = torch.nn.Linear(32, 64, bias=False).half()
        x = (torch.randn(1, 32)*.01).half()
        selected_weight = model.weight
        forward = lambda module, value: module(value)
        hashes = {}
    elif probe == 'split512':
        from native_split_swin512 import RecoveredSplitSwin512, RECORD_SIZES
        records = original_records(40, RECORD_SIZES)
        model = RecoveredSplitSwin512(**records)
        x = (torch.randn(1, 64, 512)*.01).half()
        selected_weight = model.attention_projection.weight
        forward = lambda module, value: module(value, value)
        hashes = model.record_sha256
    elif probe == 'vit1024':
        from native_vit1024 import RecoveredVit1024, RECORD_SIZES
        records = original_records(31, RECORD_SIZES)
        model = RecoveredVit1024(**records)
        x = (torch.randn(1, 17, 1024)*.01).half()
        selected_weight = model.project
        forward = lambda module, value: module(value)
        hashes = model.record_sha256
    elif probe == 'downsample128':
        from native_multiscale import RecoveredDownsampleSwin
        records = original_records(14, {'boundary': 229936})
        model = RecoveredDownsampleSwin(records['boundary'], record_kind='swin128')
        x = (torch.randn(8*8*128)*.01).half()
        selected_weight = model.pool_project
        def forward(module, value):
            outputs = module(value, 8, 8, 0, 0, window_batch=1)
            return torch.cat([outputs['skip'], outputs['downsampled'], outputs['captured_outview']])
        hashes = {key: hashlib.sha256(raw).hexdigest().upper() for key, raw in records.items()}
    elif probe in ('upsample256','upsample32'):
        from native_upsample_swin import RecoveredUpsampleSwin, RECORD_SIZES
        c = 256 if probe=='upsample256' else 32
        records = original_records(48 if c==256 else 66, {'upsample': RECORD_SIZES[f'swin{c}']})
        model = RecoveredUpsampleSwin(records['upsample'],record_kind=f'swin{c}')
        x = (torch.randn(96*c)*.01).half()
        selected_weight = model.project
        forward = lambda module,value: module(value[:32*c],value[32*c:],8,8,0,0,window_batch=1)
        hashes = {'upsample':model.original_record_sha256}
    else:
        raise ValueError('unsupported probe')

    # CPU is a separate diagnostic baseline only. The measured forward and
    # backward below execute with every parameter, buffer and input on the GPU.
    model.requires_grad_(False)
    with torch.no_grad():
        reference = forward(model, x)
    phase('cpu_reference_complete')
    weight_name = next(name for name, value in model.named_parameters() if value is selected_weight)
    model.cuda()
    selected_weight = dict(model.named_parameters())[weight_name]
    selected_weight.requires_grad_(True)
    dx = x.cuda().requires_grad_()
    if any(t.device.type != 'cuda' for t in [dx, *model.parameters(), *model.buffers()]):
        raise RuntimeError('host neural tensor in GPU execution')
    wait()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    phase('gpu_forward_submitting')
    host_start = time.monotonic()
    begin.record()
    actual = forward(model, dx)
    end.record()
    wait()
    forward_ms = begin.elapsed_time(end)
    forward_host_ms = (time.monotonic()-host_start)*1000
    phase('gpu_forward_complete')
    begin.record()
    actual.float().square().mean().backward()
    end.record()
    wait()
    backward_ms = begin.elapsed_time(end)
    phase('gpu_backward_complete')
    got = actual.detach().cpu().float()
    grads = [dx.grad.detach().cpu().float(), selected_weight.grad.detach().cpu().float()]
    finite = bool(torch.isfinite(got).all()) and all(bool(torch.isfinite(g).all()) for g in grads)
    nonzero = all(bool(torch.count_nonzero(g)) for g in grads)
    delta = got-reference.float()
    nrmse = float(delta.square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-8))
    phase('gpu_work_complete')
    return {**summarize_event_times([forward_ms, backward_ms]),
            'name': probe, 'backend': 'pytorch_rocm', 'torch': torch.__version__,
            'hip': torch.version.hip, 'device': properties.name, 'architecture': properties.gcnArchName,
            'input_shape': list(x.shape), 'input_source': 'DETERMINISTIC_SYNTHETIC_CAPABILITY_PROBE',
            'original_model_sha256': ORIGINAL_SHA if hashes else None, 'record_hashes': hashes,
            'cpu_reference_nrmse_diagnostic': nrmse, 'max_absolute_error': float(delta.abs().max()),
            'finite': finite, 'input_and_selected_weight_gradients_nonzero': nonzero,
            'checks_pass': finite and nonzero,
            'forward_ms_first_call': forward_ms, 'backward_ms_first_call': backward_ms,
            'forward_host_ms_including_submission_and_wait': forward_host_ms,
            'timing_scope': 'diagnostic first call, not optimized network latency or game FPS',
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'cpu_neural_fallback': False, 'training_started': False,
            'full_parameter_backward_tested': False, 'rtx_quality_comparison': False}


def child(probe, output, precision='recovered_k32', iterations=1, synchronization='block', execution='eager'):
    initial_source_sha256=source_fingerprints()
    save(output/'implementation_sources_at_start.json',initial_source_sha256)
    # Keep this file open through interpreter shutdown for a teardown stack.
    trace = (output / 'traceback.log').open('x')
    configure_fault_logging(trace)
    def phase(name):
        record = {'phase': name, 'pid': os.getpid(), 'monotonic': time.monotonic()}
        with (output / 'phases.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    atexit.register(phase, 'python_atexit')
    phase('child_enter')
    from native_execution_policy import execution_policy, profile_description
    with execution_policy(precision):
        result = exercise(probe, phase, iterations, synchronization, execution, output)
        result['arithmetic_profile'] = profile_description()
    result['monitoring']={'fatal_trace_enabled':True,'periodic_traceback_timer_enabled':False,
                          'automatic_retry':False,'automatic_kill':False}
    result['implementation_source_sha256']=initial_source_sha256
    final_source_sha256=source_fingerprints()
    result['source_files_changed_during_probe']=[name for name,sha in initial_source_sha256.items()
                                               if final_source_sha256[name]!=sha]
    phase('release_begin')
    gc.collect()
    import torch
    # exercise() has returned no tensor references. No reset/TDR manipulation,
    # forced os._exit, or skipped HIP destructors to fake a successful exit.
    torch.cuda.empty_cache()
    result['allocated_after_model_release_bytes'] = torch.cuda.memory_allocated()
    result['blas_workspace_clear_available'] = hasattr(torch._C, '_cuda_clearCublasWorkspaces')
    # This version-specific PyTorch diagnostic API also backs the HIP build.
    # Retained BLAS workspaces are not model tensors. Clear only after all work
    # finished, and preserve both measurements instead of relaxing the zero gate.
    if result['blas_workspace_clear_available']:
        phase('blas_workspace_clear_begin')
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
    result['allocated_after_release_bytes'] = torch.cuda.memory_allocated()
    result['reserved_after_release_bytes'] = torch.cuda.memory_reserved()
    phase('resources_released')
    save(output / 'child.json', result)
    phase('child_return')
    # Retain trace handle (including for post-main destructor diagnostics).
    globals()['_teardown_trace_file'] = trace
    return result['checks_pass']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', choices=PROBES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--precision', choices=('recovered_k32', 'native_fp16'), default='recovered_k32')
    parser.add_argument('--iterations', type=int, choices=(1, 12, 100), default=1)
    parser.add_argument('--synchronization', choices=('block', 'frame'), default='block')
    parser.add_argument('--execution', choices=('eager', 'graph'), default='eager')
    args = parser.parse_args()
    host_timeout=host_timeout_for(args.probe,args.iterations)
    whole_frame = args.probe in ('whole_frame128','whole_frame640')
    if whole_frame and (args.synchronization != 'block' or args.execution != 'eager'):
        raise ValueError('whole-frame reviewed probe only supports staged eager execution')
    if (args.iterations != 1 or args.synchronization != 'block' or args.execution != 'eager') and args.probe != 'vit_bottleneck' and not whole_frame:
        raise ValueError('repeated probe only supports the bounded bottleneck')
    if args.execution == 'graph' and args.synchronization != 'frame':
        raise ValueError('graph replay cannot use host waits between captured blocks')
    if args.child:
        return child(args.probe, args.output, args.precision, args.iterations, args.synchronization, args.execution)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--child',
               '--probe', args.probe, '--output', str(args.output.resolve()),
               '--precision', args.precision, '--iterations', str(args.iterations),
               '--synchronization', args.synchronization, '--execution', args.execution]
    return supervise(command, args.output, timeout=host_timeout)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 2)
