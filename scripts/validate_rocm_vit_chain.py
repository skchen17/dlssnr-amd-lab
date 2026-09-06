"""Eight original ViT blocks on GPU; invoked by the lifecycle supervisor.

One original captured feature boundary enters. No teacher tensors are fed to
any internal block. CPU reference files are read only AFTER GPU calculation.
This is a bottleneck subgraph, not image-to-image DLSS quality acceptance.
"""
import json
from pathlib import Path
import struct
import statistics
import time
import torch
from native_vit1024 import RecoveredVit1024, RecoveredVitBottleneck1024, RECORD_SIZES
from native_sequence_layout import unpack_sequence, pack_sequence
from native_swin_torch import decode_e4, encode_e4
from run_swin1h_native_validation import checked, compare, digest
from validate_rocm_lifecycle import original_records, ORIGINAL_SHA, summarize_event_times
from native_execution_policy import current_profile


def exercise_chain(phase, wait, properties, *, iterations=1, synchronization='block', execution='eager'):
    if iterations not in (1, 12) or synchronization not in ('block', 'frame'):
        raise ValueError('only reviewed 1/12 iteration stages are supported')
    root = Path('results/20260831_221000_vit1d_full_graph_exact_state/cases')
    specs = {s['slot']: s for s in json.loads((root / 'manifest.json').read_bytes())['slots']}
    table = {t['name']: t for t in json.loads(Path('local_models/decoded_310_8/manifest.json').read_bytes())['tensors']}
    previous = None
    geometry = None
    models = {}
    hashes = {}
    # Prove that the capture describes one consecutive eight-block subgraph,
    # including actual stage-to-original-weight bindings, before GPU submission.
    for block in range(31, 39):
        slot = 58+(block-31)*5
        spec, last = specs[slot], specs[slot+4]
        folder = root / f'slot{slot}'
        params = checked(folder / spec['params']['path'], spec['params']['sha256'])
        height, width = struct.unpack_from('<2i', params, 64)
        if geometry is not None and geometry != (width, height):
            raise ValueError('bottleneck dimensions changed inside chain')
        geometry = width, height
        tokens = width*height
        if tokens != 96:
            raise ValueError('this bounded probe authorizes only the captured 96-token bottleneck')
        capture = checked(folder / spec['activation_arena']['path'], spec['activation_arena']['sha256'])
        offset = next(v['arena_offset'] for v in spec['activation_param_views'] if v['param_offset'] == 0)
        raw = capture[offset:offset+((tokens+31)//32*32)*1024]
        if previous is not None and raw != previous:
            raise ValueError('noncontiguous captured bottleneck boundaries')
        out = next(v for v in last['outputs'] if v['param_offset'] == 16)
        reference = checked(root / f'slot{slot+4}' / out['asset']['path'], out['asset']['sha256'])
        previous = reference
        for i in range(5):
            if specs[slot+i]['weight_view_offset'] != table[f'block{block}.layer{i}.layer']['arena_offset']:
                raise ValueError('wrong original weight binding')
        models[block] = RecoveredVit1024(**original_records(block, RECORD_SIZES))
        hashes[block] = models[block].record_sha256
        if block == 31:
            first_raw = raw
    model = RecoveredVitBottleneck1024(models).cuda()
    x = unpack_sequence(decode_e4(first_raw).cuda(), tokens, 1024)
    if any(t.device.type != 'cuda' for t in [x, *model.parameters(), *model.buffers()]):
        raise RuntimeError('CPU neural tensor in GPU chain')
    wait()
    phase('captured_boundary_and_weights_ready')
    stages, frame_times, output_hashes = [], [], []
    entry = x
    runner = None
    if execution == 'graph':
        from native_graph_replay import CapturedTensorRunner
        def captured_forward(value):
            finite_checks = []
            for module in model.blocks:
                value = module(value)
                finite_checks.append(torch.isfinite(value).all())
            return value, torch.stack(finite_checks).all()
        phase('graph_prepare_begin')
        runner = CapturedTensorRunner(captured_forward, entry, precision=current_profile(), wait=wait)
        phase('graph_prepare_complete')
    with torch.no_grad():
        for iteration in range(iterations):
            x = entry
            pending, finite_checks = [], []
            host_start = time.perf_counter()
            if runner is not None:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                x, all_finite = runner.submit(entry)
                end.record()
                finite_checks.append(all_finite)
                pending.append((begin, end, {'iteration': iteration, 'block': '31..38_graph',
                                           'input_source': 'CAPTURED_ENTRY_THEN_NATIVE_GPU_CHAIN'}))
            for block, module in ([] if runner is not None else zip(range(31, 39), model.blocks)):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                stage_start = time.perf_counter()
                begin.record()
                x = module(x)
                finite = torch.isfinite(x).all()
                finite_checks.append(finite)
                end.record()
                if synchronization == 'block':
                    wait()
                    if not bool(finite):
                        raise RuntimeError('nonfinite native chain output; stop')
                pending.append((begin, end, {'iteration': iteration, 'block': block,
                               'host_ms_submission_and_optional_wait': (time.perf_counter()-stage_start)*1000,
                               'host_waited_after_stage': synchronization == 'block',
                               'input_source': 'CAPTURED_CHAIN_ENTRY' if block == 31 else 'PREVIOUS_NATIVE_GPU_BLOCK'}))
                if iteration == 0 and synchronization == 'block':
                    phase(f'gpu_block{block}_complete')
            all_finite = torch.stack(finite_checks).all()
            wait()
            if not bool(all_finite):
                raise RuntimeError('nonfinite subgraph intermediate; stop')
            frame_times.append((time.perf_counter()-host_start)*1000)
            for begin, end, stage in pending:
                stage['gpu_event_ms'] = begin.elapsed_time(end)
                stages.append(stage)
            packed = encode_e4(pack_sequence(x))
            wait()
            actual = packed.cpu().numpy().tobytes()
            output_hashes.append(digest(actual))
            phase(f'gpu_iteration{iteration}_complete')
            if output_hashes[-1] != output_hashes[0]:
                raise RuntimeError('identical-input repeated output changed; review before further GPU work')
    phase('gpu_work_complete')
    if runner is not None:
        phase('graph_release_begin')
        runner.close(wait)
        phase('graph_release_complete')
    cpu_root = Path('results/20260905_native_vit_bottleneck_cpu_v1')
    cpu_spec = json.loads((cpu_root / 'manifest.json').read_bytes())
    cpu_case = cpu_spec['results'][0]
    if cpu_spec['original_model_sha256'] != ORIGINAL_SHA or cpu_case['input_sha256'] != digest(first_raw):
        raise ValueError('CPU control provenance differs')
    for block in cpu_spec['decoded_blocks']:
        if block['records'] != hashes[block['block']]:
            raise ValueError('CPU control original records differ')
    cpu_raw = checked(cpu_root / 'block38.e4', cpu_case['output_sha256'])
    cpu_comparison, rtx_comparison = compare(cpu_raw, actual), compare(reference, actual)
    return {**summarize_event_times([s['gpu_event_ms'] for s in stages]),
            'name': 'vit_bottleneck', 'backend': 'pytorch_rocm', 'torch': torch.__version__,
            'hip': torch.version.hip, 'device': properties.name, 'architecture': properties.gcnArchName,
            'original_model_sha256': ORIGINAL_SHA, 'record_hashes_by_block': hashes,
            'input_source': 'ONE_ORIGINAL_CAPTURED_FEATURE_BOUNDARY', 'internal_rtx_substitution': False,
            'cpu_neural_fallback': False, 'input_shape': [1, tokens, 1024],
            'input_sha256': digest(first_raw), 'output_sha256': digest(actual),
            'cpu_comparison_diagnostic': cpu_comparison, 'rtx_comparison_diagnostic': rtx_comparison,
            'checks_pass': cpu_comparison['nonfinite'] == 0 and rtx_comparison['nonfinite'] == 0,
            'stages': stages, 'iterations': iterations, 'identical_input_output_stable': len(set(output_hashes)) == 1,
            'synchronization': synchronization,
            'execution': execution,
            'output_hashes': output_hashes, 'forward_host_ms_including_submission_and_wait': frame_times[0],
            'forward_host_ms_all_iterations': frame_times,
            'warm_host_median_ms': statistics.median(frame_times[1:]) if iterations > 1 else None,
            'warm_host_max_ms': max(frame_times[1:]) if iterations > 1 else None,
            'timing_scope': f'96-token bottleneck only; {synchronization} waits; not full-frame network or FPS',
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'backward_tested': False, 'training_started': False, 'image_quality_verified': False}
