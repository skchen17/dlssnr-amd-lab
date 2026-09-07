"""One-submit 1080p reset/single-color native record0..70 NRPlan gate.

This is a bounded functional bring-up.  It does not claim the original
temporal input contract, reference parity, game readiness, or performance
acceptance.
"""
from __future__ import annotations

import argparse
import atexit
import ctypes as ct
import gc
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

from gpu_safety import require_gpu_tests_enabled
from native_cpp_nr_plan import (
    ApproxBottleneck, ApproxHead, ApproxPre, ApproxScaleTransition,
    ApproxSplit512Block, ApproxStageBlock, ApproxVitBlock, ArenaRegion,
    BindingsV3, Desc, GraphStats, ModelPackageStats, PerformanceStats,
    ResourceStats, ShapeDescV3, approximate_bottleneck_descriptor,
    approximate_edge_descriptors, approximate_scale_transition_descriptor_array,
    approximate_split512_descriptor_array, approximate_stage_descriptor_array,
    approximate_vit_descriptor_array, validate_v3_layouts,
)
from validate_native_stage_nrplan import bind, call, sha256
from validate_rocm_lifecycle import save, supervise


def descriptors(args):
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    pre, head = approximate_edge_descriptors(load(args.edge_topology))
    return (approximate_stage_descriptor_array(load(args.stage_topology)),
            approximate_split512_descriptor_array(load(args.split_topology)),
            approximate_vit_descriptor_array(load(args.vit_topology)),
            approximate_bottleneck_descriptor(load(args.bottleneck_topology)),
            approximate_scale_transition_descriptor_array(load(args.transition_topology)),
            pre, head)


def bind_extended(library):
    bind(library)
    signatures = (
        ('nrPlanConfigureApproxSplit512Blocks',
         [ct.c_void_p, ct.POINTER(ApproxSplit512Block), ct.c_uint32]),
        ('nrPlanConfigureApproxVitBlocks',
         [ct.c_void_p, ct.POINTER(ApproxVitBlock), ct.c_uint32]),
        ('nrPlanConfigureApproxBottleneck',
         [ct.c_void_p, ct.POINTER(ApproxBottleneck)]),
        ('nrPlanConfigureApproxScaleTransitions',
         [ct.c_void_p, ct.POINTER(ApproxScaleTransition), ct.c_uint32]),
        ('nrPlanConfigureApproxPre', [ct.c_void_p, ct.POINTER(ApproxPre)]),
        ('nrPlanConfigureApproxHead', [ct.c_void_p, ct.POINTER(ApproxHead)]),
    )
    for name, types in signatures:
        function = getattr(library, name)
        function.argtypes = types
        function.restype = ct.c_int
    library.nrPlanSetApproxEdgeCompactQkv.argtypes = [ct.c_void_p, ct.c_uint8]
    library.nrPlanSetApproxEdgeCompactQkv.restype = ct.c_int
    library.nrPlanSetApproxStandardCompactQkv.argtypes = [ct.c_void_p, ct.c_uint8]
    library.nrPlanSetApproxStandardCompactQkv.restype = ct.c_int
    library.nrPlanSetApproxFullFusedQkv.argtypes = [ct.c_void_p, ct.c_uint8]
    library.nrPlanSetApproxFullFusedQkv.restype = ct.c_int
    library.nrPlanSetApproxFusedQkvConsumesFp16.argtypes = [ct.c_void_p, ct.c_uint8]
    library.nrPlanSetApproxFusedQkvConsumesFp16.restype = ct.c_int
    library.nrPlanSetApproxElideRedundantPostPublish.argtypes = [ct.c_void_p, ct.c_uint8]
    library.nrPlanSetApproxElideRedundantPostPublish.restype = ct.c_int


def exercise(args, values):
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm GPU required')
    props = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:
        raise RuntimeError('gfx1201 required')
    # The plan owns roughly 2 GiB at this gate. Keep unrelated allocator growth
    # bounded while leaving sufficient room for the plan and two RGBA16F frames.
    torch.cuda.set_per_process_memory_fraction(3_500_000_000 / props.total_memory)
    torch.manual_seed(17071)
    source = torch.rand((1080, 1920, 4), device='cuda', dtype=torch.float16)
    output = torch.full_like(source, float('nan'))
    stage, split, vit, bottleneck, transitions, pre, head = values
    arena = json.loads(args.arena.read_text(encoding='utf-8'))
    regions = [ArenaRegion(row['offset'], row['bytes'], row['alignment'], 0,
                           row['name'].encode()) for row in arena['regions']]
    region_array = (ArenaRegion * len(regions))(*regions)
    library = ct.CDLL(str(args.dll.resolve()))
    bind_extended(library)
    library.nrPlanDebugSetFixedHostBoundaries.argtypes = [ct.c_void_p, ct.c_uint8]
    library.nrPlanDebugSetFixedHostBoundaries.restype = ct.c_int
    handle = ct.c_void_p()
    try:
        call(library.nrPlanCreate(ct.byref(Desc(arena['workspace_bytes'], 0, 1920, 1080)),
                                  ct.byref(handle)), 'create')
        call(library.nrPlanSetPrecisionProfile(handle, 1), 'profile')
        call(library.nrPlanSetExecutionMode(handle, 1), 'fixed sequence')
        call(library.nrPlanDebugSetFixedHostBoundaries(handle, 0),
             'disable host boundaries')
        if args.compact_edge_qkv:
            call(library.nrPlanSetApproxEdgeCompactQkv(handle, 1),
                 'compact edge QKV publisher')
        if args.compact_standard_qkv:
            call(library.nrPlanSetApproxStandardCompactQkv(handle, 1),
                 'compact standard QKV publisher')
        if args.full_fused_qkv:
            call(library.nrPlanSetApproxFullFusedQkv(handle, 1),
                 'full fused QKV')
        if args.fused_qkv_fp16_input:
            call(library.nrPlanSetApproxFusedQkvConsumesFp16(handle, 1),
                 'fused QKV FP16 input')
        if args.elide_redundant_post_publish:
            call(library.nrPlanSetApproxElideRedundantPostPublish(handle, 1),
                 'elide redundant post publish')
        package = ModelPackageStats()
        call(library.nrPlanLoadModelPackage(handle, str(args.package.resolve()).encode(),
                                            ct.byref(package)), 'package')
        call(library.nrPlanConfigureArena(handle, region_array, len(regions)), 'arena')
        shape = ShapeDescV3(ct.sizeof(ShapeDescV3), 1920, 1080, 1920, 1080,
                            0, 0, 1920, 1080, 0, 1, 2, 3)
        call(library.nrPlanPrepareShape(handle, ct.byref(shape)), 'shape')
        call(library.nrPlanConfigureApproxStageBlocks(handle, stage, len(stage), 0),
             'stage')
        call(library.nrPlanConfigureApproxSplit512Blocks(handle, split, len(split)),
             'split')
        call(library.nrPlanConfigureApproxVitBlocks(handle, vit, len(vit)), 'vit')
        call(library.nrPlanConfigureApproxBottleneck(handle, ct.byref(bottleneck)),
             'bottleneck')
        call(library.nrPlanConfigureApproxScaleTransitions(handle, transitions,
                                                            len(transitions)),
             'transitions')
        call(library.nrPlanConfigureApproxPre(handle, ct.byref(pre)), 'pre')
        call(library.nrPlanConfigureApproxHead(handle, ct.byref(head)), 'head')
        call(library.nrPlanFinalize(handle), 'finalize')
        graph = GraphStats()
        call(library.nrPlanGetGraphStats(handle, ct.byref(graph)), 'graph stats')
        binding = BindingsV3(
            ct.sizeof(BindingsV3), 3, source.data_ptr(), output.data_ptr(),
            0, 0, 0, 0, 0, 0, 1920, 1080, 1920, 1080,
            0, 0, 1920, 1080, 0., 0., 1., 1., 1., 1., 0, 1, 1, 0)
        # Input initialization is on PyTorch's stream; make it ready before the
        # diagnostic submission on the plan's separate stream.
        torch.cuda.synchronize()
        sample_times = []
        sample_hashes = []
        if args.operation_timing:
            library.nrPlanDebugWriteOperationTimings.argtypes = [ct.c_void_p, ct.c_char_p]
            library.nrPlanDebugWriteOperationTimings.restype = ct.c_int
        for iteration in range(args.warmup + args.samples):
            call(library.nrPlanSubmitV3(handle, ct.byref(binding), None, None), 'submit')
            torch.cuda.synchronize()
            measured = PerformanceStats()
            call(library.nrPlanGetPerformanceStats(handle, ct.byref(measured)), 'sample perf')
            if iteration >= args.warmup:
                index = iteration - args.warmup
                sample_times.append(measured.last_gpu_ms)
                sample_hashes.append(hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest().upper())
                if args.operation_timing:
                    call(library.nrPlanDebugWriteOperationTimings(handle,
                         str((args.output / f'operations_{index:02d}.json').resolve()).encode()),
                         'operation timings')
        # The native plan owns a non-default stream and PyTorch does not know
        # that the external output pointer was written there. This diagnostic
        # device wait is outside the measured graph span and is never a hot-path
        # implementation requirement.
        torch.cuda.synchronize()
        perf = PerformanceStats()
        call(library.nrPlanGetPerformanceStats(handle, ct.byref(perf)), 'perf')
        resource = ResourceStats()
        call(library.nrPlanGetResourceStats(handle, ct.byref(resource)), 'resources')
        finite = bool(torch.isfinite(output).all().item())
        nonzero = int(torch.count_nonzero(output).item())
        raw = output.cpu().numpy().tobytes()
        output_hash = hashlib.sha256(raw).hexdigest().upper()
        expected_hash = '2A756063FB97D2F88BE384A09129F085055EE08C1E9917B1BF8005270DDB56E3'
        expected_kernels = (510 if args.full_fused_qkv and
                            (args.fused_qkv_fp16_input or
                             args.elide_redundant_post_publish) else
                            554 if args.full_fused_qkv else
                            738 - (4 if args.compact_edge_qkv else 0) -
                            (88 if args.compact_standard_qkv else 0))
        checks = (finite and nonzero > 0 and output_hash == expected_hash and
                  all(value == expected_hash for value in sample_hashes) and
                  graph.kernel_nodes == expected_kernels and graph.memcpy_nodes == 5 and
                  bool(resource.owns_graph_source) and
                  not bool(resource.graph_references_external_allocations) and
                  resource.native_stage_block_count == 71 and
                  bool(resource.complete_native_topology) and
                  not bool(resource.temporal_contract_verified) and
                  not bool(resource.deployment_ready))
        return {
            'checks_pass': checks,
            'functional_native_71_record_reset_gate_pass': checks,
            'scope': f'records 0..70, 1080p SDR, reset/single-color route, {args.samples} measured frames',
            'compact_edge_qkv': bool(args.compact_edge_qkv),
            'compact_standard_qkv': bool(args.compact_standard_qkv),
            'full_fused_qkv': bool(args.full_fused_qkv),
            'fused_qkv_fp16_input': bool(args.fused_qkv_fp16_input),
            'elide_redundant_post_publish': bool(args.elide_redundant_post_publish),
            'explicit_non_claims': ['original temporal contract', 'reference parity',
                                    'game readiness', 'performance acceptance'],
            'output_finite': finite,
            'output_nonzero_values': nonzero,
            'output_sha256': output_hash,
            'expected_output_sha256': expected_hash,
            'hash_matches_baseline': output_hash == expected_hash,
            'diagnostic_event_span_ms': perf.last_gpu_ms,
            'event_samples_ms': sample_times,
            'event_median_ms': statistics.median(sample_times),
            'sample_hashes': sample_hashes,
            'warmup_frames': args.warmup,
            'measured_frames': args.samples,
            'operation_timing_instrumented': args.operation_timing,
            'graph': {'total_nodes': graph.total_nodes,
                      'kernel_nodes': graph.kernel_nodes,
                      'expected_kernel_nodes': expected_kernels,
                      'memcpy_nodes': graph.memcpy_nodes},
            'resource': {
                'workspace_bytes': resource.workspace_bytes,
                'weight_bytes': resource.weight_bytes,
                'native_stage_block_count': resource.native_stage_block_count,
                'complete_native_topology': bool(resource.complete_native_topology),
                'owns_graph_source': bool(resource.owns_graph_source),
                'references_external_allocations': bool(
                    resource.graph_references_external_allocations),
                'temporal_contract_verified': bool(resource.temporal_contract_verified),
                'deployment_ready': bool(resource.deployment_ready),
            },
            'device': props.name,
            'architecture': props.gcnArchName,
            'gpu_work_cancelled': False,
            'automatic_retry': False,
        }
    finally:
        if handle.value:
            call(library.nrPlanDestroy(handle), 'destroy')


def child(args):
    require_gpu_tests_enabled('71-record reset/single-color NRPlan functional gate')
    values = descriptors(args)

    def phase(name):
        with (args.output / 'phases.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase': name, 'pid': os.getpid()}) + '\n')

    atexit.register(phase, 'python_atexit')
    phase('child_enter')
    import torch
    try:
        result = exercise(args, values)
        phase('gpu_work_complete')
    except BaseException as error:
        result = {'checks_pass': False, 'error_type': type(error).__name__,
                  'error': str(error), 'gpu_work_cancelled': False,
                  'automatic_retry': False}
        phase('gpu_work_failed')
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    result['allocated_after_release_bytes'] = torch.cuda.memory_allocated()
    result['reserved_after_release_bytes'] = torch.cuda.memory_reserved()
    result['checks_pass'] &= (result['allocated_after_release_bytes'] == 0 and
                              result['reserved_after_release_bytes'] == 0)
    phase('resources_released')
    save(args.output / 'child.json', result)
    return result['checks_pass']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dll', 'package', 'arena', 'stage-topology', 'split-topology',
                 'vit-topology', 'bottleneck-topology', 'transition-topology',
                 'edge-topology', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--compact-edge-qkv', action='store_true')
    parser.add_argument('--compact-standard-qkv', action='store_true')
    parser.add_argument('--full-fused-qkv', action='store_true')
    parser.add_argument('--fused-qkv-fp16-input', action='store_true')
    parser.add_argument('--elide-redundant-post-publish', action='store_true')
    parser.add_argument('--operation-timing', action='store_true')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.samples <= 12 or not 0 <= args.warmup <= 5:
        parser.error('bounded measurement requires 1..12 samples and 0..5 warmup')
    if args.operation_timing:
        os.environ['NRPLAN_OPERATION_TIMING'] = '1'
        os.environ['NRPLAN_OPERATION_DOT'] = str((args.output / 'original_graph.dot').resolve())
    else:
        os.environ.pop('NRPLAN_OPERATION_TIMING', None)
    if args.fused_qkv_fp16_input and not args.full_fused_qkv:
        parser.error('--fused-qkv-fp16-input requires --full-fused-qkv')
    if args.elide_redundant_post_publish and not args.full_fused_qkv:
        parser.error('--elide-redundant-post-publish requires --full-fused-qkv')
    validate_v3_layouts()
    values = descriptors(args)
    preflight = {
        'status': 'CPU_PREFLIGHT_PASS_GPU_NOT_EXECUTED',
        'dll_sha256': sha256(args.dll),
        'standard_blocks': len(values[0]),
        'split_blocks': len(values[1]),
        'vit_blocks': len(values[2]),
        'transitions': len(values[4]),
        'records': [0, 70],
        'temporal_contract_verified': False,
    }
    if not args.execute:
        save(args.output, preflight)
        return True
    require_gpu_tests_enabled('71-record reset/single-color NRPlan functional gate')
    if args.child:
        return child(args)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--execute']
    for name in ('dll', 'package', 'arena', 'stage_topology', 'split_topology',
                 'vit_topology', 'bottleneck_topology', 'transition_topology',
                 'edge_topology', 'output'):
        command += ['--' + name.replace('_', '-'), str(getattr(args, name).resolve())]
    if args.compact_edge_qkv:
        command.append('--compact-edge-qkv')
    if args.compact_standard_qkv:
        command.append('--compact-standard-qkv')
    if args.full_fused_qkv:
        command.append('--full-fused-qkv')
    if args.fused_qkv_fp16_input:
        command.append('--fused-qkv-fp16-input')
    if args.elide_redundant_post_publish:
        command.append('--elide-redundant-post-publish')
    if args.operation_timing:
        command.append('--operation-timing')
    command += ['--samples', str(args.samples), '--warmup', str(args.warmup)]
    return supervise(command, args.output, timeout=240)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 2)
