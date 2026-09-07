"""Supervised two-block resident-pool chain gate for the native NRPlan."""
from __future__ import annotations

import argparse
import atexit
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import sys

from gpu_safety import require_gpu_tests_enabled
from native_cpp_nr_plan import (ApproxStageBlock, ArenaRegion, BindingsV3, Desc,
                                GraphStats, ModelPackageStats, ResourceStats,
                                ShapeDescV3, approximate_stage_descriptor_array)
from validate_native_stage_nrplan import bind, call, metric, sha256
from validate_rocm_lifecycle import save, supervise


def select(args):
    topology = json.loads(args.topology.read_text(encoding='utf-8'))
    descriptors = approximate_stage_descriptor_array(topology)
    wanted = [args.first_record, args.second_record]
    selected = []
    for number in wanted:
        matches = [(row, desc) for row, desc in zip(topology['blocks'], descriptors)
                   if row['record_number'] == number]
        if len(matches) != 1:
            raise ValueError(f'record {number} is not one unique standard stage block')
        selected.append(matches[0][0])
    if selected[0]['channels'] != selected[1]['channels'] or \
            selected[0]['channels'] not in (64, 128, 256):
        raise ValueError('two-block gate requires one C64/C128/C256 family')
    if selected[1]['record_number'] <= selected[0]['record_number']:
        raise ValueError('records must be in execution order')
    return selected


def descriptor(row, width=8, height=8):
    ox, oy = row['origin_x'], row['origin_y']
    windows = ((width - ox + 7) // 8) * ((height - oy + 7) // 8)
    fields = [name for name, _ in ApproxStageBlock._fields_
              if name not in ('struct_size', 'channels', 'windows', 'record_number',
                              'feature_width', 'feature_height', 'origin_x', 'origin_y')]
    return ApproxStageBlock(ct.sizeof(ApproxStageBlock), row['channels'], windows,
        row['record_number'], width, height, ox, oy, *(row[name] for name in fields))


def exercise(args, rows):
    import torch
    from native_model_package import load_package
    from native_packed_swin import RecoveredPackedSwin, gather_packed, scatter_packed
    from native_swin_torch import encode_e4, quantize_e4

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm GPU required; no fallback')
    props = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:
        raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(1_500_000_000 / props.total_memory)
    records, _, _ = load_package('local_models/native_single_color_v1',
        'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    channels = rows[0]['channels']
    torch.manual_seed(2401 + rows[0]['record_number'])
    logical = quantize_e4((torch.randn(8 * 8 * channels, device='cuda',
                                       dtype=torch.float16) * .1)).contiguous()
    initial = encode_e4(logical).contiguous()
    with torch.no_grad():
        for row in rows:
            model = RecoveredPackedSwin(records[(row['record_number'], 0)],
                                        record_kind=f'swin{channels}').eval().cuda()
            windows, mapping = gather_packed(logical, 8, 8, channels,
                                             row['origin_x'], row['origin_y'])
            output = model.block(windows[:, model.a_index], windows[:, model.residual_index])
            logical = quantize_e4(scatter_packed(output, mapping, 8, 8)).contiguous()
            del model, windows, mapping, output
    expected = encode_e4(logical).contiguous()

    arena = json.loads(args.arena.read_text(encoding='utf-8'))
    regions = [ArenaRegion(row['offset'], row['bytes'], row['alignment'], 0,
                           row['name'].encode('utf-8')) for row in arena['regions']]
    region_array = (ArenaRegion * len(regions))(*regions)
    native_rows = [descriptor(row) for row in rows]
    native_array = (ApproxStageBlock * len(native_rows))(*native_rows)
    library = ct.CDLL(str(args.dll.resolve()))
    bind(library)
    handle = ct.c_void_p()
    dummy_input = torch.empty(4, dtype=torch.float16, device='cuda')
    dummy_output = torch.empty_like(dummy_input)
    actual = torch.empty_like(initial)
    try:
        call(library.nrPlanCreate(ct.byref(Desc(arena['workspace_bytes'], 0, 1920, 1080)),
                                  ct.byref(handle)), 'create')
        call(library.nrPlanSetPrecisionProfile(handle, 1), 'set approximate profile')
        call(library.nrPlanSetExecutionMode(handle, 1), 'set fixed sequence')
        stats = ModelPackageStats()
        call(library.nrPlanLoadModelPackage(handle, str(args.package.resolve()).encode(),
                                             ct.byref(stats)), 'load package')
        call(library.nrPlanConfigureArena(handle, region_array, len(regions)), 'configure arena')
        shape = ShapeDescV3(ct.sizeof(ShapeDescV3), 8, 8, 8, 8, 0, 0, 8, 8,
                            0, 1, 2, 3)
        call(library.nrPlanPrepareShape(handle, ct.byref(shape)), 'prepare shape')
        call(library.nrPlanConfigureApproxStageBlocks(handle, native_array,
             len(native_rows), 0), 'configure two blocks')
        call(library.nrPlanInitializeArenaFromDevice(handle, rows[0]['raw_resident_offset'],
             ct.c_void_p(initial.data_ptr()), initial.numel()), 'initialize input')
        call(library.nrPlanFinalize(handle), 'finalize')
        graph = GraphStats()
        call(library.nrPlanGetGraphStats(handle, ct.byref(graph)), 'get graph stats')
        binding = BindingsV3(ct.sizeof(BindingsV3), 3, dummy_input.data_ptr(),
            dummy_output.data_ptr(), 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8,
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0, 1, 1, 0)
        call(library.nrPlanSubmitV3(handle, ct.byref(binding), None, None), 'submit')
        entries, post, qkv, scratch_bytes = (ct.c_uint64() for _ in range(4))
        call(library.nrPlanDebugGetFixedBoundaryState(handle, ct.byref(entries),
             ct.byref(post), ct.byref(qkv), ct.byref(scratch_bytes)), 'read chain state')
        final = native_rows[-1]
        final_scalars = final.windows * 64 * channels
        call(library.nrPlanDebugCopyFixedBoundaryScratchToDevice(handle,
             final_scalars * 6, ct.c_void_p(actual.data_ptr()), actual.numel()),
             'copy final resident output')
        error = metric(actual, expected)
        resources = ResourceStats()
        call(library.nrPlanGetResourceStats(handle, ct.byref(resources)), 'get resources')
        approximate_pass = bool(error['finite'] and error['nrmse'] <= .08 and
                                error['max_absolute_error'] <= .125)
        return {
            'checks_pass': approximate_pass and entries.value == 2 and
                           post.value == 2 and qkv.value == 2,
            'approximate_chain_gate_pass': approximate_pass,
            'records': [row['record_number'] for row in rows],
            'origins': [[row['origin_x'], row['origin_y']] for row in rows],
            'error_vs_reference': error,
            'state': {'stage_entries': entries.value, 'post_boundaries': post.value,
                      'qkv_boundaries': qkv.value, 'scratch_bytes': scratch_bytes.value},
            'graph': {'kernel_nodes': graph.kernel_nodes, 'total_nodes': graph.total_nodes},
            'resource': {'workspace_bytes': resources.workspace_bytes,
                         'weight_bytes': resources.weight_bytes,
                         'deployment_ready': bool(resources.deployment_ready)},
            'device': props.name, 'architecture': props.gcnArchName,
            'gpu_work_cancelled': False, 'automatic_retry': False,
        }
    finally:
        if handle.value:
            call(library.nrPlanDestroy(handle), 'destroy')


def child(args):
    require_gpu_tests_enabled('native NRPlan two-block resident chain gate')
    rows = select(args)
    def phase(name):
        with (args.output / 'phases.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase': name, 'pid': os.getpid()}) + '\n')
    atexit.register(phase, 'python_atexit')
    phase('child_enter')
    import torch
    try:
        result = exercise(args, rows)
        phase('gpu_work_complete')
    except BaseException as error:
        result = {'checks_pass': False, 'error_type': type(error).__name__,
                  'error': str(error), 'gpu_work_cancelled': False,
                  'automatic_retry': False}
        phase('gpu_work_failed')
    gc.collect(); torch.cuda.empty_cache()
    if hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    result['allocated_after_release_bytes'] = torch.cuda.memory_allocated()
    result['reserved_after_release_bytes'] = torch.cuda.memory_reserved()
    result['checks_pass'] &= result['allocated_after_release_bytes'] == 0 and \
        result['reserved_after_release_bytes'] == 0
    phase('resources_released')
    save(args.output / 'child.json', result)
    return result['checks_pass']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--topology', type=Path, required=True)
    parser.add_argument('--first-record', type=int, required=True)
    parser.add_argument('--second-record', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    rows = select(args)
    preflight = {'status': 'CPU_PREFLIGHT_PASS_GPU_NOT_EXECUTED',
                 'dll_sha256': sha256(args.dll),
                 'records': [row['record_number'] for row in rows],
                 'channels': rows[0]['channels'],
                 'origins': [[row['origin_x'], row['origin_y']] for row in rows]}
    if not args.execute:
        save(args.output, preflight); return True
    require_gpu_tests_enabled('native NRPlan two-block resident chain gate')
    if args.child:
        return child(args)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--execute',
        '--dll', str(args.dll.resolve()), '--package', str(args.package.resolve()),
        '--arena', str(args.arena.resolve()), '--topology', str(args.topology.resolve()),
        '--first-record', str(args.first_record), '--second-record', str(args.second_record),
        '--output', str(args.output.resolve())]
    return supervise(command, args.output, timeout=120)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 2)
