"""One-submit middle-stage block -> encoder transition -> next-stage NRPlan gate."""
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
from native_cpp_nr_plan import (ApproxScaleTransition, ApproxSplit512Block,
                                ApproxStageBlock, ArenaRegion,
                                BindingsV3, Desc, ModelPackageStats, ResourceStats,
                                ShapeDescV3, approximate_split512_descriptor_array,
                                approximate_stage_descriptor_array)
from validate_native_stage_nrplan import bind, call, metric, sha256
from validate_rocm_lifecycle import save, supervise


def stage_descriptor(row, width, height):
    ox, oy = row['origin_x'], row['origin_y']
    windows = ((width - ox + 7) // 8) * ((height - oy + 7) // 8)
    header = {'struct_size', 'channels', 'windows', 'record_number', 'feature_width',
              'feature_height', 'origin_x', 'origin_y'}
    fields = [name for name, _ in ApproxStageBlock._fields_ if name not in header]
    return ApproxStageBlock(ct.sizeof(ApproxStageBlock), row['channels'], windows,
        row['record_number'], width, height, ox, oy, *(row[name] for name in fields))


def split_descriptor(row, width, height):
    ox, oy = row['origin_x'], row['origin_y']
    windows = ((width - ox + 7) // 8) * ((height - oy + 7) // 8)
    header = {'struct_size', 'windows', 'feature_width', 'feature_height', 'origin_x',
              'origin_y', 'record_number', 'reserved'}
    fields = [name for name, _ in ApproxSplit512Block._fields_ if name not in header]
    return ApproxSplit512Block(ct.sizeof(ApproxSplit512Block), windows, width, height,
        ox, oy, row['record_number'], 0, *(row[name] for name in fields))


def select(args):
    topology = json.loads(args.topology.read_text(encoding='utf-8'))
    # Parse the complete CPU artifact first so malformed source topology cannot
    # be hidden by selecting only two rows.
    approximate_stage_descriptor_array(topology)
    anchors = {32: (4, 5), 64: (8, 9), 128: (14, 15), 256: (22, 23)}
    first_record, second_record = anchors[args.channels]
    by_record = {row['record_number']: row for row in topology['blocks']}
    first = by_record.get(first_record)
    if args.channels == 256:
        if not args.split_topology:
            raise ValueError('--split-topology is required for C256 -> C512')
        split = json.loads(args.split_topology.read_text(encoding='utf-8'))
        approximate_split512_descriptor_array(split)
        second = {row['record_number']: row for row in split['blocks']}.get(second_record)
    else:
        second = by_record.get(second_record)
    if not first or not second or first['channels'] != args.channels or \
            (args.channels != 256 and second['channels'] != args.channels * 2):
        raise ValueError('missing recovered middle encoder boundary')
    transitions = json.loads(args.transitions.read_text(encoding='utf-8'))
    matches = [row for row in transitions.get('descriptors', [])
               if row['direction'] == 1 and row['anchor_record'] == first_record]
    if len(matches) != 1:
        raise ValueError('missing unique encoder transition')
    if not args.roundtrip:
        return first, second, matches[0]
    decoder_records = {32: 66, 64: 62, 128: 56, 256: 48}
    third = by_record.get(decoder_records[args.channels])
    decoder_matches = [row for row in transitions.get('descriptors', [])
                       if row['direction'] == 2 and
                       row['anchor_record'] == decoder_records[args.channels]]
    if not third or third['channels'] != args.channels or len(decoder_matches) != 1:
        raise ValueError('missing symmetric decoder boundary')
    return first, second, matches[0], third, decoder_matches[0]


def transition_descriptor(row):
    if row['direction'] == 1:
        values = dict(row, source_width=8, source_height=8, target_width=4,
                      target_height=4)
    else:
        values = dict(row, source_width=4, source_height=4, target_width=8,
                      target_height=8)
    header = {'struct_size', 'direction', 'anchor_record', 'channels', 'source_width',
              'source_height', 'target_width', 'target_height', 'source_origin_x',
              'source_origin_y'}
    fields = [name for name, _ in ApproxScaleTransition._fields_ if name not in header]
    return ApproxScaleTransition(ct.sizeof(ApproxScaleTransition), values['direction'],
        values['anchor_record'], values['channels'], values['source_width'],
        values['source_height'], values['target_width'], values['target_height'],
        values['source_origin_x'], values['source_origin_y'],
        *(values[name] for name in fields))


def exercise(args, rows):
    import torch
    from native_model_package import load_package
    from native_multiscale import average_pool2x2, pack_image, unpack_image
    from native_packed_swin import RecoveredPackedSwin, gather_packed, scatter_packed
    from native_split_swin512 import chunked_linear
    from native_swin_torch import encode_e4, quantize_e4

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm GPU required; no fallback')
    props = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in props.gcnArchName:
        raise RuntimeError('gfx1201 required')
    torch.cuda.set_per_process_memory_fraction(1_500_000_000 / props.total_memory)
    records, _, _ = load_package('local_models/native_single_color_v1',
        'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    first, second, transition = rows[:3]
    roundtrip = len(rows) == 5
    third, decoder_transition = rows[3:] if roundtrip else (None, None)
    channels = first['channels']
    torch.manual_seed(8000 + first['record_number'])
    logical = quantize_e4((torch.randn(8 * 8 * channels, device='cuda',
                                       dtype=torch.float16) * .1)).contiguous()
    initial = encode_e4(logical).contiguous()
    with torch.no_grad():
        boundary = records[(first['record_number'], 0)]
        # Record 8 contains the standard Swin payload plus its learned
        # downsample tail, so it must be decoded by the boundary wrapper.
        from native_multiscale import RecoveredDownsampleSwin
        downsample = RecoveredDownsampleSwin(
            boundary, record_kind=f'swin{channels}').eval().cuda()
        source_model = downsample.block
        windows, mapping = gather_packed(logical, 8, 8, channels,
                                         first['origin_x'], first['origin_y'])
        source_output = source_model(windows)
        source_packed = scatter_packed(source_output, mapping, 8, 8)
        pooled = average_pool2x2(unpack_image(source_packed, 8, 8, channels))
        projected = quantize_e4(chunked_linear(
            pooled[..., downsample.permutation], downsample.pool_project))
        target_resident = pack_image(projected)
        target_channels = channels * 2
        if channels == 256:
            from native_packed_swin import logical_indices
            from native_split_swin512 import RecoveredSplitSwin512, RECORD_SIZES
            parts = {name: records[(second['record_number'], index)]
                     for index, name in enumerate(RECORD_SIZES)}
            target_model = RecoveredSplitSwin512(**parts).eval().cuda()
        else:
            target_model = RecoveredPackedSwin(records[(second['record_number'], 0)],
                record_kind=f'swin{target_channels}').eval().cuda()
        target_windows, target_mapping = gather_packed(target_resident, 4, 4,
            target_channels, second['origin_x'], second['origin_y'])
        if channels == 256:
            ai, ri = logical_indices(512); ai = ai.cuda(); ri = ri.cuda()
            target_output = target_model(target_windows[:, ai], target_windows[:, ri])
        else:
            target_output = target_model(target_windows)
        target_packed = quantize_e4(scatter_packed(
            target_output, target_mapping, 4, 4)).contiguous()
        if roundtrip:
            from native_upsample_swin import RecoveredUpsampleSwin
            upsample = RecoveredUpsampleSwin(records[(third['record_number'], 0)],
                record_kind=f'swin{channels}').eval().cuda()
            skip = quantize_e4(source_packed).contiguous()
            fused = quantize_e4(upsample.fuse_resident(
                target_packed, skip, 8, 8)).contiguous()
            final_windows, final_mapping = gather_packed(fused, 8, 8, channels,
                third['origin_x'], third['origin_y'])
            final_output = upsample.block(final_windows)
            expected = encode_e4(quantize_e4(scatter_packed(
                final_output, final_mapping, 8, 8))).contiguous()
        else:
            expected = encode_e4(target_packed).contiguous()

    arena = json.loads(args.arena.read_text(encoding='utf-8'))
    regions = [ArenaRegion(row['offset'], row['bytes'], row['alignment'], 0,
                           row['name'].encode()) for row in arena['regions']]
    region_array = (ArenaRegion * len(regions))(*regions)
    native_rows = [stage_descriptor(first, 8, 8)]
    native_split = split_descriptor(second, 4, 4) if channels == 256 else None
    if channels != 256:
        native_rows.append(stage_descriptor(second, 4, 4))
    if roundtrip:
        native_rows.append(stage_descriptor(third, 8, 8))
    native_array = (ApproxStageBlock * len(native_rows))(*native_rows)
    native_transition = transition_descriptor(transition)
    native_decoder = transition_descriptor(decoder_transition) if roundtrip else None
    library = ct.CDLL(str(args.dll.resolve()))
    bind(library)
    library.nrPlanDebugConfigureApproxEncoderTransition.argtypes = [
        ct.c_void_p, ct.POINTER(ApproxScaleTransition)]
    library.nrPlanDebugConfigureApproxEncoderTransition.restype = ct.c_int
    library.nrPlanDebugAppendApproxDecoderTransition.argtypes = [
        ct.c_void_p, ct.POINTER(ApproxScaleTransition)]
    library.nrPlanDebugAppendApproxDecoderTransition.restype = ct.c_int
    library.nrPlanDebugGetFixedTransitionState.argtypes = [ct.c_void_p] + [
        ct.POINTER(ct.c_uint64)] * 4
    library.nrPlanDebugGetFixedTransitionState.restype = ct.c_int
    library.nrPlanConfigureApproxSplit512Blocks.argtypes = [ct.c_void_p,
        ct.POINTER(ApproxSplit512Block), ct.c_uint32]
    library.nrPlanConfigureApproxSplit512Blocks.restype = ct.c_int
    handle = ct.c_void_p()
    dummy = torch.empty(4, dtype=torch.float16, device='cuda')
    actual = torch.empty_like(expected)
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
             len(native_rows), 0), 'configure stages')
        if native_split is not None:
            call(library.nrPlanConfigureApproxSplit512Blocks(handle,
                 ct.byref(native_split), 1), 'configure split stage')
        call(library.nrPlanDebugConfigureApproxEncoderTransition(
            handle, ct.byref(native_transition)), 'configure one encoder transition')
        if roundtrip:
            call(library.nrPlanDebugAppendApproxDecoderTransition(
                handle, ct.byref(native_decoder)), 'append one decoder transition')
        call(library.nrPlanInitializeArenaFromDevice(handle, first['raw_resident_offset'],
             ct.c_void_p(initial.data_ptr()), initial.numel()), 'initialize input')
        call(library.nrPlanFinalize(handle), 'finalize')
        binding = BindingsV3(ct.sizeof(BindingsV3), 3, dummy.data_ptr(), dummy.data_ptr(),
            0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8,
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0, 1, 1, 0)
        call(library.nrPlanSubmitV3(handle, ct.byref(binding), None, None), 'submit')
        entries, post, qkv, scratch_bytes = (ct.c_uint64() for _ in range(4))
        call(library.nrPlanDebugGetFixedBoundaryState(handle, ct.byref(entries),
             ct.byref(post), ct.byref(qkv), ct.byref(scratch_bytes)), 'stage state')
        enc, dec, target_bytes, skip_bytes = (ct.c_uint64() for _ in range(4))
        call(library.nrPlanDebugGetFixedTransitionState(handle, ct.byref(enc),
             ct.byref(dec), ct.byref(target_bytes), ct.byref(skip_bytes)),
             'transition state')
        final_channels = channels if roundtrip else target_channels
        final_windows = native_rows[-1].windows if roundtrip else \
            native_split.windows if native_split is not None else native_rows[-1].windows
        final_scalars = final_windows * 64 * final_channels
        call(library.nrPlanDebugCopyFixedBoundaryScratchToDevice(handle,
             final_scalars * (7 if native_split is not None and not roundtrip else 6),
             ct.c_void_p(actual.data_ptr()), actual.numel()),
             'copy final output')
        error = metric(actual, expected)
        resources = ResourceStats()
        call(library.nrPlanGetResourceStats(handle, ct.byref(resources)), 'resources')
        approximate = bool(error['finite'] and error['nrmse'] <= .1 and
                           error['max_absolute_error'] <= .25)
        return {
            'checks_pass': approximate and entries.value == len(native_rows) +
                           int(native_split is not None) and
                           post.value == len(native_rows) + int(native_split is not None) and
                           qkv.value == len(native_rows) + int(native_split is not None) and enc.value == 1 and
                           dec.value == int(roundtrip),
            'approximate_cross_scale_gate_pass': approximate,
            'records': [row['record_number'] for row in
                        ([first, second, third] if roundtrip else [first, second])],
            'geometry': ([[8, 8, channels], [4, 4, target_channels],
                          [8, 8, channels]] if roundtrip else
                         [[8, 8, channels], [4, 4, target_channels]]),
            'error_vs_reference': error,
            'stage_state': {'entries': entries.value, 'post': post.value,
                            'qkv': qkv.value, 'scratch_bytes': scratch_bytes.value},
            'transition_state': {'encoder_chains': enc.value, 'decoder_chains': dec.value,
                                 'target_bytes': target_bytes.value,
                                 'skip_pool_bytes': skip_bytes.value},
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
    require_gpu_tests_enabled('native cross-scale resident chain gate')
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
    for name in ('dll', 'package', 'arena', 'topology', 'transitions', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--split-topology', type=Path)
    parser.add_argument('--channels', type=int, choices=(32, 64, 128, 256), default=64)
    parser.add_argument('--roundtrip', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    rows = select(args)
    preflight = {'status': 'CPU_PREFLIGHT_PASS_GPU_NOT_EXECUTED',
                 'dll_sha256': sha256(args.dll),
                 'records': ([rows[0]['record_number'], rows[1]['record_number'],
                              rows[3]['record_number']] if args.roundtrip else
                             [rows[0]['record_number'], rows[1]['record_number']]),
                 'geometry': ([[8, 8, args.channels], [4, 4, args.channels * 2],
                               [8, 8, args.channels]] if args.roundtrip else
                              [[8, 8, args.channels], [4, 4, args.channels * 2]])}
    if not args.execute:
        save(args.output, preflight)
        return True
    require_gpu_tests_enabled('native cross-scale resident chain gate')
    if args.child:
        return child(args)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--execute']
    for name in ('dll', 'package', 'arena', 'topology', 'transitions', 'output'):
        command += [f'--{name}', str(getattr(args, name).resolve())]
    if args.split_topology:
        command += ['--split-topology', str(args.split_topology.resolve())]
    command += ['--channels', str(args.channels)]
    if args.roundtrip:
        command.append('--roundtrip')
    return supervise(command, args.output, timeout=120)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 2)
