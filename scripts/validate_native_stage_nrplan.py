"""Preflight and supervised one-block correctness gate for native NRPlan stages.

The default mode is CPU-only.  GPU execution requires ``--execute`` and the
repository safety record must explicitly authorize the next minimal gate.  The
gate records one origin-zero C64/C128/C256 block at 8x8 feature geometry, submits
one reset-only frame, and compares its resident E4M3 output with the PyTorch
reference.  It is a bring-up test, not a performance or full-network claim.
"""
from __future__ import annotations

import argparse
import atexit
import ctypes as ct
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

from audit_native_stage_package import audit as audit_package
from gpu_safety import require_gpu_tests_enabled
from native_cpp_nr_plan import (ApproxStageBlock, ArenaRegion, BindingsV3, Desc,
                                GraphStats, KernelNodeInfo, ModelPackageStats, PerformanceStats,
                                ResourceStats, ShapeDescV3,
                                approximate_stage_descriptor_array,
                                validate_v3_layouts)
from validate_rocm_lifecycle import save, supervise


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def e4_host_round_shift(value: int, shift: int) -> int:
    base = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return base + int(remainder > halfway or
                      (remainder == halfway and bool(base & 1)))


def e4_host_from_half_bits(bits: int) -> int:
    sign = (bits >> 8) & 0x80
    exponent = (bits >> 10) & 31
    mantissa = bits & 1023
    if exponent == 31 and mantissa:
        return 0x7f
    if exponent < 9:
        code = min(e4_host_round_shift(1024 + mantissa, 16 - exponent), 8)
    else:
        rounded = e4_host_round_shift(mantissa, 7)
        carry = rounded >= 8
        code = ((exponent - 8 + int(carry)) << 3) | (0 if carry else rounded)
        code = min(code, 0x7e)
    return sign | code


def expected_e4_lut() -> bytes:
    return bytes(e4_host_from_half_bits(bits) for bits in range(65536))


def preflight(args) -> tuple[dict, dict, dict]:
    dll = args.dll.resolve()
    build_path = dll.parent / 'build.json'
    if not dll.is_file() or not build_path.is_file():
        raise FileNotFoundError('NRPlan DLL/build.json pair is required')
    build = json.loads(build_path.read_text(encoding='utf-8-sig'))
    if build.get('gpu_executed') is not False or \
            build.get('binary_sha256', {}).get('dll') != sha256(dll):
        raise ValueError('NRPlan DLL does not match its compile-only manifest')
    package_raw = args.package.read_bytes()
    cache = json.loads(args.cache.read_text(encoding='utf-8'))
    arena = json.loads(args.arena.read_text(encoding='utf-8'))
    topology = json.loads(args.topology.read_text(encoding='utf-8'))
    package_audit = audit_package(package_raw, cache, topology)
    descriptors = approximate_stage_descriptor_array(topology)
    matches = [(row, descriptor) for row, descriptor in zip(topology['blocks'], descriptors)
               if row['record_number'] == args.record_number]
    if len(matches) != 1:
        raise ValueError('requested record is not one unique native wide-stage block')
    block, descriptor = matches[0]
    if (block['origin_x'], block['origin_y']) != (0, 0):
        raise ValueError('minimal one-window gate requires an origin-zero block')
    if arena.get('workspace_bytes', 0) > 5_000_000_000 or \
            package_audit['gfx1201_weights_sha256'] != cache['weights_sha256']:
        raise ValueError('arena budget or selected package cache is invalid')
    result = {
        'schema': 1,
        'status': 'CPU_PREFLIGHT_PASS_GPU_NOT_EXECUTED',
        'dll': str(dll),
        'dll_sha256': sha256(dll),
        'package': str(args.package.resolve()),
        'package_sha256': package_audit['package_sha256'],
        'topology': str(args.topology.resolve()),
        'arena': str(args.arena.resolve()),
        'workspace_bytes': arena['workspace_bytes'],
        'selected_record_number': block['record_number'],
        'selected_name': block['name'],
        'channels': block['channels'],
        'gate_feature_geometry': [8, 8],
        'gate_windows': 1,
        'expected_graph_kernel_nodes': ((6 if block['channels'] == 32 else 8)
                                        if getattr(args, 'stop_after_qkv', False)
                                        else (9 if block['channels'] == 32 else 11)),
        'ctypes_layouts': validate_v3_layouts(),
        'complete_native_topology': False,
        'runtime_accepted': False,
        'gpu_executed': False,
    }
    return result, arena, block


def bind(library) -> None:
    library.nrPlanCreate.argtypes = [ct.POINTER(Desc), ct.POINTER(ct.c_void_p)]
    library.nrPlanSetPrecisionProfile.argtypes = [ct.c_void_p, ct.c_uint32]
    library.nrPlanSetExecutionMode.argtypes = [ct.c_void_p, ct.c_uint32]
    library.nrPlanDebugGetExecutionState.argtypes = [ct.c_void_p,
        ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint64)]
    library.nrPlanGetStream.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    library.nrPlanLoadModelPackage.argtypes = [ct.c_void_p, ct.c_char_p,
                                               ct.POINTER(ModelPackageStats)]
    library.nrPlanConfigureArena.argtypes = [ct.c_void_p, ct.POINTER(ArenaRegion),
                                             ct.c_uint64]
    library.nrPlanPrepareShape.argtypes = [ct.c_void_p, ct.POINTER(ShapeDescV3)]
    library.nrPlanConfigureApproxStageBlocks.argtypes = [ct.c_void_p,
        ct.POINTER(ApproxStageBlock), ct.c_uint32, ct.c_uint8]
    library.nrPlanInitializeArenaFromDevice.argtypes = [ct.c_void_p, ct.c_uint64,
                                                        ct.c_void_p, ct.c_uint64]
    library.nrPlanFinalize.argtypes = [ct.c_void_p]
    library.nrPlanSubmitV3.argtypes = [ct.c_void_p, ct.POINTER(BindingsV3),
                                       ct.c_void_p, ct.c_void_p]
    library.nrPlanDebugCopyArenaToDevice.argtypes = [ct.c_void_p, ct.c_uint64,
                                                     ct.c_void_p, ct.c_uint64]
    library.nrPlanGetGraphStats.argtypes = [ct.c_void_p, ct.POINTER(GraphStats)]
    library.nrPlanGetKernelNodeInfos.argtypes = [ct.c_void_p, ct.POINTER(KernelNodeInfo),
                                                 ct.c_uint64, ct.POINTER(ct.c_uint64)]
    library.nrPlanDebugDotPrint.argtypes = [ct.c_void_p, ct.c_char_p]
    library.nrPlanDebugGetKernelU64Arguments.argtypes = [ct.c_void_p, ct.c_uint64,
                                                          ct.c_uint32,
                                                          ct.POINTER(ct.c_uint64)]
    library.nrPlanDebugGetOwnedAddresses.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64),
                                                      ct.POINTER(ct.c_uint64),
                                                      ct.POINTER(ct.c_uint64)]
    library.nrPlanDebugCopyStageE4LutToDevice.argtypes = [ct.c_void_p, ct.c_void_p,
                                                           ct.c_uint64]
    library.nrPlanDebugStopAfterStandardQkv.argtypes = [ct.c_void_p, ct.c_uint8]
    library.nrPlanDebugGetFixedBoundaryState.argtypes = [ct.c_void_p,
        ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64),
        ct.POINTER(ct.c_uint64)]
    library.nrPlanDebugCopyFixedBoundaryScratchToDevice.argtypes = [ct.c_void_p,
        ct.c_uint64, ct.c_void_p, ct.c_uint64]
    library.nr_stage_debug_pack_e4x4_c256.argtypes = [ct.c_void_p, ct.c_void_p,
                                                       ct.c_void_p, ct.c_uint64,
                                                       ct.c_int, ct.c_void_p]
    library.nr_stage_debug_pack_e4x4_from_fp16.argtypes = [ct.c_void_p, ct.c_void_p,
        ct.c_void_p, ct.c_uint64, ct.c_void_p]
    library.nr_stage_c256_qkv_norm_fp8.argtypes = [ct.c_void_p, ct.c_void_p,
        ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]
    library.nrPlanGetResourceStats.argtypes = [ct.c_void_p, ct.POINTER(ResourceStats)]
    library.nrPlanGetPerformanceStats.argtypes = [ct.c_void_p,
                                                  ct.POINTER(PerformanceStats)]
    library.nrPlanDestroy.argtypes = [ct.c_void_p]
    for name in ('nrPlanCreate', 'nrPlanSetPrecisionProfile', 'nrPlanSetExecutionMode',
                 'nrPlanDebugGetExecutionState',
                 'nrPlanGetStream', 'nrPlanLoadModelPackage',
                 'nrPlanConfigureArena', 'nrPlanPrepareShape',
                 'nrPlanConfigureApproxStageBlocks', 'nrPlanInitializeArenaFromDevice',
                 'nrPlanFinalize', 'nrPlanSubmitV3', 'nrPlanDebugCopyArenaToDevice',
                  'nrPlanGetGraphStats', 'nrPlanGetKernelNodeInfos',
                  'nrPlanDebugDotPrint', 'nrPlanDebugGetKernelU64Arguments',
                  'nrPlanDebugGetOwnedAddresses', 'nrPlanDebugCopyStageE4LutToDevice',
                  'nrPlanDebugStopAfterStandardQkv',
                   'nrPlanDebugGetFixedBoundaryState',
                   'nrPlanDebugCopyFixedBoundaryScratchToDevice',
                  'nr_stage_debug_pack_e4x4_c256',
                  'nr_stage_debug_pack_e4x4_from_fp16',
                  'nr_stage_c256_qkv_norm_fp8',
                  'nrPlanGetResourceStats',
                 'nrPlanGetPerformanceStats', 'nrPlanDestroy'):
        getattr(library, name).restype = ct.c_int


def call(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f'NRPlan {operation} failed: HIP error {code}')


def metric(actual, expected) -> dict:
    import torch
    actual_f = actual.view(torch.float8_e4m3fn).half().float()
    expected_f = expected.view(torch.float8_e4m3fn).half().float()
    delta = actual_f - expected_f
    denominator = expected_f.square().mean().sqrt()
    finite = bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all())
    actual_raw = actual.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    expected_raw = expected.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {
        'byte_exact': bool(torch.equal(actual, expected)),
        'different_bytes': int((actual != expected).sum()),
        'max_absolute_error': float(delta.abs().max()) if finite else None,
        'nrmse': float(delta.square().mean().sqrt() /
                       denominator.clamp_min(1e-8)) if finite else None,
        'finite': finite,
        'actual_sha256': hashlib.sha256(actual_raw).hexdigest().upper(),
        'expected_sha256': hashlib.sha256(expected_raw).hexdigest().upper(),
        'actual_first_16_bytes': list(actual_raw[:16]),
        'expected_first_16_bytes': list(expected_raw[:16]),
    }


def metric_fp16(actual, expected) -> dict:
    import torch
    delta = actual.float() - expected.float()
    denominator = expected.float().square().mean().sqrt()
    actual_raw = actual.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    expected_raw = expected.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {
        'bit_exact': bool(torch.equal(actual.view(torch.int16), expected.view(torch.int16))),
        'different_values': int((actual.view(torch.int16) != expected.view(torch.int16)).sum()),
        'max_absolute_error': float(delta.abs().max()),
        'nrmse': float(delta.square().mean().sqrt() / denominator.clamp_min(1e-8)),
        'finite': bool(torch.isfinite(actual).all()),
        'actual_sha256': hashlib.sha256(actual_raw).hexdigest().upper(),
        'expected_sha256': hashlib.sha256(expected_raw).hexdigest().upper(),
        'actual_first_16_bytes': list(actual_raw[:16]),
        'expected_first_16_bytes': list(expected_raw[:16]),
    }


def exercise(args, arena: dict, block: dict) -> dict:
    import torch
    from native_model_package import load_package
    from native_packed_swin import RecoveredPackedSwin, gather_packed, scatter_packed
    from native_swin_torch import encode_e4, quantize_e4
    from native_window_attention import normalize_qk, window_attention_prepared

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm GPU required; no CPU/CUDA fallback')
    properties = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in properties.gcnArchName:
        raise RuntimeError('gfx1201 is required')
    torch.cuda.set_per_process_memory_fraction(1_500_000_000 / properties.total_memory)
    torch.manual_seed(1201 + block['record_number'])
    records, _, _ = load_package('local_models/native_single_color_v1',
        'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3')
    channels = block['channels']
    model = RecoveredPackedSwin(records[(block['record_number'], 0)],
                                record_kind=f'swin{channels}').eval().cuda()
    raw = quantize_e4((torch.randn(8 * 8 * channels, device='cuda',
                                   dtype=torch.float16) * .1)).contiguous()
    raw_bytes = encode_e4(raw).contiguous()
    windows, mapping = gather_packed(raw, 8, 8, channels, 0, 0)
    with torch.no_grad():
        post_expected = model.block.ffn(windows[:, model.a_index],
                                        windows[:, model.residual_index])
        attention = model.block.attention
        projection = attention._linear(post_expected, attention.qkv)
        q0, k0, v0 = [part.reshape(-1, 64, attention.heads, 32).transpose(1, 2)
                      for part in projection.chunk(3, dim=-1)]
        q_expected = quantize_e4((normalize_qk(q0) *
            attention.q_scale[None, :, None, None]).half()).contiguous()
        k_expected = quantize_e4(normalize_qk(k0)).contiguous()
        v_expected = quantize_e4(v0).contiguous()
        value_expected = quantize_e4(window_attention_prepared(
            q_expected, k_expected, v_expected, attention.position_bias)).contiguous()
        projected_value = value_expected.transpose(1, 2).reshape(-1, 64, channels)
        output_expected = attention._linear(projected_value, attention.project,
            (post_expected * attention.attention_scale).half()).contiguous()
        expected = encode_e4(quantize_e4(scatter_packed(
            output_expected, mapping, 8, 8))).contiguous()

    library = ct.CDLL(str(args.dll.resolve()))
    bind(library)
    handle = ct.c_void_p()
    descriptor = dict(block, feature_width=8, feature_height=8, windows=1,
                      origin_x=0, origin_y=0)
    fields = [name for name, _ in ApproxStageBlock._fields_
              if name not in ('struct_size', 'channels', 'windows', 'record_number',
                              'feature_width', 'feature_height', 'origin_x', 'origin_y')]
    native_block = ApproxStageBlock(ct.sizeof(ApproxStageBlock), channels, 1,
                                    block['record_number'],
                                    8, 8, 0, 0,
                                    *(descriptor[name] for name in fields))
    regions = []
    for row in arena['regions']:
        regions.append(ArenaRegion(row['offset'], row['bytes'], row['alignment'], 0,
                                   row['name'].encode('utf-8')))
    region_array = (ArenaRegion * len(regions))(*regions)
    dummy_input = torch.empty(4, device='cuda', dtype=torch.float16)
    dummy_output = torch.empty_like(dummy_input)
    output_bytes = torch.empty_like(raw_bytes)
    try:
        call(library.nrPlanCreate(ct.byref(Desc(arena['workspace_bytes'], 0, 1920, 1080)),
                                  ct.byref(handle)), 'create')
        call(library.nrPlanSetPrecisionProfile(handle, 1), 'set approximate profile')
        if args.fixed_sequence:
            call(library.nrPlanSetExecutionMode(handle, 1), 'set fixed command sequence')
        if args.stop_after_qkv:
            call(library.nrPlanDebugStopAfterStandardQkv(handle, 1),
                 'enable QKV diagnostic stop')
        package_stats = ModelPackageStats()
        call(library.nrPlanLoadModelPackage(handle, str(args.package.resolve()).encode('utf-8'),
                                            ct.byref(package_stats)), 'load model package')
        call(library.nrPlanConfigureArena(handle, region_array, len(regions)), 'configure arena')
        shape = ShapeDescV3(ct.sizeof(ShapeDescV3), 8, 8, 8, 8, 0, 0, 8, 8,
                            0, 1, 2, 3)
        call(library.nrPlanPrepareShape(handle, ct.byref(shape)), 'prepare shape')
        call(library.nrPlanConfigureApproxStageBlocks(handle, ct.byref(native_block), 1, 0),
             'configure one block')
        call(library.nrPlanInitializeArenaFromDevice(handle, block['raw_resident_offset'],
             ct.c_void_p(raw_bytes.data_ptr()), raw_bytes.numel()), 'initialize resident input')
        # Poison all three resident outputs before capture. A missing or partial
        # QKV publisher can therefore never be mistaken for allocator leftovers.
        qkv_poisons = {
            'q': torch.full_like(encode_e4(q_expected), 0xA1),
            'k': torch.full_like(encode_e4(k_expected), 0xA2),
            'v': torch.full_like(encode_e4(v_expected), 0xA3),
        }
        post_resident_poison = torch.full_like(encode_e4(post_expected), 0xA4)
        # The poison tensors are produced on PyTorch's default stream while
        # NRPlan owns a private stream. Complete the producers before any D2D
        # initialization so this diagnostic cannot race two HIP streams.
        torch.cuda.synchronize()
        for label, offset in (('q', block['q_offset']), ('k', block['k_offset']),
                              ('v', block['v_offset'])):
            poison = qkv_poisons[label]
            call(library.nrPlanInitializeArenaFromDevice(handle, offset,
                 ct.c_void_p(poison.data_ptr()), poison.numel()),
                 f'poison {label} resident output')
        call(library.nrPlanInitializeArenaFromDevice(handle, block['post_resident_offset'],
             ct.c_void_p(post_resident_poison.data_ptr()), post_resident_poison.numel()),
             'poison post-FFN resident output')
        call(library.nrPlanFinalize(handle), 'finalize graph')
        def copy_arena(offset, template, operation='copy diagnostic intermediate'):
            target = torch.empty_like(template)
            call(library.nrPlanDebugCopyArenaToDevice(handle, offset,
                 ct.c_void_p(target.data_ptr()), target.numel() * target.element_size()),
                 operation)
            return target

        poison_before_submit = {
            label: copy_arena(offset, qkv_poisons[label],
                              f'copy pre-submit {label} poison')
            for label, offset in (('q', block['q_offset']), ('k', block['k_offset']),
                                  ('v', block['v_offset']))
        }
        if not all(torch.equal(poison_before_submit[label], qkv_poisons[label])
                   for label in ('q', 'k', 'v')):
            raise RuntimeError('Q/K/V arena poison changed before graph submission')
        post_resident_poison_before_submit = copy_arena(
            block['post_resident_offset'], post_resident_poison,
            'copy pre-submit post-FFN resident poison')
        if not torch.equal(post_resident_poison_before_submit, post_resident_poison):
            raise RuntimeError('post-FFN resident arena poison changed before submission')
        graph = GraphStats()
        call(library.nrPlanGetGraphStats(handle, ct.byref(graph)), 'get graph stats')
        node_count = ct.c_uint64()
        call(library.nrPlanGetKernelNodeInfos(handle, None, 0, ct.byref(node_count)),
             'count graph kernel nodes')
        node_values = (KernelNodeInfo * node_count.value)()
        call(library.nrPlanGetKernelNodeInfos(handle, node_values, node_count.value,
             ct.byref(node_count)), 'read graph kernel nodes')
        kernel_nodes = [{
            'kernel_ordinal': int(node.kernel_ordinal),
            'graph_node_ordinal': int(node.graph_node_ordinal),
            'name': bytes(node.name).split(b'\0', 1)[0].decode('utf-8', 'replace'),
            'grid': [int(node.grid_x), int(node.grid_y), int(node.grid_z)],
            'block': [int(node.block_x), int(node.block_y), int(node.block_z)],
            'registers_per_thread': int(node.registers_per_thread),
            'static_shared_bytes': int(node.static_shared_bytes),
        } for node in node_values]
        pack_ordinal = 3 if channels == 32 else 5
        captured_arguments = (ct.c_uint64 * 4)()
        call(library.nrPlanDebugGetKernelU64Arguments(handle, pack_ordinal, 4,
             captured_arguments), 'read QKV pack graph arguments')
        workspace_address = ct.c_uint64()
        weights_address = ct.c_uint64()
        lut_address = ct.c_uint64()
        call(library.nrPlanDebugGetOwnedAddresses(handle, ct.byref(workspace_address),
             ct.byref(weights_address), ct.byref(lut_address)), 'read owned addresses')
        captured = [int(value) for value in captured_arguments]
        expected_arguments = [
            workspace_address.value + block['qkv_projection_fp16_offset'],
            workspace_address.value + block['q_offset'],
            lut_address.value,
            channels * 64,
        ]
        kernel_argument_audit = {
            'kernel_ordinal': pack_ordinal,
            'captured_u64': captured,
            'expected_u64': expected_arguments,
            'exact': captured == expected_arguments,
            'workspace_address': workspace_address.value,
            'weights_address': weights_address.value,
            'stage_e4_lut_address': lut_address.value,
        }
        pointer_specs = {
            'qkv_project': (1 if channels == 32 else 3, [
                workspace_address.value + block['post_resident_offset'],
                weights_address.value + block['qkv_weight_offset'],
                weights_address.value + block['permutation_weight_offset'],
                workspace_address.value + block['qkv_projection_fp16_offset'],
            ]),
            'qkv_norm': (2 if channels == 32 else 4, [
                workspace_address.value + block['qkv_projection_fp16_offset'],
                weights_address.value + block['qscale_weight_offset'],
            ]),
            'qkv_pack_q': (pack_ordinal, expected_arguments[:3]),
            'qkv_pack_k': (pack_ordinal + 1, [
                workspace_address.value + block['qkv_projection_fp16_offset'],
                workspace_address.value + block['k_offset'],
                lut_address.value,
            ]),
            'qkv_pack_v': (pack_ordinal + 2, [
                workspace_address.value + block['qkv_projection_fp16_offset'],
                workspace_address.value + block['v_offset'],
                lut_address.value,
            ]),
        }
        if channels != 32:
            pointer_specs['ffn_mix'] = (1, [
                workspace_address.value + block['grouped_offset'],
                weights_address.value + block['ffn_mix_weight_offset'],
                workspace_address.value + block['seed_fp16_offset'],
                workspace_address.value + block['post_fp16_offset'],
            ])
            pointer_specs['ffn_resident_publish'] = (2, [
                workspace_address.value + block['post_fp16_offset'],
                workspace_address.value + block['post_resident_offset'],
                lut_address.value,
            ])
        if not args.stop_after_qkv:
            pointer_specs.update({
            'attention': (6 if channels == 32 else 8, [
                workspace_address.value + block['q_offset'],
                workspace_address.value + block['k_offset'],
                workspace_address.value + block['v_offset'],
                weights_address.value + block['position_bias_weight_offset'],
                workspace_address.value + block['value_offset'],
            ]),
            'project': (7 if channels == 32 else 9, [
                workspace_address.value + block['post_fp16_offset'],
                workspace_address.value + block['value_offset'],
                weights_address.value + block['project_weight_offset'],
                weights_address.value + block['residual_scale_weight_offset'],
                weights_address.value + block['permutation_weight_offset'],
                workspace_address.value + block['output_fp16_offset'],
                workspace_address.value + block['post_resident_offset'],
            ]),
            'scatter': (8 if channels == 32 else 10, [
                workspace_address.value + block['post_resident_offset'],
                workspace_address.value + block['next_resident_offset'],
            ]),
            })
        pointer_audits = {}
        for label, (ordinal, expected_pointers) in pointer_specs.items():
            values = (ct.c_uint64 * len(expected_pointers))()
            call(library.nrPlanDebugGetKernelU64Arguments(handle, ordinal,
                 len(expected_pointers), values), f'read {label} graph pointers')
            captured_pointers = [int(value) for value in values]
            pointer_audits[label] = {
                'kernel_ordinal': ordinal,
                'captured': captured_pointers,
                'expected': expected_pointers,
                'exact': captured_pointers == expected_pointers,
            }
        lut_expected = expected_e4_lut()
        lut_device = torch.empty(len(lut_expected), dtype=torch.uint8, device='cuda')
        call(library.nrPlanDebugCopyStageE4LutToDevice(handle,
             ct.c_void_p(lut_device.data_ptr()), lut_device.numel()),
             'copy stage E4M3 LUT')
        lut_actual = lut_device.cpu().numpy().tobytes()
        stage_e4_lut_audit = {
            'bytes': len(lut_expected),
            'exact': lut_actual == lut_expected,
            'actual_sha256': hashlib.sha256(lut_actual).hexdigest().upper(),
            'expected_sha256': hashlib.sha256(lut_expected).hexdigest().upper(),
            'different_bytes': sum(a != b for a, b in zip(lut_actual, lut_expected)),
        }
        dot_path = args.output / 'captured_graph.dot'
        call(library.nrPlanDebugDotPrint(handle, str(dot_path.resolve()).encode('utf-8')),
             'write captured graph DOT')
        binding = BindingsV3(ct.sizeof(BindingsV3), 3, dummy_input.data_ptr(),
            dummy_output.data_ptr(), 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 8, 8,
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0, 1, 1, 0)
        call(library.nrPlanSubmitV3(handle, ct.byref(binding), None, None), 'submit')
        execution_mode = ct.c_uint32()
        fixed_sequence_submits = ct.c_uint64()
        call(library.nrPlanDebugGetExecutionState(handle, ct.byref(execution_mode),
             ct.byref(fixed_sequence_submits)), 'read execution state')
        execution_state = {
            'mode': execution_mode.value,
            'fixed_sequence_submits': fixed_sequence_submits.value,
        }
        fixed_boundary_state = None
        fixed_boundary_snapshots = None
        if args.fixed_sequence and channels in (64, 128, 256):
            stage_entries = ct.c_uint64()
            post_copies = ct.c_uint64()
            qkv_copies = ct.c_uint64()
            scratch_bytes = ct.c_uint64()
            call(library.nrPlanDebugGetFixedBoundaryState(handle,
                 ct.byref(stage_entries), ct.byref(post_copies), ct.byref(qkv_copies),
                 ct.byref(scratch_bytes)), 'read fixed boundary state')
            fixed_boundary_state = {
                'stage_entries': stage_entries.value,
                'post_copies': post_copies.value,
                'qkv_copies': qkv_copies.value,
                'scratch_bytes': scratch_bytes.value,
            }
            scalar_bytes = channels * 64
            fixed_boundary_snapshots = []
            for part in range(4 if args.stop_after_qkv else 7):
                target = torch.empty(scalar_bytes, dtype=torch.uint8, device='cuda')
                call(library.nrPlanDebugCopyFixedBoundaryScratchToDevice(handle,
                     part * scalar_bytes, ct.c_void_p(target.data_ptr()), scalar_bytes),
                     f'copy fixed boundary scratch part {part}')
                fixed_boundary_snapshots.append(target)
        call(library.nrPlanDebugCopyArenaToDevice(handle, block['next_resident_offset'],
             ct.c_void_p(output_bytes.data_ptr()), output_bytes.numel()), 'copy resident output')
        if fixed_boundary_snapshots is not None and not args.stop_after_qkv:
            output_bytes = fixed_boundary_snapshots[6].reshape_as(output_bytes)
        resources = ResourceStats()
        call(library.nrPlanGetResourceStats(handle, ct.byref(resources)), 'get resource stats')
        error = metric(output_bytes, expected)
        numerical_gate_pass = (error['finite'] and error['nrmse'] <= 0.02 and
                               error['max_absolute_error'] <= 0.25)
        approximate_performance_gate_pass = (error['finite'] and error['nrmse'] <= 0.05 and
                                             error['max_absolute_error'] <= 0.0625)
        diagnostics = None
        if args.diagnose_intermediates:
            def copy(offset, template):
                return copy_arena(offset, template)
            actual_post = copy(block['post_fp16_offset'], post_expected)
            actual_post_resident = copy(block['post_resident_offset'],
                                        encode_e4(post_expected))
            actual_projection = copy(block['qkv_projection_fp16_offset'], projection)
            actual_q = copy(block['q_offset'], encode_e4(q_expected))
            actual_k = copy(block['k_offset'], encode_e4(k_expected))
            actual_v = copy(block['v_offset'], encode_e4(v_expected))
            direct_pack = None
            direct_plan_arena_pack = None
            direct_plan_stream_pack = None
            direct_qkv_entry = None
            direct_post_resident_publish = None
            if channels == 256:
                direct_post_resident = torch.empty_like(actual_post_resident)
                call(library.nr_stage_debug_pack_e4x4_from_fp16(
                     ct.c_void_p(actual_post.data_ptr()),
                     ct.c_void_p(direct_post_resident.data_ptr()),
                     ct.c_void_p(lut_address.value), actual_post.numel(),
                     ct.c_void_p(torch.cuda.current_stream().cuda_stream)),
                     'direct post-FFN resident publish')
                torch.cuda.synchronize()
                direct_post_resident_publish = metric(
                    direct_post_resident, encode_e4(actual_post))
                direct_parts = []
                for part in range(3):
                    target = torch.empty(channels * 64, dtype=torch.uint8, device='cuda')
                    call(library.nr_stage_debug_pack_e4x4_c256(
                         ct.c_void_p(actual_projection.data_ptr()),
                         ct.c_void_p(target.data_ptr()), ct.c_void_p(lut_address.value),
                         target.numel(), part,
                         ct.c_void_p(torch.cuda.current_stream().cuda_stream)),
                         f'direct pack part {part}')
                    direct_parts.append(target)
                torch.cuda.synchronize()
                projection_parts = actual_projection.reshape(64, 3, channels)
                direct_pack = {}
                for part, label in enumerate(('q', 'k', 'v')):
                    expected_direct = encode_e4(projection_parts[:, part]
                        .reshape(64, channels // 32, 32).permute(1, 0, 2)
                        .contiguous()).reshape(-1)
                    direct_pack[label] = metric(direct_parts[part], expected_direct)
                graph_parts = (actual_q, actual_k, actual_v)
                for part, offset in enumerate((block['q_offset'], block['k_offset'],
                                               block['v_offset'])):
                    call(library.nr_stage_debug_pack_e4x4_c256(
                         ct.c_void_p(workspace_address.value +
                                     block['qkv_projection_fp16_offset']),
                         ct.c_void_p(workspace_address.value + offset),
                         ct.c_void_p(lut_address.value), channels * 64, part, None),
                         f'direct plan-arena pack part {part}')
                torch.cuda.synchronize()
                repaired_parts = (
                    copy(block['q_offset'], graph_parts[0]),
                    copy(block['k_offset'], graph_parts[1]),
                    copy(block['v_offset'], graph_parts[2]),
                )
                direct_plan_arena_pack = {}
                for part, label in enumerate(('q', 'k', 'v')):
                    expected_direct = encode_e4(projection_parts[:, part]
                        .reshape(64, channels // 32, 32).permute(1, 0, 2)
                        .contiguous()).reshape(-1)
                    direct_plan_arena_pack[label] = metric(repaired_parts[part].reshape(-1),
                                                            expected_direct)
                plan_stream = ct.c_void_p()
                call(library.nrPlanGetStream(handle, ct.byref(plan_stream)),
                     'read plan stream')
                for part, offset in enumerate((block['q_offset'], block['k_offset'],
                                               block['v_offset'])):
                    call(library.nr_stage_debug_pack_e4x4_c256(
                         ct.c_void_p(workspace_address.value +
                                     block['qkv_projection_fp16_offset']),
                         ct.c_void_p(workspace_address.value + offset),
                         ct.c_void_p(lut_address.value), channels * 64, part,
                         plan_stream), f'direct plan-stream pack part {part}')
                plan_stream_parts = (
                    copy(block['q_offset'], graph_parts[0]),
                    copy(block['k_offset'], graph_parts[1]),
                    copy(block['v_offset'], graph_parts[2]),
                )
                direct_plan_stream_pack = {}
                for part, label in enumerate(('q', 'k', 'v')):
                    expected_direct = encode_e4(projection_parts[:, part]
                        .reshape(64, channels // 32, 32).permute(1, 0, 2)
                        .contiguous()).reshape(-1)
                    direct_plan_stream_pack[label] = metric(
                        plan_stream_parts[part].reshape(-1), expected_direct)
                call(library.nr_stage_c256_qkv_norm_fp8(
                     ct.c_void_p(workspace_address.value + block['post_resident_offset']),
                     ct.c_void_p(weights_address.value + block['qkv_weight_offset']),
                     ct.c_void_p(weights_address.value + block['qscale_weight_offset']),
                     ct.c_void_p(weights_address.value + block['permutation_weight_offset']),
                     ct.c_void_p(workspace_address.value +
                                 block['qkv_projection_fp16_offset']),
                     ct.c_void_p(workspace_address.value + block['q_offset']),
                     ct.c_void_p(workspace_address.value + block['k_offset']),
                     ct.c_void_p(workspace_address.value + block['v_offset']),
                     ct.c_void_p(lut_address.value), 1, plan_stream),
                     'direct exported QKV entry on plan stream')
                entry_projection = copy(block['qkv_projection_fp16_offset'], projection)
                entry_parts = (
                    copy(block['q_offset'], graph_parts[0]).reshape(-1),
                    copy(block['k_offset'], graph_parts[1]).reshape(-1),
                    copy(block['v_offset'], graph_parts[2]).reshape(-1),
                )
                entry_projection_parts = entry_projection.reshape(64, 3, channels)
                direct_qkv_entry = {}
                for part, label in enumerate(('q', 'k', 'v')):
                    expected_entry = encode_e4(entry_projection_parts[:, part]
                        .reshape(64, channels // 32, 32).permute(1, 0, 2)
                        .contiguous()).reshape(-1)
                    direct_qkv_entry[label] = metric(entry_parts[part], expected_entry)
            diagnostic_tensors = {
                'post_ffn_fp16': actual_post,
                'post_ffn_resident_e4m3': actual_post_resident,
                'qkv_projection_fp16': actual_projection,
                'q_e4m3': actual_q,
                'k_e4m3': actual_k,
                'v_e4m3': actual_v,
            }
            diagnostic_files = {}
            for name, tensor in diagnostic_tensors.items():
                raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                path = args.output / f'{name}.raw'
                path.write_bytes(raw)
                diagnostic_files[name] = {
                    'file': path.name, 'bytes': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest().upper(),
                }
            # Recompute the expected Q/K/V from the exact projection produced
            # by the native kernel, on CPU. This separates QKV projection error
            # from normalization/quantization/layout error without launching
            # another neural GPU kernel after the one captured graph submit.
            projection_cpu = actual_projection.cpu()
            pq, pk, pv = [part.reshape(-1, 64, attention.heads, 32).transpose(1, 2)
                          for part in projection_cpu.chunk(3, dim=-1)]
            qscale_cpu = attention.q_scale.detach().cpu()
            q_from_native_projection = encode_e4(quantize_e4((normalize_qk(pq) *
                qscale_cpu[None, :, None, None]).half()).contiguous())
            k_from_native_projection = encode_e4(
                quantize_e4(normalize_qk(pk)).contiguous())
            v_from_native_projection = encode_e4(quantize_e4(pv).contiguous())
            inplace_projection_parts = actual_projection.reshape(64, 3, channels)
            inplace_projection_resident = [encode_e4(inplace_projection_parts[:, part]
                .reshape(64, channels // 32, 32).permute(1, 0, 2).contiguous()).reshape(-1)
                for part in range(3)]
            diagnostics = {
                'post_ffn_fp16': metric_fp16(actual_post, post_expected),
                'post_ffn_resident_e4m3': metric(actual_post_resident,
                                                  encode_e4(post_expected)),
                'post_ffn_resident_vs_native_fp16': metric(
                    actual_post_resident, encode_e4(actual_post)),
                'qkv_projection_fp16': metric_fp16(actual_projection, projection),
                'q_e4m3': metric(actual_q, encode_e4(q_expected)),
                'k_e4m3': metric(actual_k, encode_e4(k_expected)),
                'v_e4m3': metric(actual_v, encode_e4(v_expected)),
                'q_e4m3_vs_native_projection': metric(actual_q.cpu(),
                                                       q_from_native_projection),
                'k_e4m3_vs_native_projection': metric(actual_k.cpu(),
                                                       k_from_native_projection),
                'v_e4m3_vs_native_projection': metric(actual_v.cpu(),
                                                       v_from_native_projection),
                'attention_value_e4m3': metric(
                    copy(block['value_offset'], encode_e4(value_expected)),
                    encode_e4(value_expected)),
                'output_projection_fp16': metric_fp16(
                    copy(block['output_fp16_offset'], output_expected), output_expected),
                'output_projection_e4m3_before_scatter': metric(
                    copy(block['post_resident_offset'],
                         encode_e4(quantize_e4(output_expected))),
                    encode_e4(quantize_e4(output_expected))),
                'direct_e4x4_pack_vs_projection': direct_pack,
                'direct_plan_arena_e4x4_pack_vs_projection': direct_plan_arena_pack,
                'direct_plan_stream_e4x4_pack_vs_projection': direct_plan_stream_pack,
                'direct_exported_qkv_entry_vs_projection': direct_qkv_entry,
                'direct_post_resident_publish_vs_native_fp16':
                    direct_post_resident_publish,
                'fixed_boundary_scratch': None if fixed_boundary_snapshots is None else {
                    'post_vs_native_fp16': metric(fixed_boundary_snapshots[0].reshape_as(
                                                  encode_e4(actual_post)),
                                                  encode_e4(actual_post)),
                    'q_vs_inplace_projection': metric(fixed_boundary_snapshots[1].reshape_as(
                        inplace_projection_resident[0]), inplace_projection_resident[0]),
                    'k_vs_inplace_projection': metric(fixed_boundary_snapshots[2].reshape_as(
                        inplace_projection_resident[1]), inplace_projection_resident[1]),
                    'v_vs_inplace_projection': metric(fixed_boundary_snapshots[3].reshape_as(
                        inplace_projection_resident[2]), inplace_projection_resident[2]),
                    'attention_value': None if args.stop_after_qkv else metric(
                        fixed_boundary_snapshots[4].reshape_as(encode_e4(value_expected)),
                        encode_e4(value_expected)),
                    'project_resident': None if args.stop_after_qkv else metric(
                        fixed_boundary_snapshots[5].reshape_as(
                            encode_e4(quantize_e4(output_expected))),
                        encode_e4(quantize_e4(output_expected))),
                    'scatter_output': None if args.stop_after_qkv else metric(
                        fixed_boundary_snapshots[6].reshape_as(expected), expected),
                },
                'raw_files': diagnostic_files,
            }
        return {
            'checks_pass': numerical_gate_pass and kernel_argument_audit['exact'] and \
                all(row['exact'] for row in pointer_audits.values()) and \
                stage_e4_lut_audit['exact'] and \
                graph.kernel_nodes == ((6 if channels == 32 else 8)
                    if args.stop_after_qkv else (9 if channels == 32 else 11)),
            'lifecycle_gate_pass': True,
            'numerical_gate_pass': numerical_gate_pass,
            'approximate_performance_gate_pass': approximate_performance_gate_pass,
            'numerical_gate_limits': {'nrmse': 0.02, 'max_absolute_error': 0.25},
            'accuracy_track': 'approximate_fp8_not_promoted',
            'execution_state': execution_state,
            'fixed_boundary_state': fixed_boundary_state,
            'error_vs_reference': error,
            'intermediate_diagnostics': diagnostics,
            'graph': {'total_nodes': graph.total_nodes, 'kernel_nodes': graph.kernel_nodes,
                      'memcpy_nodes': graph.memcpy_nodes, 'kernel_details': kernel_nodes,
                      'dot_file': dot_path.name, 'dot_sha256': sha256(dot_path)},
            'kernel_argument_audit': kernel_argument_audit,
            'kernel_pointer_audits': pointer_audits,
            'stage_e4_lut_audit': stage_e4_lut_audit,
            'qkv_resident_poison': {
                'bytes': {'q': 0xA1, 'k': 0xA2, 'v': 0xA3},
                'bytes_per_buffer': next(iter(qkv_poisons.values())).numel(),
                'verified_before_submit': True,
                'remaining_after_submit': {
                    'q': int((actual_q == 0xA1).sum()) if diagnostics is not None else None,
                    'k': int((actual_k == 0xA2).sum()) if diagnostics is not None else None,
                    'v': int((actual_v == 0xA3).sum()) if diagnostics is not None else None,
                },
            },
            'post_resident_poison': {
                'byte': 0xA4,
                'bytes': post_resident_poison.numel(),
                'verified_before_submit': True,
                'remaining_after_submit': int((actual_post_resident == 0xA4).sum())
                    if diagnostics is not None else None,
            },
            'resource': {'workspace_bytes': resources.workspace_bytes,
                         'weight_bytes': resources.weight_bytes,
                         'owns_workspace': bool(resources.owns_workspace),
                         'owns_weights': bool(resources.owns_weights),
                         'owns_graph_source': bool(resources.owns_graph_source),
                         'owns_graph_executable': bool(resources.owns_graph_executable),
                         'graph_references_external_allocations':
                             bool(resources.graph_references_external_allocations),
                         'deployment_ready': bool(resources.deployment_ready),
                         'native_stage_block_count': resources.native_stage_block_count,
                         'complete_native_topology': bool(resources.complete_native_topology)},
            'device': properties.name,
            'architecture': properties.gcnArchName,
            'gpu_work_cancelled': False,
            'automatic_retry': False,
        }
    finally:
        if handle.value:
            call(library.nrPlanDestroy(handle), 'destroy')


def child(args) -> bool:
    require_gpu_tests_enabled('native NRPlan one-block gate')
    preflight_result, arena, block = preflight(args)
    def phase(name):
        with (args.output / 'phases.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase': name, 'pid': os.getpid()}) + '\n')
    atexit.register(phase, 'python_atexit')
    phase('child_enter')
    import torch
    try:
        result = exercise(args, arena, block)
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
    result['checks_pass'] &= result['allocated_after_release_bytes'] == 0 and \
        result['reserved_after_release_bytes'] == 0
    result['preflight'] = preflight_result
    phase('resources_released')
    save(args.output / 'child.json', result)
    return result['checks_pass']


def main() -> bool:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--topology', type=Path, required=True)
    parser.add_argument('--record-number', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--diagnose-intermediates', action='store_true')
    parser.add_argument('--stop-after-qkv', action='store_true')
    parser.add_argument('--fixed-sequence', action='store_true')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    preflight_result, _, _ = preflight(args)
    if not args.execute:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, preflight_result)
        print(json.dumps(preflight_result, indent=2))
        return True
    require_gpu_tests_enabled('native NRPlan one-block gate')
    if args.child:
        return child(args)
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--child', '--execute',
        '--dll', str(args.dll.resolve()), '--package', str(args.package.resolve()),
        '--cache', str(args.cache.resolve()), '--arena', str(args.arena.resolve()),
        '--topology', str(args.topology.resolve()), '--record-number', str(args.record_number),
        '--output', str(args.output.resolve())]
    if args.diagnose_intermediates:
        command.append('--diagnose-intermediates')
    if args.stop_after_qkv:
        command.append('--stop-after-qkv')
    if args.fixed_sequence:
        command.append('--fixed-sequence')
    return supervise(command, args.output, timeout=120)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 2)
