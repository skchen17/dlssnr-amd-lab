"""Private original-record coverage and explicit remaining full-graph contracts.

Coverage counts are NOT progress percentages or evidence of GPU execution.
Quality diagnostics cannot prevent constructing the missing mathematical graph.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ORIGINAL_SHA = 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5'
PLAIN = {32: (1, 2, 3, 67, 68, 69), 64: (5, 6, 7, 63, 64, 65),
         128: (9, 10, 11, 12, 13, 57, 58, 59, 60, 61),
         256: (15, 16, 17, 18, 19, 20, 21, 49, 50, 51, 52, 53, 54, 55)}


def record_role(name):
    if name == 'block70.layer0.blend_scale':
        return 'original_output_blend_scalar', 'NOT_IN_FORMAL_COLOR_COMPOSITION'
    match = re.fullmatch(r'block(\d+)\.layer(\d+)\.layer', name)
    if not match:
        return 'unknown', 'UNRESOLVED'
    block, layer = map(int, match.groups())
    if layer == 0:
        for c, blocks in PLAIN.items():
            if block in blocks:
                return f'swin{c}', 'TENSOR_IMPLEMENTED_ROCM_ISOLATED_TESTS_ONLY'
        if block in (14, 22):
            return 'encoder_downsample', 'TENSOR_IMPLEMENTED_CAPTURED_CPU_BOUNDARY_ONLY'
        if block in (4, 8):
            return 'encoder_downsample', 'TENSOR_CANDIDATE_OUTPUT_LAYOUT_UNRESOLVED'
        if block == 70:
            return 'output_head32', 'TENSOR_IMPLEMENTED_ROCM_LEGACY_SDR_CONTROL_ONLY'
        if block == 39:
            return 'decoder_input_projection', 'TENSOR_IMPLEMENTED_CAPTURED_CPU_BOUNDARY_ONLY'
        if block == 0:
            return 'preprocess_noise_input_projection', 'TENSOR_IMPLEMENTED_SINGLE_COLOR_BRANCH_ONLY'
        if block in (48, 56, 62, 66):
            return 'decoder_upsample_swin_fusion', 'TENSOR_IMPLEMENTED_CAPTURED_CPU_BOUNDARY_ONLY'
    if block in list(range(23, 31)) + list(range(40, 48)) and layer in range(4):
        return f'split512_{layer}', 'TENSOR_IMPLEMENTED_CPU_REFERENCE_GPU_PENDING'
    if block in range(31, 39) and layer in range(5):
        return f'vit1024_{layer}', ('ABI_PLACEHOLDER_NOT_A_NEURAL_WEIGHT' if layer == 3
                                  else 'TENSOR_IMPLEMENTED_CPU_REFERENCE_GPU_PENDING')
    if block == 30 and layer == 4:
        return 'encoder_final_projection', 'TENSOR_IMPLEMENTED_CAPTURED_CPU_BOUNDARY_ONLY'
    return 'unknown', 'UNRESOLVED'


def evidence_records(report):
    """Bind observed execution to exact records, never promote an entire family."""
    probe = report.get('probe', {})
    if (report.get('pass') is not True or report.get('normal_exit') is not True
            or report.get('host_timeout') is not False or report.get('returncode') != 0
            or probe.get('checks_pass') is not True or probe.get('allocated_after_release_bytes') != 0
            or probe.get('backend') != 'pytorch_rocm' or probe.get('cpu_neural_fallback') is not False
            or probe.get('original_model_sha256') != ORIGINAL_SHA):
        raise ValueError('GPU evidence lacks original-weight ROCm execution and normal release/exit')
    if probe['name'] in ('whole_frame128','whole_frame640'):
        if ([s.get('block') for s in probe.get('stages',[])] != list(range(71))
                or probe.get('all_encoder_skips_generated') is not True
                or probe.get('rtx_intermediate_inputs') is not False
                or probe.get('single_color_tensor_chain_connected') is not True
                or probe.get('composition') != 'LEGACY_SDR_DIAGNOSTIC_ONLY'):
            raise ValueError('incomplete or teacher-fed single-color chain')
        mapping=probe.get('record_hashes',{})
        if len(mapping)!=152 or 'block70.layer0.blend_scale' in mapping:
            raise ValueError('single-color record scope must preserve inactive temporal blend distinction')
        result={}
        for name,sha in mapping.items():
            role,status=record_role(name)
            if role=='unknown':
                raise ValueError('unknown original chain record')
            placeholder=status=='ABI_PLACEHOLDER_NOT_A_NEURAL_WEIGHT'
            result[name]={'sha256':sha,'used_as_neural_weight':not placeholder,
                          'status':status if placeholder else 'TENSOR_IMPLEMENTED_ROCM_SINGLE_COLOR_CHAIN_ONLY'}
        return result
    if probe['name'] == 'decoder_pyramid':
        stages = probe.get('stages', [])
        if ([s.get('block') for s in stages] != list(range(48, 70))
                or probe.get('decoder_interior_rtx_substitution') is not False
                or probe.get('encoder_skip_source') != 'ORIGINAL_CAPTURED_EXTERNAL_INPUTS'
                or not probe.get('external_input_hashes')
                or any(s.get('comparison_diagnostic', {}).get('nonfinite') != 0 for s in stages)):
            raise ValueError('incomplete, nonfinite or teacher-injected decoder interior')
        return {f"block{s['block']}.layer0.layer": {
                    'sha256': s['original_record_sha256'],
                    'status': 'TENSOR_IMPLEMENTED_ROCM_CAPTURED_DECODER_SUBGRAPH_ONLY',
                    'used_as_neural_weight': True} for s in stages}
    if probe['name'] == 'vit_bottleneck':
        parts = ('expand', 'contract', 'qkv', 'attention_abi_placeholder', 'projection')
        mapping = probe['record_hashes_by_block']
        if set(mapping) != {str(b) for b in range(31, 39)} or probe.get('internal_rtx_substitution') is not False:
            raise ValueError('incomplete or teacher-injected bottleneck')
        status = 'TENSOR_IMPLEMENTED_ROCM_BOUNDED_SUBGRAPH_ONLY'
    elif probe['name'] == 'split512':
        parts = ('ffwd', 'ffwd_projection', 'attention', 'attention_projection')
        mapping = {'40': probe['record_hashes']}
        status = 'TENSOR_IMPLEMENTED_ROCM_SYNTHETIC_PROBE_ONLY'
    elif probe['name'] == 'downsample128':
        parts, mapping = ('boundary',), {'14': probe['record_hashes']}
        status = 'TENSOR_IMPLEMENTED_CPU_BOUNDARY_ROCM_SYNTHETIC_ONLY'
    else:
        raise ValueError('unsupported GPU evidence scope')
    result = {}
    for block, hashes in mapping.items():
        if set(hashes) != set(parts):
            raise ValueError('missing original record hashes')
        for layer, key in enumerate(parts):
            result[f'block{block}.layer{layer}.layer'] = {
                'sha256': hashes[key], 'status': 'ABI_PLACEHOLDER_NOT_A_NEURAL_WEIGHT' if key == 'attention_abi_placeholder' else status,
                'used_as_neural_weight': key != 'attention_abi_placeholder'}
    return result


def build(model_dir, output, gpu_evidence=()):
    arena = (model_dir/'model_arena.raw').read_bytes()
    if hashlib.sha256(arena).hexdigest().upper() != ORIGINAL_SHA:
        raise ValueError('original model hash mismatch')
    manifest = json.loads((model_dir/'manifest.json').read_bytes())
    records = []
    for rec in manifest['tensors']:
        role, status = record_role(rec['name'])
        first, size = rec['arena_offset'], rec['data_bytes']
        if first < 0 or size <= 0 or first + size > len(arena):
            raise ValueError('invalid original record bounds')
        records.append({'name': rec['name'], 'arena_offset': first, 'bytes': size, 'role': role, 'status': status,
                        'sha256': hashlib.sha256(arena[first:first+size]).hexdigest().upper()})
    by_name = {r['name']: r for r in records}
    for path in gpu_evidence:
        for name, evidence in evidence_records(json.loads(path.read_bytes())).items():
            if name not in by_name or evidence['sha256'] != by_name[name]['sha256']:
                raise ValueError('GPU evidence does not match original model record')
            by_name[name]['status'] = evidence['status']
            by_name[name]['gpu_evidence'] = {'manifest': str(path), 'used_as_neural_weight': evidence['used_as_neural_weight']}
    report = {'schema': 2, 'status': 'PARTIAL_NATIVE_TENSOR_RECONSTRUCTION',
              'original_model_sha256': ORIGINAL_SHA, 'record_count': len(records), 'records': records,
              'status_counts': dict(Counter(r['status'] for r in records)),
              'native_graph_complete': False, 'formal_cpu_neural_fallback_allowed': False,
              'training_started': False, 'quality_mismatch_blocks_reconstruction': False,
              'remaining_graph_contracts': [
                  'original optional history/motion/depth/mask input branches and reset/state semantics',
                  'formal HDR/SDR input color/exposure and output composition; active temporal blend scalar',
                  'same-input complete-image structural/sequence validation; not just finite values',
                  'larger/nonaligned dimensions GPU validation and reviewed resource budgets',
                  'resident GPU interop/pre-HUD visible output and device-loss lifecycle',
                  'whole-frame performance optimization and actual 4K60/SDR-HDR stability acceptance'],
              'runtime_safety_hold': 'NOT_EVALUATED_BY_RECORD_COVERAGE',
              'bounded_gpu_evidence': [str(p) for p in gpu_evidence],
              'runtime_safety_note': 'Record coverage neither clears nor imposes a current runtime hold. See docs/ROCM_TRACEBACK_REVIEW.md and separate repetition assessments. Historical access violation/hang remain unresolved; supplied probes do not establish game stability.',
              'next_quality_phase': 'After full graph and structural/color checks: frozen-weight images, then optional bounded fine-tuning',
              'warning': 'Record coverage is not numerical correctness, full graph completion, arbitrary-resolution support, or GPU verification.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'record_count': len(records), 'status_counts': report['status_counts'], 'native_graph_complete': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=Path('local_models/decoded_310_8'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-evidence', type=Path, nargs='*', default=[])
    args = p.parse_args()
    build(args.model_dir, args.output, args.gpu_evidence)
