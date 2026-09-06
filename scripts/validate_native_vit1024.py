"""CPU-only original ViT reconstruction checks; numerical error is diagnostic."""
import argparse
import json
from pathlib import Path
import struct
import time
import torch
from native_vit1024 import RecoveredVit1024, RecoveredVitBottleneck1024, RECORD_SIZES
from native_sequence_layout import unpack_sequence, pack_sequence
from native_swin_torch import decode_e4, encode_e4
from run_swin1h_native_validation import checked, compare, digest


ORIGINAL_SHA = 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5'


def run(output, blocks, chain=False):
    if not blocks or len(set(blocks)) != len(blocks) or not set(blocks) <= set(range(31, 39)):
        raise ValueError('expected unique ViT blocks31..38')
    if chain and blocks != list(range(31, 39)):
        raise ValueError('bottleneck chain requires ordered blocks31..38')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    root = Path('local_models/decoded_310_8')
    arena = checked(root / 'model_arena.raw', ORIGINAL_SHA)
    tensors = {r['name']: r for r in json.loads((root / 'manifest.json').read_bytes())['tensors']}
    def block_model(block):
        parts = {}
        for i, (key, size) in enumerate(RECORD_SIZES.items()):
            rec = tensors[f'block{block}.layer{i}.layer']
            if rec['data_bytes'] != size:
                raise ValueError('original ViT record length mismatch')
            parts[key] = arena[rec['arena_offset']:rec['arena_offset'] + size]
        return RecoveredVit1024(**parts)
    decoded = []
    for block in range(31, 39):
        model = block_model(block)
        decoded.append({'block': block, 'records': model.record_sha256, 'finite_parameters': True})
    root = Path('results/20260831_221000_vit1d_full_graph_exact_state/cases')
    specs = {s['slot']: s for s in json.loads((root / 'manifest.json').read_bytes())['slots']}
    results = []
    chain_model = RecoveredVitBottleneck1024({b: block_model(b) for b in blocks}) if chain else None
    previous_reference = None
    first_x = None
    for block in blocks:
        slot = 58 + (block-31)*5
        spec, last = specs[slot], specs[slot+4]
        params = checked(root / f'slot{slot}' / spec['params']['path'], spec['params']['sha256'])
        height, width = struct.unpack_from('<2i', params, 64)
        tokens = height * width
        data = checked(root / f'slot{slot}' / spec['activation_arena']['path'], spec['activation_arena']['sha256'])
        offset = next(v['arena_offset'] for v in spec['activation_param_views'] if v['param_offset'] == 0)
        raw = data[offset:offset + ((tokens+31)//32*32)*1024]
        out = next(v for v in last['outputs'] if v['param_offset'] == 16)
        reference = checked(root / f'slot{slot+4}' / out['asset']['path'], out['asset']['sha256'])
        for i in range(5):
            if specs[slot+i]['weight_view_offset'] != tensors[f'block{block}.layer{i}.layer']['arena_offset']:
                raise ValueError('original weight binding does not match captured stage')
        if chain:
            if previous_reference is not None and raw != previous_reference:
                raise ValueError('captured ViT boundaries do not form one contiguous chain')
            previous_reference = reference
            if first_x is None:
                first_x = unpack_sequence(decode_e4(raw), tokens, 1024)
            if block != 38:
                continue
        model = block_model(block)
        start = time.monotonic()
        with torch.no_grad():
            x = unpack_sequence(decode_e4(raw), tokens, 1024)
            result = chain_model(first_x) if chain else model(x)
            actual = encode_e4(pack_sequence(result)).numpy().tobytes()
        (output / f'block{block}.e4').write_bytes(actual)
        record = {'block': block, 'slots': list(range(58, 98)) if chain else list(range(slot, slot+5)), 'tokens': tokens,
                  'input_sha256': digest(raw), 'output_sha256': digest(actual),
                  'reference_comparison_diagnostic_only': compare(reference, actual),
                  'cpu_reference_seconds': time.monotonic()-start, 'internal_rtx_substitution': False}
        if chain:
            record['input_sha256'] = digest(encode_e4(pack_sequence(first_x)).numpy().tobytes())
            record['execution_mode'] = 'EIGHT_ORIGINAL_VIT_BLOCKS_CONTIGUOUS_TENSORS'
        results.append(record)
        print(json.dumps(record), flush=True)
    report = {'status': 'VIT_CPU_TENSOR_BASELINE_GPU_NOT_RETESTED', 'backend': 'cpu_reference_only',
              'original_model_sha256': ORIGINAL_SHA, 'decoded_blocks': decoded, 'results': results,
              'native_graph_complete': False, 'image_quality_verified': False, 'gpu_execution_verified': False,
              'training_started': False, 'stability_hold': 'ROCM_PROCESS_TEARDOWN_UNRESOLVED',
              'ffn_output_storage': 'e4m3_not_legacy_fp16_label',
              'attention': 'global_chunked_bounded_exp_then_unnormalized_fp8_PV_then_normalize',
              'note': 'Mismatch remains diagnostic; continue restoring the full graph before quality tuning.'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return all(r['reference_comparison_diagnostic_only'].get('nonfinite', 0) == 0 for r in results)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--blocks', type=int, nargs='+', default=[31, 32])
    p.add_argument('--chain', action='store_true')
    args = p.parse_args()
    raise SystemExit(0 if run(args.output, args.blocks, args.chain) else 1)
