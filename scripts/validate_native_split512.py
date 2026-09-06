"""CPU reference only while ROCm process-teardown safety hold is unresolved.

Runs actual original-weight four-stage split-Swin tensors. RTX images are NOT
substituted for internal stages. Feature error is diagnostic, not a build gate.
This is explicitly not a CPU deployment fallback or proof of ROCm execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import torch
from native_split_swin512 import RecoveredSplitSwin512, RECORD_SIZES
from native_packed_swin import logical_indices, gather_packed, scatter_packed
from native_swin_torch import decode_e4, encode_e4
from run_swin1h_native_validation import checked, compare


ORIGINAL_SHA = 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5'


def run(output, blocks):
    if not blocks or len(set(blocks)) != len(blocks) or not set(blocks) <= set(range(40, 48)):
        raise ValueError('expected unique decoder blocks40..47')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    root = Path('local_models/decoded_310_8')
    arena = checked(root / 'model_arena.raw', ORIGINAL_SHA)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    tensors = {r['name']: r for r in manifest['tensors']}
    def original_block(block):
        parts = {}
        for i, (name, size) in enumerate(RECORD_SIZES.items()):
            rec = tensors[f'block{block}.layer{i}.layer']
            if rec['data_bytes'] != size:
                raise ValueError('named split block size mismatch')
            parts[name] = arena[rec['arena_offset']:rec['arena_offset'] + size]
        return RecoveredSplitSwin512(**parts)
    decoded = []
    for block in list(range(23, 31)) + list(range(40, 48)):
        model = original_block(block)
        decoded.append({'block': block, 'records': model.record_sha256,
                        'finite_parameters': True, 'frozen': not any(p.requires_grad for p in model.parameters())})
    cases = Path('results/20260831_230000_decoder_full_graph_exact_state/cases')
    records = {s['slot']: s for s in json.loads((cases / 'manifest.json').read_bytes())['slots']}
    a_index, residual_index = logical_indices(512)
    results = []
    for block in blocks:
        first = 100 + (block - 40) * 4
        spec, attn, last = records[first], records[first + 2], records[first + 3]
        params = checked(cases / f'slot{first+2}' / attn['params']['path'], attn['params']['sha256'])
        height, width, ox, oy = struct.unpack_from('<4i', params, 24)
        capture = checked(cases / f'slot{first}' / spec['activation_arena']['path'], spec['activation_arena']['sha256'])
        offset = next(v['arena_offset'] for v in spec['activation_param_views'] if v['param_offset'] == 0)
        raw = capture[offset:offset + width * height * 512]
        expected_spec = next(v for v in last['outputs'] if v['param_offset'] == 16)
        expected = checked(cases / f'slot{first+3}' / expected_spec['asset']['path'], expected_spec['asset']['sha256'])
        model = original_block(block)
        # Bind every stage to its recorded original record, not names/sizes alone.
        for i in range(4):
            actual_view = records[first+i]['weight_param_views'][0]['weight_offset']
            if actual_view != tensors[f'block{block}.layer{i}.layer']['arena_offset']:
                raise ValueError('record/captured-stage weight binding differs')
        windows, mapping = gather_packed(decode_e4(raw), width, height, 512, ox, oy)
        start = time.monotonic()
        with torch.no_grad():
            values = torch.cat([model(batch[:, a_index], batch[:, residual_index]) for batch in windows.split(1)])
            packed = encode_e4(scatter_packed(values, mapping, width, height))
        actual = packed.numpy().tobytes()
        comparison = compare(expected, actual)
        (output / f'block{block}.e4').write_bytes(actual)
        result = {'block': block, 'slots': list(range(first, first+4)), 'geometry': [width, height, ox, oy],
                  'reference_comparison_diagnostic_only': comparison,
                  'cpu_reference_seconds': time.monotonic()-start,
                  'input_sha256': hashlib.sha256(raw).hexdigest().upper(),
                  'output_sha256': hashlib.sha256(actual).hexdigest().upper(),
                  'internal_rtx_substitution': False}
        results.append(result)
        print(json.dumps(result), flush=True)
    # Trainable tensor graph probe, no optimizer, no derived checkpoint.
    model = original_block(40)
    torch.manual_seed(1201)
    x = (torch.randn(1, 64, 512)*.01).half().requires_grad_()
    model.attention_projection.weight.requires_grad_(True)
    model(x, x).float().square().mean().backward()
    grads = [x.grad, model.attention_projection.weight.grad]
    grad_ok = all(g is not None and bool(torch.isfinite(g).all()) and bool(torch.count_nonzero(g)) for g in grads)
    report = {'status': 'CPU_REFERENCE_IMPLEMENTED_GPU_NOT_RETESTED',
              'backend': 'cpu_reference_only', 'original_model_sha256': ORIGINAL_SHA,
              'decoded_blocks': decoded, 'four_stage_results': results,
              'cpu_backward_finite_nonzero': grad_ok, 'training_started': False,
              'gpu_execution_verified': False, 'native_graph_complete': False,
              'image_quality_verified': False, 'stability_hold': 'ROCM_PROCESS_TEARDOWN_UNRESOLVED',
              'note': 'Numerical mismatch does not block remaining model reconstruction.'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return grad_ok and all(r['reference_comparison_diagnostic_only'].get('nonfinite', 0) == 0 for r in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--blocks', type=int, nargs='+', default=[40, 41])
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.blocks) else 1)
