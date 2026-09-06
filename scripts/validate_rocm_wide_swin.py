"""Original captured-input C64/128/256 Swin comparison; not full-network quality."""
import argparse
import faulthandler
import json
from pathlib import Path
import struct
import torch
from native_packed_swin import RecoveredPackedSwin, gather_packed, scatter_packed
from native_window_attention import LAYOUTS
from native_swin_torch import decode_e4, encode_e4
from run_swin1h_native_validation import checked, compare, digest
from validate_rocm_swin import wait


DECODER_SLOTS = [133, 134, 135, 136, 137, 138, 141, 142, 143, 144, 147, 148]


def validate_selection(slots, chain):
    if not slots or slots != sorted(set(slots)) or not set(slots) <= set([7, 8, 11, 12, 17, 18] + DECODER_SLOTS):
        raise ValueError('slots must be unique, supported and ascending')
    groups = [(7, 8), (11, 12), (17, 18), tuple(range(133, 139)), tuple(range(141, 145)), (147, 148)]
    if chain and (slots != list(range(slots[0], slots[-1] + 1)) or not any(set(slots) <= set(g) for g in groups)):
        raise ValueError('chain must be contiguous within one recovered family')


def load_cases(selected_slots):
    root = Path('results/20260831_170015_rtx5070_swin2h_slots6_9/_swin2h_slots6_9_reference_result_20260831_170015')
    payload = Path('deliverables/swin2h_slots6_9_reference_20260831_165256/payload')
    manifest = json.loads((root / 'manifest.json').read_bytes())
    for spec in manifest['slots']:
        slot = spec['slot']
        if slot not in (7, 8) or slot not in selected_slots:
            continue
        values = {key: checked(payload / f'slot{slot}_{key}.raw', spec[f'{key}_sha256']) for key in ('input', 'weights', 'params')}
        values['reference'] = checked(root / f'slot{slot}' / 'output.raw', spec['output_sha256'])
        yield slot, 'swin64', values
    for directory, kind, slots in [
        ('results/20260831_191000_swin4h_full_graph_exact_state/cases', 'swin128', (11, 12)),
        ('results/20260831_193100_swin8h_full_graph_exact_state/cases', 'swin256', (17, 18)),
    ]:
        root = Path(directory)
        manifest = json.loads((root / 'manifest.json').read_bytes())
        for spec in manifest['slots']:
            slot = spec['slot']
            if slot not in slots or slot not in selected_slots:
                continue
            values = {}
            for key, name in [('input', 'input'), ('weights', 'weights'), ('params', 'params'), ('reference', 'output_reference')]:
                asset = spec['assets'][name]
                values[key] = checked(root / f'slot{slot}' / asset['path'], asset['sha256'])
            yield slot, kind, values
    if not set(selected_slots).intersection(DECODER_SLOTS):
        return
    root = Path('results/20260831_230000_decoder_full_graph_exact_state/cases')
    manifest = json.loads((root / 'manifest.json').read_bytes())
    weights_arena = checked(Path('local_models/decoded_310_8/model_arena.raw'),
                            'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5')
    for spec in manifest['slots']:
        slot = spec['slot']
        if slot not in selected_slots or slot not in DECODER_SLOTS:
            continue
        kind = 'swin256' if slot < 140 else 'swin128' if slot < 146 else 'swin64'
        folder = root / f'slot{slot}'
        params = checked(folder / spec['params']['path'], spec['params']['sha256'])
        asset = spec['activation_arena']
        arena = checked(folder / asset['path'], asset['sha256'])
        height, width = struct.unpack_from('<2i', params, 32)
        size = width * height * LAYOUTS[kind].channels
        views = {view['param_offset']: view['arena_offset'] for view in spec['activation_param_views']}
        offset = next(view['weight_offset'] for view in spec['weight_param_views'] if view['param_offset'] == 16)
        result = next(item for item in spec['outputs'] if item['param_offset'] == 8)
        if result['logical_bytes'] != size or views[0] + size > len(arena):
            raise ValueError('decoder view geometry mismatch')
        yield slot, kind, {'input': arena[views[0]:views[0] + size], 'params': params,
                           'weights': weights_arena[offset:offset + LAYOUTS[kind].record_bytes],
                           'reference': checked(folder / result['asset']['path'], result['asset']['sha256'])}


def run(output, selected_slots, chain=False):
    validate_selection(selected_slots, chain)
    if not torch.version.hip or not torch.cuda.is_available() or 'gfx1201' not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError('gfx1201 ROCm required; no CPU neural fallback')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    previous_gpu = None
    previous_reference = None
    previous_geometry = None
    torch.cuda.reset_peak_memory_stats()
    for slot, kind, assets in load_cases(selected_slots):
        if slot not in selected_slots:
            continue
        layout = LAYOUTS[kind]
        height, width, ox, oy = struct.unpack_from('<4i', assets['params'], 32)
        logical_bytes = width * height * layout.channels
        if logical_bytes > len(assets['input']) or logical_bytes > len(assets['reference']):
            raise ValueError('capture does not cover logical feature region')
        weights = assets['weights'][:layout.record_bytes]
        model = RecoveredPackedSwin(weights, record_kind=kind).cuda()
        captured_input = assets['input'][:logical_bytes]
        if chain and previous_gpu is not None:
            if (slot != records[-1]['slot'] + 1 or previous_geometry != (width, height, layout.channels)
                    or captured_input != previous_reference):
                raise ValueError('chain requires verified contiguous same-geometry tensor boundaries')
            source = previous_gpu
        else:
            source = decode_e4(captured_input).cuda()
        windows, mapping = gather_packed(source, width, height, layout.channels, ox, oy)
        event, start = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            model(windows[:1]); event.record(); wait(event)
            chunks = []
            start.record()
            for batch in windows.split(12):
                value = model(batch)
                finite = torch.isfinite(value).all()
                event.record(); wait(event)
                if not bool(finite):
                    raise RuntimeError(f'nonfinite slot{slot}; stop without retry')
                chunks.append(value)
            packed = encode_e4(scatter_packed(torch.cat(chunks), mapping, width, height))
            event.record(); wait(event)
            elapsed = start.elapsed_time(event)
        actual = packed.cpu().numpy().tobytes()
        (output / f'slot{slot}.e4').write_bytes(actual)
        comparison = compare(assets['reference'][:logical_bytes], actual)
        record = {'slot': slot, 'kind': kind, 'geometry': [width, height, ox, oy],
                  'logical_bytes': logical_bytes, 'comparison': comparison,
                  'input_sha256': digest(assets['input']), 'weights_capture_sha256': digest(assets['weights']),
                  'weights_record_sha256': digest(weights), 'params_sha256': digest(assets['params']),
                  'reference_sha256': digest(assets['reference']), 'output_sha256': digest(actual),
                  'execution_input': 'PRIOR_NATIVE_GPU_OUTPUT' if chain and records else 'CAPTURED_ISOLATED_INPUT',
                  'execution_input_sha256': records[-1]['output_sha256'] if chain and records else digest(captured_input),
                  'gpu_span_ms': elapsed, 'timing_scope': '12-window batches with host waits, not optimized throughput'}
        records.append(record)
        print(json.dumps(record), flush=True)
        if chain and slot != selected_slots[-1]:
            previous_gpu = packed.view(torch.float8_e4m3fn).half()
            event.record(); wait(event)
            previous_reference = assets['reference'][:logical_bytes]
            previous_geometry = (width, height, layout.channels)
        del model, source, windows, chunks, packed, value
    passed = len(records) == len(selected_slots) and all(r['comparison']['pass'] for r in records)
    report = {'status': ('NATIVE_SUBGRAPH_PASS' if chain else 'ISOLATED_WIDE_SWIN_PASS') if passed else 'NUMERICAL_GATE_FAIL',
              'backend': 'pytorch_rocm', 'slots': records,
              'execution_mode': 'GPU_RESIDENT_CONTIGUOUS_SUBGRAPH' if chain else 'ISOLATED_CAPTURED_INPUTS',
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'native_graph_complete': False, 'game_runtime_ready': False,
              'teacher_intermediates_used_as_isolated_inputs_only': True,
              'note': 'Local operator tolerance is not SDR/HDR image quality acceptance.'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return passed


if __name__ == '__main__':
    # Diagnostic only: a Python timeout cannot cancel submitted GPU work.
    # Leave enabled through teardown to capture a future stuck interpreter.
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chain', action='store_true', help='feed preceding native GPU output, forbid disconnected slots')
    p.add_argument('--slots', type=int, nargs='+', choices=[7, 8, 11, 12, 17, 18] + DECODER_SLOTS,
                   default=[7, 8, 11, 12, 17, 18] + DECODER_SLOTS)
    args = p.parse_args()
    raise SystemExit(0 if run(args.output, args.slots, args.chain) else 1)
