"""Bounded same-input RTX comparison of the native torch 1h/32 prototype."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from native_swin_torch import RecoveredSwin32, decode_e4, encode_e4, gather_windows, scatter_windows
from run_swin1h_native_validation import checked, compare, digest


def wait(event):
    deadline = time.monotonic() + 10
    while not event.query():
        if time.monotonic() > deadline:
            raise TimeoutError('GPU timeout: no automatic retry')
        time.sleep(.002)


def run(output, cases, slots):
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm required; no CPU fallback')
    prop = torch.cuda.get_device_properties(0)
    if 'gfx1201' not in prop.gcnArchName:
        raise RuntimeError('expected gfx1201')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((cases / 'manifest.json').read_text())
    records = []
    for spec in manifest['slots']:
        slot = spec['slot']
        if slot not in slots:
            continue
        folder = cases / f'slot{slot}'
        raw = checked(folder / 'input.e4', spec['input_sha256'])
        weights = checked(folder / 'weights.raw', spec['weights_sha256'])
        reference = checked(Path(spec['reference']), spec['reference_sha256'])
        w, h, ox, oy = spec['geometry']
        model = RecoveredSwin32(weights).cuda()
        inputs = decode_e4(raw).cuda()
        windows, mapping = gather_windows(inputs, w, h, ox, oy)
        end = torch.cuda.Event(enable_timing=True)
        start = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            model(windows[:1])
            end.record(); wait(end)
            start.record()
            chunks = []
            for batch in windows.split(64):
                chunks.append(model(batch))
                end.record(); wait(end)
            values = torch.cat(chunks)
            packed = encode_e4(scatter_windows(values, mapping, w, h))
            end.record(); wait(end)
            elapsed = start.elapsed_time(end)
        actual = packed.cpu().numpy().tobytes()
        (output / f'slot{slot}.e4').write_bytes(actual)
        rec = {'slot': slot, 'geometry': spec['geometry'], 'gpu_event_span_ms': elapsed,
               'timing_scope': 'warm model; bounded batches, host gaps included; input/weights upload excluded',
               'comparison': compare(reference, actual), 'output_sha256': digest(actual),
               'input_sha256': digest(raw), 'weights_sha256': digest(weights),
               'reference_sha256': digest(reference)}
        records.append(rec)
        print(json.dumps(rec), flush=True)
    passed = len(records) == len(slots) and all(r['comparison']['pass'] for r in records)
    report = {'schema': 1, 'status': 'ISOLATED_FAMILY_PASS' if passed else 'NUMERICAL_GATE_FAIL',
              'backend': 'pytorch_rocm', 'device': prop.name, 'slots': records,
              'native_graph_complete': False, 'game_runtime_ready': False,
              'captured_inputs_only_for_isolated_test': True,
              'note': 'The source DXIL prototype fails integrated image quality; this is not promotion.'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', type=Path, default=Path('results/20260905_052500_swin1h_native_family'))
    parser.add_argument('--slots', type=int, nargs='+', default=[3, 4, 151, 152])
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.cases, args.slots) else 1)
