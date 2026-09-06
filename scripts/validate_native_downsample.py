"""CPU reference for learned downsampling; both skip and next-scale views audited."""
import argparse
import json
from pathlib import Path
import struct
import torch
from native_multiscale import RecoveredDownsampleSwin, DOWNSAMPLE_RECORDS
from native_swin_torch import decode_e4, encode_e4
from run_swin1h_native_validation import checked, compare, digest


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    records = []
    for case_dir, slot, kind in [
        ('results/20260831_191000_swin4h_full_graph_exact_state/cases', 15, 'swin128'),
        ('results/20260831_193100_swin8h_full_graph_exact_state/cases', 23, 'swin256'),
    ]:
        root = Path(case_dir)
        spec = next(s for s in json.loads((root/'manifest.json').read_bytes())['slots'] if s['slot'] == slot)
        def asset(name):
            value = spec['assets'][name]
            return checked(root / f'slot{slot}' / value['path'], value['sha256'])
        height, width, ox, oy = struct.unpack_from('<4i', asset('params'), 32)
        weight_capture = asset('weights')
        weight = weight_capture[:DOWNSAMPLE_RECORDS[kind]]
        model = RecoveredDownsampleSwin(weight, record_kind=kind)
        raw = asset('input')[:width*height*model.channels]
        with torch.no_grad():
            values = model(decode_e4(raw), width, height, ox, oy)
        checks = {}
        for key, name, size in [('skip', 'output_reference', spec['main_logical_bytes']),
                                ('captured_outview', 'extra_output_reference', spec['extra_logical_bytes'])]:
            actual = encode_e4(values[key]).numpy().tobytes()
            if len(actual) != size:
                raise ValueError('native boundary size differs from captured contract')
            (output / f'slot{slot}_{key}.e4').write_bytes(actual)
            checks[key] = compare(asset(name)[:size], actual)
        record = {'slot': slot, 'kind': kind, 'geometry': [width, height, ox, oy],
                  'weights_record_sha256': digest(weight), 'input_sha256': digest(raw),
                  'reference_comparison_diagnostic_only': checks}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'status': 'DOWNSAMPLE_CPU_CANDIDATE', 'backend': 'cpu_reference_only', 'results': records,
              'gpu_execution_verified': False, 'native_graph_complete': False, 'training_started': False,
              'note': 'Stored skip/outview adapters may differ; mismatch does not halt full-graph reconstruction.'}
    (output/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    run(args.output)
