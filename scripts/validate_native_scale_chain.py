"""CPU diagnostic: C128 downsample -> C256 input-view Swin -> plain Swin.

Real original weights/captured entry; no RTX tensor substituted inside chain.
Numerical differences remain diagnostic, not NVIDIA image-quality acceptance.
"""
import argparse
import json
from pathlib import Path
import struct
import torch
from native_multiscale import RecoveredDownsampleSwin, RecoveredOutviewSwin
from native_packed_swin import RecoveredPackedSwin, gather_packed, scatter_packed
from native_swin_torch import decode_e4, encode_e4, quantize_e4
from native_window_attention import LAYOUTS
from run_swin1h_native_validation import checked, compare, digest
from native_execution_policy import execution_policy, PROFILES


def run(output, precision):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    reports, prior_reference, source = [], None, None
    for slot, root, kind, size in [
        (15, Path('results/20260831_191000_swin4h_full_graph_exact_state/cases'), 'swin128', 229936),
        (16, Path('results/20260831_193100_swin8h_full_graph_exact_state/cases'), 'swin256', 689232),
        (17, Path('results/20260831_193100_swin8h_full_graph_exact_state/cases'), 'swin256', 689232)]:
        spec = next(s for s in json.loads((root/'manifest.json').read_bytes())['slots'] if s['slot'] == slot)
        def asset(name):
            record = spec['assets'][name]
            return checked(root/f'slot{slot}'/record['path'], record['sha256'])
        params, weights = asset('params'), asset('weights')[:size]
        h,w,ox,oy = struct.unpack_from('<4i', params, 32)
        channels = LAYOUTS[kind].channels
        captured = asset('input')[:w*h*channels]
        if prior_reference is not None and captured != prior_reference:
            raise ValueError('historical captures do not form contiguous same-frame boundaries')
        if source is None:
            source = decode_e4(captured)
        with torch.no_grad(), execution_policy(precision):
            if slot == 15:
                model = RecoveredDownsampleSwin(weights, record_kind=kind)
                values = model(source, w,h,ox,oy)
                source = values['captured_outview']
                reference = asset('extra_output_reference')[:spec['extra_logical_bytes']]
            elif slot == 16:
                model = RecoveredOutviewSwin(weights, record_kind=kind)
                source = model(source, w,h,ox,oy)
                reference = asset('output_reference')[:spec['main_logical_bytes']]
            else:
                model = RecoveredPackedSwin(weights, record_kind=kind)
                windows, mapping = gather_packed(source,w,h,channels,ox,oy)
                logical = torch.cat([model(part) for part in windows.split(12)])
                source = quantize_e4(scatter_packed(logical,mapping,w,h))
                reference = asset('output_reference')[:spec['main_logical_bytes']]
        actual = encode_e4(source).numpy().tobytes()
        if len(actual) != len(reference):
            raise ValueError('boundary size differs')
        comparison = compare(reference,actual)
        if comparison['nonfinite']:
            raise RuntimeError('nonfinite boundary')
        (output/f'slot{slot}.e4').write_bytes(actual)
        record = {'slot': slot, 'geometry':[w,h,ox,oy], 'comparison_diagnostic':comparison,
                  'captured_input_sha256':digest(captured), 'weights_sha256':digest(weights),
                  'reference_sha256':digest(reference), 'output_sha256':digest(actual),
                  'execution_input':'CAPTURED_ENTRY' if slot==15 else 'PRIOR_NATIVE_TENSOR',
                  'execution_input_sha256':digest(captured) if slot==15 else reports[-1]['output_sha256']}
        reports.append(record)
        print(json.dumps(record),flush=True)
        prior_reference = reference
    report = {'status':'CPU_CROSS_SCALE_CHAIN_EXECUTED', 'arithmetic_profile':precision,
              'backend':'cpu_reference_only', 'slots':reports, 'internal_rtx_substitution':False,
              'native_graph_complete':False, 'gpu_execution_verified':False, 'image_quality_verified':False,
              'training_started':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--precision',choices=PROFILES,default='recovered_k32')
    args=parser.parse_args()
    run(args.output,args.precision)
