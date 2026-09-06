"""Offline original decoder input and exact 2D/sequence layout validation."""
import argparse
import json
from pathlib import Path
import torch
from native_capture_fixture import CaptureCases, OriginalRecords
from native_transition_projections import DecoderInputProjection, EncoderFinalProjection
from native_multiscale import unpack_image, pack_image
from native_sequence_layout import unpack_sequence, pack_sequence
from native_swin_torch import decode_e4, encode_e4
from native_execution_policy import execution_policy
from run_swin1h_native_validation import compare, digest


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    original = OriginalRecords()
    vit = CaptureCases('results/20260831_221000_vit1d_full_graph_exact_state/cases')
    decoder = CaptureCases('results/20260831_230000_decoder_full_graph_exact_state/cases')
    reports = []
    with torch.no_grad(), execution_policy('native_fp16'):
        for slot, inverse in [(57, False), (98, True)]:
            spec = vit.specs[slot]
            h, w = vit.integers(slot, 16, 2)
            # ViT case schema records the arena input and output separately.
            raw = vit.input(slot, 0, ((w*h+31)//32*32 if inverse else w*h)*1024)
            if inverse:
                logical = unpack_sequence(decode_e4(raw), w*h, 1024)[0].reshape(h,w,1024)
                actual = encode_e4(pack_image(logical)).numpy().tobytes()
            else:
                logical = unpack_image(decode_e4(raw), w,h,1024).reshape(1,w*h,1024)
                actual = encode_e4(pack_sequence(logical)).numpy().tobytes()
            reference = vit.output(slot, 8)
            reports.append({'slot':slot,'geometry':[w,h], 'input_sha256':digest(raw),
                            'comparison':compare(reference,actual),'output_sha256':digest(actual)})
        h,w,_,_ = decoder.integers(99,64,4)
        # These parameters are the padded LOW input dimensions; output dimensions
        # are supplied independently by the following genuine encoder-resolution stage.
        dh,dw,_,_ = decoder.integers(102,24,4)
        raw = decoder.input(99,0,w*h*1024)
        skip = decoder.input(99,8,dw*dh*512)
        original.bind(39,0,decoder.specs[99]['weight_param_views'][0]['weight_offset'])
        model = DecoderInputProjection(original.get(39))
        fused = model.fuse(unpack_image(decode_e4(raw),w,h,1024),
                           unpack_image(decode_e4(skip),dw,dh,512),dw,dh)
        actual = encode_e4(pack_image(fused)).numpy().tobytes()
        reports.append({'slot':99,'geometry':[dw,dh],'low_geometry':[w,h],
                        'input_sha256':digest(raw),'skip_sha256':digest(skip),
                        'comparison':compare(decoder.output(99,16),actual),'output_sha256':digest(actual)})
    report = {'backend':'cpu_reference_only','native_graph_complete':False,'training_started':False,
              'results':reports}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
