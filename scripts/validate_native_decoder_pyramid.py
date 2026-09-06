"""Captured-entry CPU check of22 consecutive original decoder tensor blocks.

Four captured encoder skips are EXTERNAL inputs, not a rebuilt encoder claim.
No teacher output replaces any calculated decoder stage.
"""
import argparse
import json
from pathlib import Path
import struct
import torch
from native_decoder_pyramid import RecoveredDecoderPyramid,stage_role
from native_swin_torch import decode_e4,encode_e4
from native_execution_policy import execution_policy,PROFILES
from validate_rocm_lifecycle import ORIGINAL_SHA
from run_swin1h_native_validation import checked,compare,digest


def load_fixture():
    torch.set_num_threads(2)
    root=Path('results/20260831_230000_decoder_full_graph_exact_state/cases')
    specs={s['slot']:s for s in json.loads((root/'manifest.json').read_bytes())['slots']}
    model_root=Path('local_models/decoded_310_8')
    weights=checked(model_root/'model_arena.raw',ORIGINAL_SHA)
    table={t['name']:t for t in json.loads((model_root/'manifest.json').read_bytes())['tensors']}
    records,origins,references,skips,input_hashes={}, {}, {}, {}, {}
    previous=None
    for block in range(48,70):
        slot=block+84
        spec=specs[slot]; folder=root/f'slot{slot}'
        c,role=stage_role(block)
        params=checked(folder/spec['params']['path'],spec['params']['sha256'])
        h,w,ox,oy=struct.unpack_from('<4i',params,24 if c==32 else 32)
        if (w,h)!=(320//(c//32),192//(c//32)):
            raise ValueError('capture geometry is not this bounded decoder fixture')
        rec=table[f'block{block}.layer0.layer']
        if rec['arena_offset']!=spec['weight_param_views'][0]['weight_offset']:
            raise ValueError('original record/captured layer mismatch')
        records[block]=weights[rec['arena_offset']:rec['arena_offset']+rec['data_bytes']]
        origins[block]=(ox,oy)
        arena=checked(folder/spec['activation_arena']['path'],spec['activation_arena']['sha256'])
        views={v['param_offset']:v['arena_offset'] for v in spec['activation_param_views']}
        size=w*h*c//2 if role=='upsample' else w*h*c
        captured=arena[views[0]:views[0]+size]
        if previous is not None and previous!=captured:
            raise ValueError(f'noncontiguous decoder captured boundary at slot{slot}')
        if block==48:
            entry=decode_e4(captured)
            input_hashes['entry']=digest(captured)
        if role=='upsample':
            offset=views[80 if c==32 else 24]
            raw_skip=arena[offset:offset+w*h*c]
            skips[c]=decode_e4(raw_skip)
            input_hashes[f'encoder_skip{c}']=digest(raw_skip)
        out=spec['outputs'][0]['asset']
        references[block]=checked(folder/out['path'],out['sha256'])
        previous=references[block]
    model=RecoveredDecoderPyramid(records,origins)
    return model,entry,skips,references,records,input_hashes


def run(output,precision):
    output.mkdir(parents=True,exist_ok=False)
    model,entry,skips,references,records,input_hashes=load_fixture()
    results=[]
    with torch.no_grad(),execution_policy(precision):
        for block,value in model.stages(entry,skips,320,192):
            actual=encode_e4(value).numpy().tobytes()
            comparison=compare(references[block],actual)
            if comparison['nonfinite']:
                raise RuntimeError('nonfinite decoder stage')
            (output/f'slot{block+84}.e4').write_bytes(actual)
            item={'block':block,'slot':block+84,'comparison_diagnostic':comparison,
                  'output_sha256':digest(actual),'record_sha256':digest(records[block])}
            results.append(item)
            print(json.dumps(item),flush=True)
    report={'status':'CPU_DECODER_PYRAMID22_EXECUTED','backend':'cpu_reference_only',
            'arithmetic_profile':precision,'original_model_sha256':ORIGINAL_SHA,
            'external_input_hashes':input_hashes,'encoder_skip_source':'ORIGINAL_CAPTURED_EXTERNAL_INPUTS',
            'decoder_interior_rtx_substitution':False,'stages':results,
            'native_graph_complete':False,'image_quality_verified':False,'gpu_execution_verified':False,
            'training_started':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precision',choices=PROFILES,default='native_fp16')
    args=p.parse_args()
    run(args.output,args.precision)
