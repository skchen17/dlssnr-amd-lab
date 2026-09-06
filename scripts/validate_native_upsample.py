"""CPU captured-input checks for four original decoder upsample/skip families."""
import argparse
import json
from pathlib import Path
import struct
import torch
from native_upsample_swin import RecoveredUpsampleSwin, RECORD_SIZES
from native_execution_policy import execution_policy, PROFILES
from native_swin_torch import decode_e4, encode_e4
from run_swin1h_native_validation import checked, compare, digest
from validate_rocm_lifecycle import ORIGINAL_SHA


def load_cases(slots):
    model_root=Path('local_models/decoded_310_8')
    arena=checked(model_root/'model_arena.raw',ORIGINAL_SHA)
    table={t['name']:t for t in json.loads((model_root/'manifest.json').read_bytes())['tensors']}
    root=Path('results/20260831_230000_decoder_full_graph_exact_state/cases')
    specs={s['slot']:s for s in json.loads((root/'manifest.json').read_bytes())['slots']}
    for slot,block,kind in [(132,48,'swin256'),(140,56,'swin128'),(146,62,'swin64'),(150,66,'swin32')]:
        if slot not in slots:
            continue
        spec=specs[slot]
        folder=root/f'slot{slot}'
        params=checked(folder/spec['params']['path'],spec['params']['sha256'])
        h,w,ox,oy=struct.unpack_from('<4i',params,24 if slot==150 else 32)
        rec=table[f'block{block}.layer0.layer']
        if rec['arena_offset']!=spec['weight_param_views'][0]['weight_offset'] or rec['data_bytes']!=RECORD_SIZES[kind]:
            raise ValueError('original record/captured slot binding differs')
        raw=arena[rec['arena_offset']:rec['arena_offset']+rec['data_bytes']]
        capture=checked(folder/spec['activation_arena']['path'],spec['activation_arena']['sha256'])
        views={v['param_offset']:v['arena_offset'] for v in spec['activation_param_views']}
        c={'swin32':32,'swin64':64,'swin128':128,'swin256':256}[kind]
        low=capture[views[0]:views[0]+w*h*c//2]
        skip_offset=views[80 if slot==150 else 24]
        skip=capture[skip_offset:skip_offset+w*h*c]
        out=spec['outputs'][0]
        expected=checked(folder/out['asset']['path'],out['asset']['sha256'])
        if len(low)!=w*h*c//2 or len(skip)!=w*h*c or len(expected)!=w*h*c:
            raise ValueError('captured geometry/actual skip size differs')
        yield slot,kind,(w,h,ox,oy),raw,low,skip,expected


def run(output,slots,precision):
    if not slots or set(slots)-{132,140,146,150} or len(set(slots))!=len(slots):
        raise ValueError('expected unique supported upsample slots')
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    records=[]
    for slot,kind,geometry,weights,low,skip,expected in load_cases(slots):
        model=RecoveredUpsampleSwin(weights,record_kind=kind)
        with torch.no_grad(),execution_policy(precision):
            result=model(decode_e4(low),decode_e4(skip),*geometry)
        actual=encode_e4(result).numpy().tobytes()
        (output/f'slot{slot}.e4').write_bytes(actual)
        comparison=compare(expected,actual)
        if comparison['nonfinite']:
            raise RuntimeError('nonfinite candidate output')
        record={'slot':slot,'kind':kind,'geometry':geometry,'weights_sha256':digest(weights),
                'low_input_sha256':digest(low),'actual_encoder_skip_sha256':digest(skip),
                'output_sha256':digest(actual),'reference_sha256':digest(expected),
                'comparison_diagnostic':comparison,'interior_rtx_substitution':False}
        records.append(record)
        print(json.dumps(record),flush=True)
    report={'status':'ORIGINAL_UPSAMPLE_CPU_CANDIDATES_EXECUTED','backend':'cpu_reference_only',
            'arithmetic_profile':precision,'original_model_sha256':ORIGINAL_SHA,'slots':records,
            'native_graph_complete':False,'gpu_execution_verified':False,'image_quality_verified':False,
            'training_started':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--slots',type=int,nargs='+',default=[132,140,146,150])
    p.add_argument('--precision',choices=PROFILES,default='recovered_k32')
    args=p.parse_args()
    run(args.output,args.slots,args.precision)
