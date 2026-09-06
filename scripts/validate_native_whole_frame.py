"""Offline whole single-color candidate; all features/skips freshly calculated.

This is not temporal/HDR/DLSS quality or game acceptance. CPU is diagnostic only.
Synthetic input is explicitly labeled and never presented as a teacher sample.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from native_capture_fixture import whole_frame_fixture
from native_whole_frame import SingleColorWholeFrame
from native_execution_policy import execution_policy
from native_swin_torch import encode_e4
from run_swin1h_native_validation import digest
from describe_native_reconstruction import ORIGINAL_SHA


def run(output,width,height,input_path=None):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    records,origins,settings=whole_frame_fixture()
    model=SingleColorWholeFrame(records,origins,**settings)
    if input_path:
        raw=input_path.read_bytes()
        if len(raw)!=width*height*8:
            raise ValueError('actual RGBA16F size mismatch')
        color=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(height,width,4)
        source='ACTUAL_SUPPLIED_RGBA16F'
    else:
        y,x=torch.meshgrid(torch.linspace(0,1,height),torch.linspace(0,1,width),indexing='ij')
        color=torch.stack([x,y,(x+y)*.5,torch.ones_like(x)],-1).half()
        raw=color.numpy().tobytes(); source='SYNTHETIC_GRADIENT_NOT_TEACHER_DATA'
    if not bool(torch.isfinite(color).all()):
        raise ValueError('nonfinite actual pixels')
    stages=[]
    with torch.no_grad(),execution_policy('native_fp16'):
        iterator=iter(model.stages(color,0))
        for index in range(71):
            start=time.perf_counter()
            b,value=next(iterator)
            if b!=index or not bool(torch.isfinite(value).all()):
                raise RuntimeError('nonfinite or missing native stage')
            raw_output=value.numpy().tobytes()
            item={'block':b,'shape':list(value.shape),'seconds':time.perf_counter()-start,
                  'fp16_sha256':digest(raw_output),'min':float(value.min()),'max':float(value.max())}
            stages.append(item); print(json.dumps(item),flush=True)
        try:
            next(iterator)
            raise RuntimeError('unexpected extra stage')
        except StopIteration:
            pass
    (output/'output.rgba16f').write_bytes(raw_output)
    (output/'input.rgba16f').write_bytes(raw)
    report={'status':'SINGLE_COLOR_WHOLE_FRAME_CPU_CANDIDATE_EXECUTED','backend':'cpu_reference_only',
            'original_model_sha256':ORIGINAL_SHA,'arithmetic_profile':'native_fp16',
            'input_source':source,'input_sha256':digest(raw),'input_geometry':[width,height],
            'stages':stages,'all_encoder_skips_generated':True,'rtx_intermediate_inputs':False,
            'native_graph_complete':False,'single_color_tensor_chain_connected':True,
            'composition':'LEGACY_SDR_DIAGNOSTIC_ONLY','temporal_inputs_supported':False,
            'hdr_supported':False,'image_quality_verified':False,'game_runtime_ready':False,
            'training_started':False,'output_sha256':digest(raw_output)}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(report['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--height',type=int,default=128)
    p.add_argument('--input',type=Path)
    a=p.parse_args(); run(a.output,a.width,a.height,a.input)
