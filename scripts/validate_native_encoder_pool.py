"""Original block30 projection/pool/padded final-head CPU boundary checks."""
import argparse
import json
from pathlib import Path
import torch
from native_capture_fixture import CaptureCases,OriginalRecords
from native_split_swin512 import SplitProjection512
from native_split_image512 import pool_to_bottleneck
from native_packed_swin import gather_packed,scatter_packed,logical_indices
from native_multiscale import pack_image,unpack_image
from native_transition_projections import EncoderFinalProjection
from native_swin_torch import decode_e4,encode_e4,quantize_e4
from native_execution_policy import execution_policy
from run_swin1h_native_validation import compare,digest


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    originals=OriginalRecords()
    cases=CaptureCases('results/20260831_210000_split16h_full_resource_exact_state/cases')
    ox,oy,h,w=cases.integers(55,56,4)
    print('geometry',w,h,ox,oy,flush=True)
    size=w*h*512
    source=cases.asset(55,'input')[:size]
    residual=cases.asset(55,'input2')[:size]
    originals.bind(30,3,cases.specs[55]['weight_view_offset'])
    originals.bind(30,4,cases.specs[56]['weight_view_offset'])
    model=SplitProjection512(originals.get(30,3),kind='attention_projection')
    _,index=logical_indices(512)
    windows,mapping=gather_packed(decode_e4(source),w,h,512,ox,oy)
    skip,_=gather_packed(decode_e4(residual),w,h,512,ox,oy)
    with torch.no_grad(),execution_policy('native_fp16'):
        logical=model.unquantized(windows[:,index],skip[:,index])
        packed=scatter_packed(logical,mapping,w,h)
        pooled=pool_to_bottleneck(unpack_image(packed,w,h,512))
        projected=EncoderFinalProjection(originals.get(30,4))(pooled)
    reports=[]
    for label,value,ref in [
        ('skip',quantize_e4(packed),cases.asset(55,'output_reference')[:size]),
        ('pool',pack_image(pooled),cases.asset(55,'extra_output_reference')[:pooled.numel()]),
        ('projection',pack_image(projected),cases.asset(56,'output_reference')[:projected.numel()])]:
        raw=encode_e4(value).numpy().tobytes()
        (output/f'{label}.e4').write_bytes(raw)
        reports.append({'stage':label,'comparison':compare(ref,raw),'output_sha256':digest(raw)})
    report={'backend':'cpu_reference_only','native_graph_complete':False,'results':reports}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
