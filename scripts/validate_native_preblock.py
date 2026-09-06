"""Single-color original preblock check against the historical zero-color run."""
import argparse
import json
from pathlib import Path
import zipfile
import torch
from native_capture_fixture import whole_frame_fixture
from native_preblock import RecoveredSingleColorPreblock
from native_execution_policy import execution_policy
from native_swin_torch import encode_e4
from run_swin1h_native_validation import digest,compare


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    records,_,settings=whole_frame_fixture()
    model=RecoveredSingleColorPreblock(records[(0,0)])
    with torch.no_grad(),execution_policy('native_fp16'):
        values=model(torch.zeros(360,640,4).half(),640,384,0,**settings)
    archive=Path('../_full_graph_reference_result_20260831_181952(1).zip')
    comparisons=[]
    with zipfile.ZipFile(archive) as z:
        if z.read('n0_weights_before.raw')[:21696]!=records[(0,0)]:
            raise ValueError('original preblock record differs')
        for name,path in [('skip','n0_scratch_after.raw'),('outview','n0_output_after.raw')]:
            actual=encode_e4(values[name]).numpy().tobytes(); ref=z.read(path)
            (output/f'{name}.e4').write_bytes(actual)
            comparisons.append({'name':name,'comparison':compare(ref,actual),
                                'reference_sha256':digest(ref),'output_sha256':digest(actual)})
    report={'backend':'cpu_reference_only','input':'historical_zero_rgba16f_control',
            'frame_seed':0,'conditioning':settings,'results':comparisons,
            'temporal_inputs_supported':False,'native_graph_complete':False,'image_quality_verified':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
