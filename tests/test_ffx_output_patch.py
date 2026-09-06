import json
import struct
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_ffx_output_patch import analyze, PATCH


def fixture(root, live=True):
    root.mkdir(); output=root/'output_patch'; output.mkdir(); width,height=12,10
    before=b''.join(struct.pack('<4e',x/12,y/10,(x+y)/22,1) for y in range(height) for x in range(width))
    after=bytearray(before)
    for y in range(8):
        for x in range(8): after[(y*width+x)*8:(y*width+x+1)*8]=PATCH
    metadata=dict(width=width,height=height,format=10,raw_bytes=len(after),fence_completed=True,write_back_performed=True,
                  replacement_pixels_supplied=True,patch_rect=[0,0,8,8],game_frame_capture_verified=live,nr_verified=False)
    (output/'boundary_before.raw').write_bytes(before);(output/'boundary_output.raw').write_bytes(after)
    (output/'metadata.json').write_text(json.dumps(metadata)); common=dict(window=1)
    events=[dict(common,event='boundary_output_arm',write_back_performed=True,replacement_pixels_supplied=True),
            dict(common,event='output_boundary_return',batch_supported=True,lifecycle_match=True,original_callback_returned=True,original_forward_calls=1,capture_authorized=False),
            dict(common,event='boundary_output_recorded',isolated_host_only=not live,write_back_performed=True,replacement_pixels_supplied=True,game_frame_capture_verified=False),
            dict(common,event='boundary_output_submitted',isolated_host_only=not live,write_back_performed=True,replacement_pixels_supplied=True,game_frame_capture_verified=False,submitted_identity_matches=1,identity_aliases=1,closed_identity_aliases=1,raw_pointer_match=True),
            dict(common,event='boundary_output_complete',**metadata)]
    (root/'session.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
    (root/'provenance.json').write_text(json.dumps(dict(exit_code=0,before={'a':'h'},after={'a':'h'},game_files_deployed=[],observation_deferred=True)))
    return events,metadata,before,after


def test_live_and_isolated_pass(tmp_path):
    fixture(tmp_path/'live'); fixture(tmp_path/'isolated',False)
    assert analyze(tmp_path/'live')['status']=='GAME_OUTPUT_PATCH_PASS'
    assert analyze(tmp_path/'isolated',False)['status']=='ISOLATED_OUTPUT_PATCH_PASS'


@pytest.mark.parametrize('mutation,match',[('outside','outside patch'),('inside','patch pixel'),('truncate','byte count'),('flag','provenance'),('nan','non-finite'),('tamper','session provenance')])
def test_failures(tmp_path,mutation,match):
    root=tmp_path/'run';events,_,before,after=fixture(root); path=root/'output_patch'/'boundary_output.raw'
    if mutation=='outside': data=bytearray(after);data[-1]^=1;path.write_bytes(data)
    elif mutation=='inside': data=bytearray(after);data[0]^=1;path.write_bytes(data)
    elif mutation=='truncate': path.write_bytes(after[:-2])
    elif mutation=='flag': events[-2]['replacement_pixels_supplied']=False;(root/'session.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
    elif mutation=='nan': data=bytearray(after);data[0:2]=struct.pack('<H',0x7e00);path.write_bytes(data)
    else: (root/'provenance.json').write_text(json.dumps(dict(exit_code=0,before={'a':'h'},after={'a':'x'},game_files_deployed=[],observation_deferred=True)))
    with pytest.raises(ValueError,match=match): analyze(root)
