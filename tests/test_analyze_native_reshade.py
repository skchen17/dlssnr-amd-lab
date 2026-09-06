import json
import numpy as np
from scripts.analyze_native_reshade import analyze


def fixture(root):
    for name,value in [('host.json',{'width':2,'height':2,'d3d12_errors':0}),('supervisor.json',{'exit_code':0,'host_timeout':False,'network_frames':1})]:
        (root/name).write_text(json.dumps(value))
    (root/'addon.jsonl').write_text(json.dumps({'event':'frame_written','sequence':1,'present_index':2})+'\n'+json.dumps({'event':'bounded_session_complete'}))
    np.ones((2,2,4),dtype='<f2').tofile(root/'audit_output1.rgba16f')
    np.full((2,2,4),255,dtype='u1').tofile(root/'frame2.rgba8')


def test_exact_standalone_is_not_game_or_realtime_acceptance(tmp_path):
    fixture(tmp_path);r=analyze(tmp_path)
    assert r['pass'] and not r['game_accepted'] and not r['realtime_accepted'] and not r['hdr_accepted']


def test_visible_buffer_difference_fails_even_if_lower_half_matches(tmp_path):
    fixture(tmp_path);data=bytearray((tmp_path/'frame2.rgba8').read_bytes());data[0]=0;(tmp_path/'frame2.rgba8').write_bytes(data)
    r=analyze(tmp_path);assert not r['pass'] and r['comparisons'][0]['lower_half_exact']


def test_gpu_error_fails_even_when_pixels_match(tmp_path):
    fixture(tmp_path);(tmp_path/'host.json').write_text(json.dumps({'width':2,'height':2,'d3d12_errors':1}))
    assert not analyze(tmp_path)['pass']


def test_ten_bit_packed_output_is_compared_in_ten_bit_units(tmp_path):
    fixture(tmp_path)
    (tmp_path/'host.json').write_text(json.dumps({'width':2,'height':2,'d3d12_errors':0,'format':24}))
    np.full((2,2),0xffffffff,dtype='<u4').tofile(tmp_path/'frame2.rgba8')
    assert analyze(tmp_path)['pass']
    packed=np.full((2,2),0xffffffff,dtype='<u4');packed[0,0]-=1;packed.tofile(tmp_path/'frame2.rgba8')
    r=analyze(tmp_path);assert not r['pass'] and r['comparisons'][0]['max_error']==1


def test_unknown_encoding_is_not_assumed_eight_bit(tmp_path):
    import pytest
    fixture(tmp_path)
    (tmp_path/'host.json').write_text(json.dumps({'width':2,'height':2,'d3d12_errors':0,'format':10}))
    with pytest.raises(ValueError):analyze(tmp_path)
