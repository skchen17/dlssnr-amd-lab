import importlib.util
import sys
from pathlib import Path
import pytest

scripts=Path(__file__).parents[1]/'scripts'
if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('depth_capture_analysis',scripts/'analyze_ffx_depth_capture.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.fixture
def depth():
    return dict(role='depth',width=5,height=3,dxgi_format=39,source_dxgi_format=20,plane_count=2,
                copied_subresource=0,barrier_subresource=0,row_bytes=20,raw_bytes=60,
                ffx_state_restored=12,row_pitch=512)

def verify(r,raw=bytes(60)):
    return module.verify_surface(r,raw,bytes(60),5,3,4,39,20)

def test_depth_plane_valid(depth):
    assert len(verify(depth))==64

@pytest.mark.parametrize('key,value',[('plane_count',1),('copied_subresource',1),('barrier_subresource',4294967295),
                                    ('source_dxgi_format',41),('dxgi_format',60),('row_pitch',20),
                                    ('ffx_state_restored',4),('width',4)])
def test_wrong_plane_state_or_layout(depth,key,value):
    depth[key]=value
    with pytest.raises(ValueError):verify(depth)

@pytest.mark.parametrize('raw',[bytes(59),bytes([1])+bytes(59)])
def test_wrong_capture_pixels(depth,raw):
    with pytest.raises(ValueError):verify(depth,raw)
