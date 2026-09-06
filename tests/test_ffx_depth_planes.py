import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('depth_analysis',Path(__file__).parents[1]/'scripts/analyze_ffx_depth_planes.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.mark.parametrize('plane',[0,1])
def test_pattern_and_corruption(tmp_path,plane):
    w,h=7,5;bpp=1 if plane else 4;name=f'{w}x{h}_plane{plane}.raw'
    a=np.full((h,w),0xa7 if plane else .25,dtype='u1' if plane else '<f4')
    a[h//3:2*h//3,w//3:2*w//3]=0x3c if plane else .75
    r=dict(width=w,height=h,plane=plane,source_dxgi_format=20,plane_count=2,copy_format=60 if plane else 39,
           file=name,row_pitch=256,row_bytes=w*bpp,raw_bytes=w*h*bpp,footprint_bytes=256*(h-1)+w*bpp)
    (tmp_path/name).write_bytes(a.tobytes())
    assert module.verify_plane(tmp_path,r)['raw_bytes']==w*h*bpp
    a[0,0]=0
    (tmp_path/name).write_bytes(a.tobytes())
    with pytest.raises(ValueError,match='known-value'):
        module.verify_plane(tmp_path,r)
