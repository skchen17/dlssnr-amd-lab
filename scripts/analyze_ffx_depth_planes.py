"""Independently verify depth/stencil plane bytes against the probe's clear rectangles."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def verify_plane(root, record):
    w,h,plane = (record[k] for k in ('width','height','plane'))
    if type(w) is not int or type(h) is not int or not 1<=w<=8192 or not 1<=h<=8192 or plane not in (0,1):
        raise ValueError('invalid dimensions/plane')
    if record['source_dxgi_format'] != 20 or record['plane_count'] != 2 or record['copy_format'] != (39 if plane==0 else 60):
        raise ValueError('unexpected depth plane format')
    filename=f'{w}x{h}_plane{plane}.raw'
    if record['file']!=filename:
        raise ValueError('unexpected artifact filename')
    bpp=4 if plane==0 else 1
    raw=(root/filename).read_bytes()
    pitch=record['row_pitch']
    if len(raw)!=w*h*bpp or record['row_bytes']!=w*bpp or record['raw_bytes']!=len(raw) or pitch<w*bpp or pitch%256 or record['footprint_bytes']!=pitch*(h-1)+w*bpp:
        raise ValueError('layout mismatch')
    expected=np.full((h,w),.25 if plane==0 else 0xa7,dtype='<f4' if plane==0 else 'u1')
    expected[h//3:2*h//3,w//3:2*w//3]=.75 if plane==0 else 0x3c
    if raw!=expected.tobytes():
        raise ValueError('wrong plane, stride or known-value pixels')
    return dict(file=filename,sha256=hashlib.sha256(raw).hexdigest(),raw_bytes=len(raw))


def analyze(root):
    m=json.loads((root/'manifest.json').read_text())
    if m.get('status')!='DEPTH_PLANE_PASS' or m.get('debug_errors')!=0 or m.get('completed_fences')!=3 or m.get('adapter_vendor')!=4098 or m.get('mismatched_pixels')!=0:
        raise ValueError('native completion gate failed')
    warnings=(root/'d3d12_messages.log').read_text().splitlines()
    if len(warnings)!=3 or any(not s.startswith('2: ID3D12CommandList::ClearDepthStencilView: The clear values do not match') for s in warnings):
        raise ValueError('unexpected debug messages (only intentional clear-value performance warnings expected)')
    records=[json.loads(line) for line in (root/'planes.jsonl').read_text().splitlines()]
    expected={(w,h,p) for w,h in ((321,181),(1552,872),(2560,1440)) for p in (0,1)}
    if len(records)!=6 or {(r['width'],r['height'],r['plane']) for r in records}!=expected:
        raise ValueError('missing/duplicate test cases')
    artifacts=[verify_plane(root,r) for r in records]
    return dict(status='DEPTH_PLANE_ARTIFACT_PASS',artifacts=artifacts,debug_errors=0,
                expected_clear_value_performance_warnings=3,game_pixels_captured=False,game_resource_states_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path);args=p.parse_args()
    report=analyze(args.path)
    (args.path/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
