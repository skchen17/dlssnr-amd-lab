"""Validate game-shaped synthetic depth-plane FFX capture; not actual game pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_ffx_dispatch_probe import analyze_provider


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_surface(resource,raw,expected,w,h,bpp,copy_format,source_format):
    required=dict(width=w,height=h,dxgi_format=copy_format,source_dxgi_format=source_format,
                  plane_count=2 if source_format==20 else 1,copied_subresource=0,barrier_subresource=0,
                  row_bytes=w*bpp,raw_bytes=w*h*bpp,ffx_state_restored=2 if resource['role']=='output' else 12)
    if any(resource.get(k)!=v for k,v in required.items()):
        raise ValueError('plane/format/state/shape metadata mismatch')
    pitch=resource.get('row_pitch',0)
    if type(pitch) is not int or pitch<w*bpp or pitch%256:
        raise ValueError('invalid padded row pitch')
    if len(raw)!=w*h*bpp or raw!=expected:
        raise ValueError('captured pixels differ from independent producer/output')
    return hashlib.sha256(raw).hexdigest()


def analyze_pair(base,observed,layout='observed'):
    a,b=analyze_provider(base),analyze_provider(observed)
    if a['output_sha256']!=b['output_sha256']:
        raise ValueError('capture changed original FFX output')
    ma,mb=read(base/'manifest.json'),read(observed/'manifest.json')
    for key in ('requested_provider','requested_provider_id','adapter_vendor','adapter_device','render_size','output_size'):
        if key not in ma or ma[key]!=mb.get(key):
            raise ValueError('provider/device/dimensions mismatch')
    if ma.get('depth_plane_mode') is not True or ma.get('capture_enabled') is not False:
        raise ValueError('baseline not independent depth-plane mode')
    required=dict(depth_plane_mode=True,capture_enabled=True,capture_completed=3,capture_pending=0,capture_dropped=1,
                  capture_invalid_rejected=15,depth_invalid_rejected=9,capture_early_poll_checks=5,
                  capture_byte_budget_rejected=True,stencil_mismatched_bytes=0,debug_errors=0)
    if any(mb.get(k)!=v for k,v in required.items()):
        raise ValueError('native depth capture lifecycle/gates failed')
    rw,rh=mb['render_size'];ow,oh=mb['output_size']
    dimensions={'observed':(1552,872,2342,1317),'aligned':(1536,864,2304,1296)}
    if layout not in dimensions or (rw,rh,ow,oh)!=dimensions[layout]:
        raise ValueError('test does not cover requested dimensions')
    for directory in ('depth_policy_rejected','stale_context_rejected','capture_byte_budget'):
        if list((observed/directory).iterdir()):
            raise ValueError('rejected contract emitted artifacts')
    if {p.name for p in (observed/'capture').iterdir()}!={'0','1','2'}:
        raise ValueError('capture budget not respected')
    stencil=bytes([0xa7])*(rw*rh)
    for i in range(4):
        for run in (base,observed):
            if (run/f'stencil_{i}.raw').read_bytes()!=stencil:
                raise ValueError('stencil changed or wrong stencil plane')
        if (base/f'input_{i}.raw').read_bytes()!=(observed/f'input_{i}.raw').read_bytes():
            raise ValueError('input control mismatch')
    hashes={};total=0
    specs={'color':(rw,rh,8,10,10),'depth':(rw,rh,4,39,20),'motion':(rw,rh,4,34,34),
           'reactive':(rw,rh,1,61,61),'output':(ow,oh,8,10,10)}
    for i in range(3):
        directory=observed/'capture'/str(i);m=read(directory/'manifest.json')
        required=dict(frame=i,fence_completed=True,fence_value=i+1,context_create_flags=169,requested_provider_id=mb['requested_provider_id'],
                      context_max_render=[rw,rh],context_max_upscale=[ow,oh],render_size=[rw,rh],
                      upscale_size=[0,0] if i in (0,2) else [ow,oh],effective_upscale_size=[ow,oh],reset=True,
                      motion_scale=[-rw/2,rh/2],game_frame=False,caller_validated_synthetic_host=True)
        if any(m.get(k)!=v for k,v in required.items()):
            raise ValueError('current context/default/frame metadata mismatch')
        resources=m.get('resources',[])
        if len(resources)!=5 or {r['role'] for r in resources}!=set(specs):
            raise ValueError('unexpected/missing captured surfaces')
        for r in resources:
            role=r['role'];raw=(directory/(role+'.raw')).read_bytes()
            if role=='color':expected=(base/f'input_{i}.raw').read_bytes()
            elif role=='output':expected=(base/f'output_{i}.raw').read_bytes()
            elif role=='depth':expected=np.full((rh,rw),.5,dtype='<f4').tobytes()
            else:expected=bytes(rw*rh*(4 if role=='motion' else 1))
            hashes[f'{i}/{role}']=verify_surface(r,raw,expected,*specs[role]);total+=len(raw)
    return dict(status='DEPTH_CAPTURE_PASS',layout=layout,render_size=[rw,rh],output_size=[ow,oh],on_off_outputs_equal=True,stencil_unchanged_all_four_frames=True,
                captured_frames=3,captured_resources=15,captured_raw_bytes=total,artifact_sha256=hashes,
                context_default_dimensions_verified=True,unknown_depth_policy_rejected=True,stale_context_rejected=True,
                debug_errors=0,debug_warnings=mb.get('debug_warnings'),game_pixels_captured=False,game_resource_states_verified=False,
                full_nr_quality_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path)
    p.add_argument('--layout',choices=('observed','aligned'),default='observed');args=p.parse_args()
    suffix='_depth' if args.layout=='observed' else '_depth-aligned'
    report={}
    for n in ('fsr3','fsr4'):
        try:report[n]=analyze_pair(args.path/(n+suffix+'-base'),args.path/(n+suffix+'-capture'),args.layout)
        except (ValueError,KeyError,TypeError,OSError) as exc:report[n]=dict(status='DEPTH_CAPTURE_FAIL',error=str(exc))
    report['status']='PASS' if all(r['status']=='DEPTH_CAPTURE_PASS' for r in report.values()) else 'FAIL'
    (args.path/('depth_capture_analysis.json' if args.layout=='observed' else 'depth_aligned_capture_analysis.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['status']=='PASS' else 1)
