"""Independent host readback comparison. Not game/HDR/realtime acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(root):
    root=Path(root)
    host=json.loads((root/'host.json').read_text(encoding='utf-8-sig'))
    supervisor=json.loads((root/'supervisor.json').read_text(encoding='utf-8-sig'))
    events=[json.loads(s) for s in (root/'addon.jsonl').read_text().splitlines() if s.strip()]
    frames=[e for e in events if e['event']=='frame_written']
    failures=[e for e in events if e['event']=='failure']
    comparisons=[]
    w,h=host['width'],host['height']
    for f in frames[:2]:
        seq=f['sequence'];idx=f['present_index']
        neural=np.fromfile(root/f'audit_output{seq}.rgba16f',dtype='<f2').reshape(h,w,4)
        if host.get('format',28)==24:
            packed=np.fromfile(root/f'frame{idx}.rgba8',dtype='<u4').reshape(h,w)
            actual=np.stack([packed&1023,(packed>>10)&1023,(packed>>20)&1023,packed>>30],axis=-1)
            expected=np.rint(np.clip(neural.astype('f4'),0,1)*1023).astype('u2');expected[:,:,3]=3
        elif host.get('format',28)==28:
            actual=np.fromfile(root/f'frame{idx}.rgba8',dtype='u1').reshape(h,w,4)
            expected=np.rint(np.clip(neural.astype('f4'),0,1)*255).astype('u1');expected[:,:,3]=255
        else:raise ValueError('unknown output encoding')
        diff=np.abs(actual.astype('i2')-expected.astype('i2'))
        # ReShade splash can draw AFTER the neural pass. Never hide this in a full-frame pass.
        comparisons.append({'sequence':seq,'present_index':idx,'all_pixels_exact':bool(not diff.any()),
                            'max_error':int(diff.max()),'different_values':int(np.count_nonzero(diff)),
                            'lower_half_exact':bool(not diff[h//2:].any())})
    done=[e for e in events if e['event']=='bounded_session_complete']
    passed=bool(frames and comparisons and len(done)==1 and not failures and host['d3d12_errors']==0
                and supervisor['exit_code']==0 and not supervisor['host_timeout']
                and len(frames)==supervisor['network_frames'] and all(c['all_pixels_exact'] for c in comparisons))
    return {'status':'STANDALONE_RESHade_GPU_PRESENT_PASS' if passed else 'NOT_ACCEPTED',
            'pass':passed,'frames_written':len(frames),'d3d12_errors':host['d3d12_errors'],
            'comparisons':comparisons,'game_accepted':False,'realtime_accepted':False,'hdr_accepted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    r=analyze(a.run)
    with (a.run/'assessment.json').open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r));raise SystemExit(0 if r['pass'] else 1)
