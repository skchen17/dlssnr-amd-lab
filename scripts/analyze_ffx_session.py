"""Verify automatic session capture against independent real-provider outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_ffx_dispatch_probe import analyze_provider


def check_events(events):
    if any(e.get('event')=='failure' for e in events):raise ValueError('session failure event')
    names=['context_create','candidate','capture_recorded_inputs','capture_recorded_output','capture_submitted','capture_complete']
    groups={n:[(i,e) for i,e in enumerate(events) if e.get('event')==n] for n in names}
    if any(len(groups[n])!=1 for n in names):raise ValueError('missing/duplicate lifecycle stages')
    positions=[groups[n][0][0] for n in names]
    if positions!=sorted(positions):raise ValueError('lifecycle event ordering')
    context,candidate,inputs,output,submission,done=[groups[n][0][1] for n in names]
    if candidate.get('context_known') is not True or candidate.get('recording_gate_ready') is not True or candidate.get('reset_observed') is not True:
        raise ValueError('unverified context/list state')
    if candidate.get('context')!=context.get('context') or candidate.get('epoch')!=context.get('epoch') or inputs.get('epoch')!=context.get('epoch'):
        raise ValueError('stale context epoch')
    if candidate.get('generation',0)<1 or submission.get('generation')!=candidate['generation'] or inputs.get('ticket')!=submission.get('ticket'):
        raise ValueError('submission/list generation mismatch')
    if output.get('original_status')!=0 or done.get('frames')!=1 or done.get('fence_completed') is not True:
        raise ValueError('original status/fence completion')
    roles=candidate.get('resources',[])
    if {r.get('role') for r in roles}!={'color','depth','motion','exposure','output'} or len(roles)!=5:raise ValueError('state role set')
    for r in roles:
        expected=8 if r['role']=='output' else 64
        if r.get('expected_state')!=expected or r.get('plane0_recorded_state')!=expected:raise ValueError('actual recorded plane state mismatch')


def analyze_pair(base,observed):
    a,b=analyze_provider(base),analyze_provider(observed)
    if a['output_sha256']!=b['output_sha256']:raise ValueError('session changed original output')
    m=json.loads((observed/'manifest.json').read_text());bm=json.loads((base/'manifest.json').read_text())
    if m.get('session_enabled') is not True or m.get('session_poll_status')!=1:raise ValueError('native session completion')
    for key in ('requested_provider_id','adapter_vendor','adapter_device'):
        if m.get(key)!=bm.get(key):raise ValueError('provider/adapter identity')
    events=[json.loads(line) for line in (observed/'session.jsonl').read_text().splitlines()];check_events(events)
    if {p.name for p in (observed/'capture').iterdir()}!={'0'}:raise ValueError('one-frame budget')
    frame=observed/'capture/0';c=json.loads((frame/'manifest.json').read_text())
    if c.get('game_frame') is not False or c.get('fence_completed') is not True or c.get('requested_provider_id')!=m['requested_provider_id'] or c.get('provider_selection_verified') is not True:raise ValueError('capture provenance')
    expected={'color':(base/'input_0.raw').read_bytes(),'output':(base/'output_0.raw').read_bytes(),
              'depth':np.full((180,320),.5,dtype='<f4').tobytes(),'motion':bytes(320*180*4),'exposure':np.array([1],dtype='<f4').tobytes()}
    hashes={}
    for role,raw in expected.items():
        actual=(frame/(role+'.raw')).read_bytes()
        if actual!=raw:raise ValueError('captured '+role+' differs')
        hashes[role]=hashlib.sha256(actual).hexdigest()
    return dict(status='SESSION_CAPTURE_PASS',on_off_output_equal=True,captured_frames=1,captured_resources=5,
                captured_raw_bytes=sum(map(len,expected.values())),artifact_sha256=hashes,automatic_submission_fence_verified=True,
                game_frame_capture_verified=False,dlss_nr_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path);args=p.parse_args()
    report={n:analyze_pair(args.path/n,args.path/(n+'_session')) for n in ('fsr3','fsr4')};report['status']='PASS'
    (args.path/'session_analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
