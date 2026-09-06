"""Read-only standalone texture-loop evidence gate. Never accepts game integration."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from native_model_package import ORIGINAL_SHA


def assess(report,journal,output_sha):
    p=report.get('probe',{});frames=p.get('frames',[])
    pairs={(f.get('input_sha256'),f.get('output_sha256')) for f in frames}
    checks={
        'normal_process':report.get('pass') is True and report.get('normal_exit') is True
                         and report.get('returncode')==0 and report.get('host_timeout') is False,
        'original_native_network':p.get('mode')=='network' and p.get('original_model_sha256')==ORIGINAL_SHA
                                 and p.get('backend')=='pytorch_rocm' and p.get('cpu_neural_fallback') is False,
        'complete_journal':len(frames)>=12 and frames==journal and len(frames)==p.get('iterations')
                           and [f.get('frame_id') for f in frames]==list(range(len(frames))),
        'transport_exact':bool(frames) and all(f.get('input_exact') is True and f.get('consumer_exact') is True
                                              and f.get('finite') is True for f in frames),
        'alternation':len(pairs)==2 and len({i for i,o in pairs})==2 and len({o for i,o in pairs})==2
                      and p.get('alternating_outputs_consistent') is True,
        'baseline_preserved':p.get('same_candidate_staged_output_exact') is True,
        'last_output_saved':bool(frames) and output_sha==frames[-1].get('output_sha256')==p.get('output_sha256'),
        'release':p.get('allocated_after_release_bytes')==0 and p.get('reserved_after_release_bytes')==0,
        'source_unchanged_during_probe':p.get('source_files_changed_during_probe')==[],
        'gpu_sync_identity':p.get('transport',{}).get('same_adapter_luid_checked') is True
                            and p.get('transport',{}).get('separate_input_output_fences') is True,
    }
    # Frame-level input/output identities are authoritative. Older v3 reports
    # paired the FIRST input at top level with the LAST saved output; preserved.
    return {'status':'NATIVE_STANDALONE_TEXTURE_LOOP_PASS' if all(checks.values()) else 'NOT_ACCEPTED',
            'checks':checks,'failures':[k for k,v in checks.items() if not v],
            'last_frame':frames[-1] if frames else None,
            'warm_host_frame_median_ms':statistics.median(f['frame_submit_and_wait_ms'] for f in frames[1:]) if len(frames)>1 else None,
            'game_integration_accepted':False,'pre_hud_visible_output_accepted':False,
            'hdr_accepted':False,'realtime_4k60_accepted':False,
            'scope':'standalone same-process shared GPU buffers; alternating fixed fixtures; CPU diagnostic endpoints'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.manifest.read_bytes();root=a.manifest.parent
    report=json.loads(raw);journal=[json.loads(line) for line in (root/'completed_frames.jsonl').read_text().splitlines()]
    result=assess(report,journal,hashlib.sha256((root/'output.rgba16f').read_bytes()).hexdigest().upper())
    result['manifest_sha256']=hashlib.sha256(raw).hexdigest().upper();result['manifest']=str(a.manifest)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result));raise SystemExit(0 if not result['failures'] else 2)
