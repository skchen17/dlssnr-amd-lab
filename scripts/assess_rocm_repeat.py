"""Fail-closed review of a static-input ROCm repetition/lifecycle probe.

No GPU imports or execution. This does not accept temporal images, gameplay,
HDR, NVIDIA quality, or real-time performance from repeated identical input.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def assess(report, journal, *, minimum_frames=100, source_snapshot=None, output_sha256=None):
    p=report.get('probe',{})
    frames=p.get('frames',[])
    checks={
        'process_pass':report.get('pass') is True and report.get('returncode')==0
                       and report.get('normal_exit') is True and report.get('host_timeout') is False,
        'whole_color_probe':p.get('name') in ('whole_frame128','whole_frame640'),
        'native_gpu':p.get('backend')=='pytorch_rocm' and 'gfx1201' in p.get('architecture','')
                     and p.get('cpu_neural_fallback') is False,
        'no_teacher_activations':p.get('all_encoder_skips_generated') is True
                                 and p.get('rtx_intermediate_inputs') is False,
        'count':len(frames)>=minimum_frames and len(frames)==p.get('iterations') and len(journal)==len(frames),
        'ordered_complete_journal':bool(frames) and frames==journal
                                   and [f.get('iteration') for f in frames]==list(range(len(frames))),
        'all_stages': [s.get('block') for s in p.get('stages',[])]==list(range(71)),
        'timer_disabled':p.get('monitoring',{}).get('periodic_traceback_timer_enabled') is False,
        'release':p.get('allocated_after_release_bytes')==0 and p.get('reserved_after_release_bytes')==0,
        'source_snapshot':bool(source_snapshot) and p.get('implementation_source_sha256')==source_snapshot
                           and p.get('source_files_changed_during_probe')==[],
    }
    hashes=[f.get('output_sha256') for f in frames]
    checks['stable_output']=(bool(hashes) and isinstance(hashes[0],str) and len(hashes[0])==64
                              and len(set(hashes))==1 and hashes[0]==p.get('output_sha256')
                              and p.get('identical_input_repeat_exact') is True)
    checks['saved_output_hash']=output_sha256 is not None and output_sha256==p.get('output_sha256')
    times=[f.get('host_stage_sum_ms') for f in frames]
    time_valid=bool(times) and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in times)
    checks['finite_host_times']=time_valid
    warm=frames[1:] if len(frames)>1 else frames
    memory=[f.get('allocated_bytes') for f in warm]
    memory_valid=bool(memory) and all(isinstance(v,int) and v>=0 for v in memory)
    spread=max(memory)-min(memory) if memory_valid else None
    # Permit allocator-bin alternation, not unbounded per-frame growth. This
    # deliberately is a limited empirical gate, not a proof of no memory leaks.
    checks['bounded_frame_end_allocations']=spread is not None and spread<=1048576
    passed=all(checks.values())
    return {'status':'STATIC_GPU_REPEAT_PASS' if passed else 'NOT_ACCEPTED', 'checks':checks,
            'failures':[k for k,v in checks.items() if not v], 'frames':len(frames),
            'minimum_frames':minimum_frames,'output_sha256':p.get('output_sha256'),
            'warm_host_stage_median_ms':statistics.median(times[1:] or times) if time_valid else None,
            'warm_host_stage_max_ms':max(times[1:] or times) if time_valid else None,
            'frame_end_allocation_spread_bytes':spread,
            'root_cause_proven':False,'temporal_stability_accepted':False,'hdr_accepted':False,
            'game_runtime_accepted':False,'realtime_4k60_accepted':False,
            'scope':'identical color input/seed; offline original-weight candidate; host staged timings only'}


def run(manifest,output,minimum_frames):
    root=manifest.parent
    raw=manifest.read_bytes(); report=json.loads(raw)
    journal_path=root/'completed_frames.jsonl'
    journal=[json.loads(line) for line in journal_path.read_text().splitlines()] if journal_path.exists() else []
    snapshot=root/'implementation_sources_at_start.json'
    output_pixels=root/'output.rgba16f'
    result=assess(report,journal,minimum_frames=minimum_frames,
                  source_snapshot=json.loads(snapshot.read_bytes()) if snapshot.exists() else None,
                  output_sha256=hashlib.sha256(output_pixels.read_bytes()).hexdigest().upper() if output_pixels.exists() else None)
    result.update(manifest=str(manifest),manifest_sha256=hashlib.sha256(raw).hexdigest().upper())
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))
    return result['status']=='STATIC_GPU_REPEAT_PASS'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-frames',type=int,default=100)
    a=p.parse_args()
    if a.minimum_frames<1:
        p.error('positive minimum required')
    raise SystemExit(0 if run(a.manifest,a.output,a.minimum_frames) else 2)
