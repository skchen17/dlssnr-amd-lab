"""Metadata-only consumer-location evidence; never proves a shader read or safe split."""
import argparse
import json
from pathlib import Path


def analyze(events):
    tails=[e for e in events if e.get('event')=='output_boundary_tail']
    valid=bool(tails) and all(e.get('metadata_only') is True and e.get('close_succeeded') is True for e in tails)
    calls=[sum(e.get(k,0) for k in ('draw','draw_indexed','dispatch','indirect','copy_texture','copy_resource')) for e in tails]
    return {'status':'BOUNDARY_TAIL_METADATA_OBSERVED' if valid else 'INSUFFICIENT_EVIDENCE',
            'samples':len(tails),'same_list_gpu_commands_after_boundary':any(c>0 for c in calls),
            'per_sample':[{'window':e.get('window'),'sample':e.get('sample'),'list':e.get('list'),
                           'generation':e.get('generation'),**{k:e.get(k,0) for k in
                             ('draw','draw_indexed','dispatch','indirect','copy_texture','copy_resource','target_copy_reads','target_copy_writes')}} for e in tails],
            'shader_reads_of_target_proven':False,'hud_boundary_proven':False,
            'safe_command_list_split_implemented':False,'game_native_network_accepted':False,
            'interpretation':'Post-boundary commands exist before Close; list-end placement is not justified by these counts. Descriptor/shader dependencies and state-preserving split remain unverified.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze([json.loads(line) for line in a.log.read_text().splitlines() if line.strip()])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
