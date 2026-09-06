"""Summarize bounded whole-frame evidence without hiding the failed repeat run."""
import argparse
import json
from pathlib import Path
from run_swin1h_native_validation import digest


def summarize(root,output):
    names=['20260906_rocm_whole_frame128_one','20260906_rocm_whole_frame640_one',
           '20260906_rocm_whole_frame128_twelve','20260906_rocm_whole_frame640_twelve']
    runs=[]
    for name in names:
        folder=root/name
        raw=(folder/'manifest.json').read_bytes(); report=json.loads(raw)
        probe=report.get('probe',{})
        phases=[json.loads(line)['phase'] for line in (folder/'phases.jsonl').read_text().splitlines()]
        completed=[p for p in phases if (p.startswith('whole_frame_iteration') and p.endswith('_block70_complete'))
                   or p=='whole_frame_block70_complete']
        runs.append({'manifest':str(folder/'manifest.json'),'manifest_sha256':digest(raw),
                     'pass':report['pass'],'returncode':report['returncode'],'normal_exit':report['normal_exit'],
                     'host_timeout':report['host_timeout'],'completed_frame_phase_count':len(completed),
                     'identical_input_repeat_exact':probe.get('identical_input_repeat_exact'),
                     'first_frame_host_stage_ms':probe.get('host_stage_sum_ms'),
                     'peak_allocated_bytes':probe.get('peak_allocated_bytes'),
                     'allocated_after_release_bytes':probe.get('allocated_after_release_bytes')})
    report={'status':'SINGLE_COLOR_GPU_CHAIN_EXECUTED_STABILITY_NOT_ACCEPTED',
            'gpu_tests_paused_pending_crash_review':True,'runs':runs,
            'exception_observation':'640x360 repeat process exited0xC0000005 at30s; traceback timer active; native stack in python312!PyCode_Addr2Line',
            'root_cause':'UNRESOLVED; stack-walker involvement observed, not proof GPU/runtime fault absent',
            'mitigation':'periodic asynchronous stack walking removed; fatal tracing, host timeout, PID/phase log retained; new per-frame hash journal',
            'mitigation_gpu_retested':False,'automatic_gpu_retry':False,
            'native_graph_complete':False,'single_color_chain_connected':True,
            'all_encoder_skips_generated':True,'rtx_intermediate_inputs':False,
            'full_original_temporal_color_contract':False,'hdr_verified':False,
            'game_runtime_ready':False,'realtime_4k60_accepted':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); summarize(args.results,args.output)
