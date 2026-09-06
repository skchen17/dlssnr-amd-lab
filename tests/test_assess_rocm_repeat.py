import copy
import pytest
from scripts.assess_rocm_repeat import assess


def fixture():
    frames=[{'iteration':i,'output_sha256':'A'*64,'host_stage_sum_ms':2100.,'allocated_bytes':400000000+i%2*65536}
            for i in range(100)]
    report={'pass':True,'returncode':0,'normal_exit':True,'host_timeout':False,
            'probe':{'name':'whole_frame640','backend':'pytorch_rocm','architecture':'gfx1201',
                     'cpu_neural_fallback':False,'all_encoder_skips_generated':True,'rtx_intermediate_inputs':False,
                     'iterations':100,'frames':frames,'stages':[{'block':b} for b in range(71)],
                     'monitoring':{'periodic_traceback_timer_enabled':False},
                     'allocated_after_release_bytes':0,'reserved_after_release_bytes':0,
                     'implementation_source_sha256':{'model.py':'hash'},'source_files_changed_during_probe':[],
                     'output_sha256':'A'*64,'identical_input_repeat_exact':True}}
    return report,copy.deepcopy(frames)


def check(report,journal):
    return assess(report,journal,source_snapshot={'model.py':'hash'},output_sha256='A'*64)


def test_static_pass_does_not_accept_game_hdr_temporal_or_root_cause():
    r=check(*fixture())
    assert r['status']=='STATIC_GPU_REPEAT_PASS'
    for key in ('root_cause_proven','temporal_stability_accepted','hdr_accepted','game_runtime_accepted','realtime_4k60_accepted'):
        assert r[key] is False


@pytest.mark.parametrize('failure',['crash','short','journal','hash','source','growth','teacher','timer','time'])
def test_repeated_probe_cannot_hide_missing_or_failed_evidence(failure):
    r,j=fixture(); p=r['probe']
    if failure=='crash': r['returncode']=3221225477
    elif failure=='short': p['iterations']=12; p['frames']=p['frames'][:12]; j=j[:12]
    elif failure=='journal': j.pop()
    elif failure=='hash': p['frames'][3]['output_sha256']='B'*64
    elif failure=='source': p['source_files_changed_during_probe']=['model.py']
    elif failure=='growth':
        for frame in p['frames']: frame['allocated_bytes']+=frame['iteration']*1048576
        j=copy.deepcopy(p['frames'])
    elif failure=='teacher': p['rtx_intermediate_inputs']=True
    elif failure=='timer': p['monitoring']['periodic_traceback_timer_enabled']=True
    else: p['frames'][0]['host_stage_sum_ms']=float('nan')
    assert check(r,j)['status']=='NOT_ACCEPTED'
