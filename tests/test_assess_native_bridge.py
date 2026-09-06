import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from assess_native_bridge import assess,ORIGINAL_SHA


def fixture():
    frames=[{'frame_id':i,'input_sha256':str(i%2)*64,'output_sha256':str(2+i%2)*64,
             'input_exact':True,'consumer_exact':True,'finite':True,'frame_submit_and_wait_ms':100.} for i in range(12)]
    return {'pass':True,'normal_exit':True,'returncode':0,'host_timeout':False,
            'probe':{'mode':'network','original_model_sha256':ORIGINAL_SHA,'backend':'pytorch_rocm',
                     'cpu_neural_fallback':False,'frames':frames,'iterations':12,'output_sha256':'3'*64,
                     'alternating_outputs_consistent':True,'same_candidate_staged_output_exact':True,
                     'allocated_after_release_bytes':0,'reserved_after_release_bytes':0,
                     'source_files_changed_during_probe':[],
                     'transport':{'same_adapter_luid_checked':True,'separate_input_output_fences':True}}}


def test_standalone_pass_cannot_accept_game_or_hdr():
    report=fixture();result=assess(report,report['probe']['frames'],'3'*64)
    assert not result['failures'] and not result['game_integration_accepted']
    assert not result['pre_hud_visible_output_accepted'] and not result['hdr_accepted']
    assert result['last_frame']['input_sha256']=='1'*64


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(returncode=1),lambda r:r.update(host_timeout=True),
    lambda r:r['probe'].update(mode='transport'),lambda r:r['probe'].update(cpu_neural_fallback=True),
    lambda r:r['probe'].update(original_model_sha256='wrong'),
    lambda r:r['probe']['frames'][5].update(consumer_exact=False),
    lambda r:r['probe']['frames'][4].update(output_sha256='stale'),
    lambda r:r['probe'].update(allocated_after_release_bytes=1)])
def test_incomplete_or_stale_evidence_fails(mutation):
    report=fixture();journal=copy.deepcopy(report['probe']['frames']);mutation(report)
    assert assess(report,journal,'3'*64)['failures']
