import copy
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('live_analysis',Path(__file__).parents[1]/'scripts/analyze_ffx_live_inspection.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

@pytest.fixture
def events():
    r=dict(present=True,width=320,height=180,dxgi_format=10,dimension=3,array_size=1,mips=1,samples=1,
           same_device=True,descriptor_matches=True,row_pitch=2560,row_bytes=2560,footprint_bytes=460800)
    resources=[dict(r,role=role) for role in ('color','depth','motion','output')]
    resources += [dict(role=role,present=False) for role in ('exposure','reactive','transparency')]
    return [dict(event='queue_hook',installed=True),dict(event='resources',sample_id=1,resources=resources,
            total_footprint_bytes=4*460800,all_descriptors_match=True,gpu_copies=0,resource_states_verified=False),
            dict(event='dispatch_return',sample_id=1,status=0),dict(event='execute',queue='0x123',queue_type=0,sample_ids=[1])]

def test_pass(events):
    r=module.analyze_events(events,1)
    assert r['status']=='LIVE_INSPECTION_PASS'
    assert r['game_runtime_ready'] is False and r['gpu_pixels_captured'] is False

@pytest.mark.parametrize('index',[0,1,2,3])
def test_missing_event(events,index):
    del events[index]
    assert module.analyze_events(events,1)['status']=='LIVE_INSPECTION_FAIL'

@pytest.mark.parametrize('field,value',[('same_device',False),('descriptor_matches',False),('mips',2),('row_pitch',2559),('footprint_bytes',1),('dxgi_format',999),('present',False)])
def test_bad_resource(events,field,value):
    events[1]['resources'][0][field]=value
    assert module.analyze_events(events,1)['status']=='LIVE_INSPECTION_FAIL'

def test_duplicate_submission(events):
    events.append(copy.deepcopy(events[3]))
    assert module.analyze_events(events,1)['status']=='LIVE_INSPECTION_FAIL'

def test_failed_return(events):
    events[2]['status']=1
    assert module.analyze_events(events,1)['status']=='LIVE_INSPECTION_FAIL'

def test_partial_log(tmp_path,events):
    path=tmp_path/'live.jsonl'
    path.write_text(json.dumps(events[0]))
    with pytest.raises(ValueError,match='incomplete'):
        module.read_log(path)

def test_no_samples():
    assert module.analyze_events([],1)['status']=='LIVE_INSPECTION_FAIL'
