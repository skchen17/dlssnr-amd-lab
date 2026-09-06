import importlib.util
from pathlib import Path
import sys
import pytest
scripts=Path(__file__).parents[1]/'scripts'
if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('session_analysis',scripts/'analyze_ffx_session.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.fixture
def events():
    roles=[dict(role=r,plane0_recorded_state=8 if r=='output' else 64,expected_state=8 if r=='output' else 64) for r in ('color','depth','motion','exposure','output')]
    return [dict(event='context_create',context='0x123',epoch=2),dict(event='candidate',context='0x123',epoch=2,context_known=True,recording_gate_ready=True,reset_observed=True,generation=3,resources=roles),
            dict(event='capture_recorded_inputs',epoch=2,ticket=1),dict(event='capture_recorded_output',original_status=0),
            dict(event='capture_submitted',generation=3,ticket=1),dict(event='capture_complete',frames=1,fence_completed=True)]

def test_valid(events):module.check_events(events)

@pytest.mark.parametrize('index',range(6))
def test_missing_stages(events,index):
    del events[index]
    with pytest.raises(ValueError):module.check_events(events)

@pytest.mark.parametrize('index,key,value',[(1,'context_known',False),(1,'reset_observed',False),(1,'epoch',1),(4,'generation',4),(5,'fence_completed',False),(3,'original_status',1)])
def test_unknown_stale_unfenced(events,index,key,value):
    events[index][key]=value
    with pytest.raises(ValueError):module.check_events(events)

def test_wrong_compute_state(events):
    events[1]['resources'][0]['plane0_recorded_state']=128
    with pytest.raises(ValueError,match='plane state'):module.check_events(events)

def test_bad_order(events):
    events[4],events[5]=events[5],events[4]
    with pytest.raises(ValueError,match='ordering'):module.check_events(events)
