import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_gpu_protocol import descriptor,Commands,parse,worker_identity


def fields():return [1,10,20,1,641,129,5376,5376*129,720896,1,100,104,108,112,0,0]


def test_metadata_only_lease():
    assert descriptor(fields(),10,20)==(641,129,1)
    state=Commands(1);state.frame({'op':'frame','generation':1,'sequence':1,'seed':0})
    with pytest.raises(ValueError):state.close({'op':'close'})
    state.consumed({'op':'consumed','generation':1,'sequence':1});state.close({'op':'close'})


@pytest.mark.parametrize('index,value',[(0,2),(1,30),(2,30),(3,0),(4,8192),(6,5128),(7,1),(8,2**32),(10,0),(11,100),(15,1)])
def test_descriptor_rejects_unknown_or_unsafe_metadata(index,value):
    spec=fields();spec[index]=value
    with pytest.raises(ValueError):descriptor(spec,10,20)


@pytest.mark.parametrize('command',[
    {'op':'frame','generation':2,'sequence':1,'seed':0},
    {'op':'frame','generation':1,'sequence':2,'seed':0},
    {'op':'frame','generation':1,'sequence':1,'seed':-1},
    {'op':'frame','generation':1,'sequence':1,'seed':0,'pixels':'forbidden'}])
def test_frame_order_and_no_pixel_payload(command):
    with pytest.raises(ValueError):Commands(1).frame(command)


def test_parser_rejects_unbounded_duplicate_and_nonobject_data():
    for message in ('x'*8193,'{"op":1,"op":2}','[]'):
        with pytest.raises(ValueError):parse(message)


def test_venv_launcher_child_requires_os_parent_check():
    assert worker_identity(10,20,10,True)==20
    assert worker_identity(10,10,5,True)==10
    for identity in [(10,20,30,True),(10,20,10,False),(10,True,10,True),(10,0,10,True)]:
        with pytest.raises(ValueError):worker_identity(*identity)


def test_larger_geometry_requires_explicit_opt_in_and_pitched_budget():
    spec=fields();spec[4:9]=[1280,720,10240,10240*720,7405568]
    with pytest.raises(ValueError):descriptor(spec,10,20)
    assert descriptor(spec,10,20,max_width=1280,max_height=720)==(1280,720,1)
    for kw in ({'max_width':8192},{'max_height':True},{'max_width':0}):
        with pytest.raises(ValueError):descriptor(spec,10,20,**kw)


def test_absolute_heap_budget_is_not_unbounded_with_larger_frame_policy():
    spec=fields();spec[8]=129*1024*1024
    with pytest.raises(ValueError):descriptor(spec,10,20,max_width=3840,max_height=2160)
