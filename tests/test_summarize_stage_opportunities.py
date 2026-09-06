import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_stage_opportunities import family,summarize


def fixture():
    return {'all_outputs_exact':True,'runs':[{'block':b,'repeat':i,'host_submit_wait_ms':i+1} for b in range(71) for i in range(3)]}


def test_complete_stage_mapping():
    assert len([b for b in range(71) if family(b)=='c32'])==8
    assert family(39)=='decoder_projection'
    assert family(70)=='head'
    with pytest.raises(ValueError):family(71)


def test_summary_does_not_infer_fps():
    result=summarize(fixture(),[])
    assert len(result['stages'])==71
    assert result['stages'][0]['median_isolated_ms']==2
    assert not result['whole_frame_speedup_verified']


def test_missing_repeat_rejected():
    data=fixture();data['runs'].pop()
    with pytest.raises(ValueError):summarize(data,[])


def test_failed_core_rejected():
    with pytest.raises(ValueError):summarize(fixture(),[{'checks_pass':False}])
