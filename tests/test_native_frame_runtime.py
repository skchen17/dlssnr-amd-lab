import sys
from pathlib import Path
import pytest
pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_frame_runtime import FrameLifecycle,FrameSpec


def test_submission_retains_output_until_consumer_finishes():
    state=FrameLifecycle(); state.prepare(1); state.submit(1,0)
    for call in (lambda:state.prepare(2),state.close,lambda:state.submit(1,1),state.consume):
        with pytest.raises(RuntimeError): call()
    state.complete()
    with pytest.raises(RuntimeError): state.submit(1,1)
    state.consume(); state.submit(1,1); state.complete(); state.consume(); state.close()
    assert state.state=='CLOSED'


def test_device_failure_is_not_retry_or_free_permission():
    state=FrameLifecycle(); state.prepare(1); state.submit(1,0); state.fail()
    for call in (state.close,lambda:state.prepare(2),lambda:state.submit(1,1),state.complete,state.consume):
        with pytest.raises(RuntimeError): call()


def test_resize_and_cut_reset_reject_stale_frames():
    state=FrameLifecycle(); state.prepare(1); state.submit(1,4); state.complete(); state.consume()
    with pytest.raises(RuntimeError): state.submit(1,4)
    state.prepare(2)
    with pytest.raises(RuntimeError): state.submit(1,5)
    state.submit(2,0)


@pytest.mark.parametrize('spec',[FrameSpec(0,360,1),FrameSpec(640,360,0),
    FrameSpec(640,360,1,format='RGBA8'),FrameSpec(640,360,1,color_contract='HDR'),FrameSpec(3840,2160,1)])
def test_unsupported_contracts_reject_before_submission(spec):
    with pytest.raises(ValueError): spec.validate(1048576)


def test_odd_size_is_padding_not_rescaling():
    assert FrameSpec(641,361,1).validate(1048576)==(768,384)
