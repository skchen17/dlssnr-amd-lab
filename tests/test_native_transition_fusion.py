import sys
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

from native_transition_fusion import active_transition_fusion,transition_fusion


def test_transition_fusion_is_disabled_by_default():
    assert active_transition_fusion() is None
    with transition_fusion() as value:
        assert value is None and active_transition_fusion() is None


def test_encoder_transition_requires_explicit_dll():
    with pytest.raises(ValueError,match='explicit reviewed DLL'):
        with transition_fusion(encoder=True):pass


def test_transition_module_flags_are_explicit():
    with pytest.raises(ValueError,match='module flags'):
        from native_transition_fusion import EncoderTransitionFusion
        EncoderTransitionFusion('missing.dll',encoder=1)
