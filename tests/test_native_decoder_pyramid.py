import sys
from pathlib import Path
import pytest
pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_decoder_pyramid import RecoveredDecoderPyramid,stage_role


def test_full_decoder_stage_roles_are_explicit():
    roles={b:stage_role(b) for b in range(48,70)}
    assert [b for b,(c,r) in roles.items() if r=='upsample']==[48,56,62,66]
    assert [b for b,(c,r) in roles.items() if r=='outview']==[55,61,65,69]
    with pytest.raises(ValueError):
        stage_role(70)


def test_missing_decoder_blocks_cannot_be_substituted():
    with pytest.raises(ValueError):
        RecoveredDecoderPyramid({}, {})
