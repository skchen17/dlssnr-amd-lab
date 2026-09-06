import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_preblock import positional_noise,reflect_indices,single_color_features,RecoveredSingleColorPreblock


def test_noise_uses_full_coordinates_and_seed_not_independent_tiles():
    a=positional_noise(128,96,0,'cpu')
    b=positional_noise(64,48,0,'cpu')
    assert torch.equal(a[:48,:64],b)
    assert torch.isfinite(a).all()
    assert not torch.equal(a,positional_noise(128,96,1,'cpu'))


def test_reflection_is_explicit_not_downscaling():
    assert reflect_indices(4,8,'cpu').tolist()==[0,1,2,3,2,1,0,1]
    assert reflect_indices(1,4,'cpu').tolist()==[0,0,0,0]
    with pytest.raises(ValueError):
        reflect_indices(4,3,'cpu')


def test_no_history_branch_uses_actual_current_color():
    color=torch.full((8,8,4),.75).half().requires_grad_()
    features=single_color_features(color,8,8,0,color_scale=.125,conditioning=(0,1,1,-1,-1))
    assert features.shape==(8,8,16)
    assert torch.equal(features[...,4:7],features[...,7:10])
    assert torch.all(features[...,4:10]==.03125)
    features[...,4:10].float().sum().backward()
    assert torch.all(color.grad[...,:3]==.25)


def test_preblock_records_are_not_shape_guessed():
    with pytest.raises(ValueError):
        RecoveredSingleColorPreblock(bytes(21695))
